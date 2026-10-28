"""Replace Tau's summarizer compaction with the deterministic engine.

Install by copying this file into ``~/.tau/extensions/`` (user scope) or
``<cwd>/.tau/extensions/`` (project scope, needs ``--project-extensions``). To go
back to Tau's built-in compaction, delete it: nothing else changes, and the two
paths can be A/B'd on the same session log.

The engine derives a compaction summary from typed session structure -- roles,
content block types, tool names, JSON arguments, error flags -- and never asks a
model to write one. That buys three properties the summarizer path cannot offer:
no provider round trip, byte-identical output for the same entries, and every
omission recorded as an address instead of vanishing.

Two behaviors are deliberate:

* **QA gates the write.** The summary is checked against ground-truth probes
  before it is accepted. If a fact went missing, the engine declines and Tau's
  own summarizer runs, so a bug in this file degrades to the old behavior rather
  than to a lossy summary.
* **Thinking is erased, not summarized.** Reasoning blocks are counted for
  provenance and dropped. They are scratchwork, and re-ingesting them would
  re-inflate the very window being compacted.
"""

from __future__ import annotations

import dataclasses

from tau_agent.messages import AgentMessage
from tau_agent.types import JSONValue
from tau_coding.compaction import (
    CompactionEvent,
    ContextRow,
    CutBudget,
    compute_cut,
    count_erased_thinking_blocks,
    normalize_rows,
    utf8_bytes,
)
from tau_coding.compaction.projections import ProjectionResult, project_with_metadata
from tau_coding.compaction.qa import QaReport, qa_report
from tau_coding.compaction.render import (
    RenderedSummary,
    RenderOptions,
    format_timestamp,
    message_timestamp,
    render_summary_with_metadata,
)
from tau_coding.context_window import DEFAULT_COMPACTION_KEEP_RECENT_TOKENS
from tau_coding.extensions import (
    BeforeCompactEvent,
    CompactHookResult,
    ExtensionAPI,
    ExtensionContext,
)

ENGINE_NAME = "deterministic"
ENGINE_VERSION = 1

#: How many failed QA probes to carry in compaction details. Enough to diagnose,
#: bounded so a bad render cannot bloat the session log.
MAX_REPORTED_FAILURES = 8

#: QA scores worth showing in `/compact-engine status`.
_QA_STATUS_KEYS = ("probes", "content_score", "address_score")

#: Provenance from the most recent engine decision, for `/compact-engine status`.
#: Module state on purpose: the QA report is only computable at compaction time,
#: so the score has to be remembered rather than recomputed later.
_last_run: dict[str, JSONValue] | None = None


def _request_lines(instructions: str | None) -> tuple[str, ...]:
    """Turn a user's compaction focus into protected bullet lines."""
    if not instructions:
        return ()
    return tuple(f"- {line}" for line in instructions.splitlines() if line.strip())


def _provenance(
    event: BeforeCompactEvent,
    rows: list[ContextRow],
    cut_index: int,
    events: list[CompactionEvent],
    summary: str,
    rendered: RenderedSummary,
    projection: ProjectionResult,
    report: QaReport,
    retained_raw_tokens: int,
    budget: CutBudget | None,
) -> dict[str, JSONValue]:
    """Build the JSON blob stored beside the summary."""
    omitted = dataclasses.asdict(projection.omitted)
    details: dict[str, JSONValue] = {
        "engine": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "decision": "replace",
        "trigger": event.trigger,
        "summarized_entries": cut_index,
        "retained_entries": len(rows) - cut_index,
        "events": len(events),
        "thinking_blocks_erased": count_erased_thinking_blocks(rows[:cut_index]),
        "summary_bytes": utf8_bytes(summary),
        "clipped_bytes": (
            rendered.request_omitted_bytes
            + rendered.dialogue_omitted_bytes
            + rendered.carry_omitted_bytes
        ),
        "section_headers": list(rendered.section_headers),
        "omitted": {key: value for key, value in omitted.items()},
        "qa": {
            "probes": report.probe_count,
            "score": round(report.score, 4),
            "content_score": round(report.content_score, 4),
            "address_score": round(report.address_score, 4),
        },
        "retained_raw_tokens": retained_raw_tokens,
    }
    if budget is not None:
        details["budget"] = budget.to_json()
    return details


def run_engine(event: BeforeCompactEvent) -> CompactHookResult:
    """Decide this compaction from the addressed context entries alone.

    Returns ``action="continue"`` whenever the engine should not write: no cut
    exists, the window is empty, or QA found a silent loss. Declining is cheap --
    Tau's summarizer then does what it always did.
    """
    global _last_run
    rows: list[ContextRow] = [(entry.entry_id, entry.message) for entry in event.entries]
    keep_recent_tokens = (
        event.keep_recent_tokens
        if event.keep_recent_tokens is not None
        else DEFAULT_COMPACTION_KEEP_RECENT_TOKENS
    )
    cut = compute_cut(
        rows,
        keep_recent_tokens=keep_recent_tokens,
        context_window_tokens=event.context_window_tokens,
        tokens_before=event.tokens_before,
    )
    if cut is None:
        _last_run = {"engine": ENGINE_NAME, "decision": "no_cut", "trigger": event.trigger}
        return CompactHookResult(action="continue")

    events = normalize_rows(rows)
    cut_index = len(cut.summarized_rows)
    if cut_index == 0 or cut_index >= len(events):
        # Nothing to summarize, or nothing left to keep: Tau's own boundary is
        # the right answer, and this engine has no reason to override it.
        _last_run = {"engine": ENGINE_NAME, "decision": "empty_window", "trigger": event.trigger}
        return CompactHookResult(action="continue")

    window = events[:cut_index]
    retained = events[cut_index:]
    projection = project_with_metadata(window, retained)
    last_message: AgentMessage = rows[cut_index - 1][1]
    rendered = render_summary_with_metadata(
        projection.sections,
        RenderOptions(
            first_entry_id=rows[0][0],
            last_entry_id=rows[cut_index - 1][0],
            last_timestamp=format_timestamp(message_timestamp(last_message)),
            request_lines=_request_lines(event.instructions),
        ),
    )
    report = qa_report(events, cut_index, rendered.summary)
    if not report.ok:
        _last_run = {
            "engine": ENGINE_NAME,
            "decision": "qa_failed",
            "trigger": event.trigger,
            "probe_count": report.probe_count,
            "failures": [item.reason for item in report.failures[:MAX_REPORTED_FAILURES]],
        }
        return CompactHookResult(action="continue")

    details = _provenance(
        event,
        rows,
        cut_index,
        events,
        rendered.summary,
        rendered,
        projection,
        report,
        cut.retained_raw_tokens,
        cut.budget,
    )
    _last_run = dict(details)
    return CompactHookResult(
        action="replace",
        summary=rendered.summary,
        first_kept_entry_id=cut.first_kept_entry_id,
        details=details,
    )


def _before_compact(event: object, _context: ExtensionContext) -> CompactHookResult:
    """Answer the seam. Synchronous by design: no provider, no await."""
    if not isinstance(event, BeforeCompactEvent):
        return CompactHookResult(action="continue")
    return run_engine(event)


def _status(_args: str, _context: object) -> str:
    """Report what the engine last did, for `/compact-engine`."""
    if _last_run is None:
        return (
            "deterministic compaction: armed, no decision yet. "
            "It answers before_compact and declines on any QA failure."
        )
    lines = [f"deterministic compaction: decision={_last_run.get('decision')}"]
    for key in (
        "trigger",
        "summarized_entries",
        "retained_entries",
        "events",
        "thinking_blocks_erased",
        "summary_bytes",
        "clipped_bytes",
    ):
        if key in _last_run:
            lines.append(f"{key}: {_last_run[key]}")
    qa = _last_run.get("qa")
    if isinstance(qa, dict):
        shown = ", ".join(f"{key}={qa[key]}" for key in _QA_STATUS_KEYS if key in qa)
        lines.append(f"qa: {shown}")
    failures = _last_run.get("failures")
    if isinstance(failures, list) and failures:
        lines.append(f"failed probes: {len(failures)}")
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines)


def setup(tau: ExtensionAPI) -> None:
    """Subscribe to the compaction seam and expose the status command."""
    tau.on("before_compact", _before_compact)
    tau.register_command(
        "compact-engine",
        _status,
        description="Report the deterministic compaction engine's last decision.",
        usage="/compact-engine",
    )
