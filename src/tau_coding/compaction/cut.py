"""Closure-safe selection of the compaction cut in an active context.

Choosing where to summarize is the one compaction decision that can break a
session: providers reject a context that keeps a tool result whose call was
summarized away, or a call whose result was summarized away. This module never
proposes such a cut. It computes call/result spans structurally, requires every
span to lie wholly on one side of the boundary, and then satisfies the token
budget by moving the boundary -- never by relaxing the pairing rule.

Budget selection uses four ceilings and reports which one binds:

``continuity``
    Keep the recent tail plus room for the summary itself.
``occupancy``
    Bring context down to a target fraction of the model window.
``reserve``
    Leave enough headroom for the next response.
``reduction``
    Actually shrink this context, even when the window is misreported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, floor
from typing import Literal

from tau_agent.messages import (
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.types import JSONValue
from tau_coding.compaction.bounds import MAX_SUMMARY_BYTES
from tau_coding.compaction.events import ContextRow
from tau_coding.context_window import (
    DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    DEFAULT_COMPACTION_RESERVE_TOKENS,
    estimate_message_tokens,
    provider_context_tokens,
)

#: A summary is budgeted as raw text at the estimator's own characters-per-token.
SUMMARY_RAW_TOKEN_BUDGET = ceil(MAX_SUMMARY_BYTES / 4)
HARD_CEILING_SAFETY_RATIO = 0.9
MAX_PRECOMPACTION_RATIO = 0.95
DEFAULT_TARGET_OCCUPANCY = 0.8
MIN_TARGET_OCCUPANCY = 0.25
MAX_TARGET_OCCUPANCY = 0.85
_CALIBRATION_MIN_RAW_DELTA = 4_096
_CALIBRATION_MIN_RAW_RATIO = 0.1

type BindingConstraint = Literal["continuity", "occupancy", "reserve", "reduction"]


@dataclass(frozen=True, slots=True)
class CallResultSpan:
    """Row span of one tool call and its result."""

    first: int
    last: int
    has_call: bool
    has_result: bool


@dataclass(frozen=True, slots=True)
class CutBudget:
    """The ceilings, calibration, and outcome that produced one cut."""

    strategy: Literal["continuity", "tail"]
    context_window_tokens: int
    target_occupancy: float
    reserve_tokens: int
    keep_recent_tokens: int
    raw_tokens_before: int
    token_scale: float
    fixed_overhead_tokens: int
    retained_raw_tokens: int
    raw_tail_token_budget: int
    target_context_tokens: int
    continuity_target_tokens: int
    occupancy_ceiling_tokens: int
    safe_ceiling_tokens: int
    reduction_ceiling_tokens: int
    binding_constraint: BindingConstraint

    def to_json(self) -> dict[str, JSONValue]:
        """Return the budget as a JSON object for compaction details."""
        return {
            "strategy": self.strategy,
            "contextWindow": self.context_window_tokens,
            "targetOccupancy": self.target_occupancy,
            "reserveTokens": self.reserve_tokens,
            "keepRecentTokens": self.keep_recent_tokens,
            "rawTokensBefore": self.raw_tokens_before,
            "tokenScale": self.token_scale,
            "fixedOverheadTokens": self.fixed_overhead_tokens,
            "retainedRawTokens": self.retained_raw_tokens,
            "rawTailTokenBudget": self.raw_tail_token_budget,
            "targetContextTokens": self.target_context_tokens,
            "continuityTargetTokens": self.continuity_target_tokens,
            "occupancyCeilingTokens": self.occupancy_ceiling_tokens,
            "safeCeilingTokens": self.safe_ceiling_tokens,
            "reductionCeilingTokens": self.reduction_ceiling_tokens,
            "bindingConstraint": self.binding_constraint,
        }


@dataclass(frozen=True, slots=True)
class ContextCut:
    """An accepted cut: everything before ``first_kept_entry_id`` is summarized."""

    first_kept_entry_id: str
    summarized_rows: tuple[ContextRow, ...]
    retained_rows: tuple[ContextRow, ...]
    retained_raw_tokens: int
    budget: CutBudget | None = None


@dataclass(frozen=True, slots=True)
class _Plan:
    context_window_tokens: int
    target_occupancy: float
    reserve_tokens: int
    keep_recent_tokens: int
    raw_tokens_before: int
    token_scale: float
    fixed_overhead_tokens: int
    target_context_tokens: int
    continuity_target_tokens: int
    occupancy_ceiling_tokens: int
    safe_ceiling_tokens: int
    reduction_ceiling_tokens: int
    raw_tail_token_budget: int
    binding_constraint: BindingConstraint


def call_result_spans(rows: Sequence[ContextRow]) -> dict[str, CallResultSpan]:
    """Map each tool call id to the row span covering its call and result."""
    spans: dict[str, CallResultSpan] = {}

    def record(tool_call_id: str, index: int, kind: Literal["call", "result"]) -> None:
        if not tool_call_id:
            return
        span = spans.get(tool_call_id)
        if span is None:
            spans[tool_call_id] = CallResultSpan(
                first=index,
                last=index,
                has_call=kind == "call",
                has_result=kind == "result",
            )
            return
        spans[tool_call_id] = CallResultSpan(
            first=min(span.first, index),
            last=max(span.last, index),
            has_call=span.has_call or kind == "call",
            has_result=span.has_result or kind == "result",
        )

    for index, (_entry_id, message) in enumerate(rows):
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                record(call.id, index, "call")
        elif isinstance(message, ToolResultMessage):
            record(message.tool_call_id, index, "result")
    return spans


def _span_crosses(span: CallResultSpan, boundary: int) -> bool:
    """Return whether a cut at ``boundary`` would split this span."""
    return span.has_call and span.has_result and span.first < boundary <= span.last


def closure_safe(spans: dict[str, CallResultSpan], boundary: int) -> bool:
    """Return whether every call/result span lies wholly on one side."""
    return not any(_span_crosses(span, boundary) for span in spans.values())


def _is_turn_boundary(message: object) -> bool:
    return isinstance(
        message,
        (
            UserMessage,
            CustomMessage,
            BashExecutionMessage,
            BranchSummaryMessage,
            CompactionSummaryMessage,
        ),
    )


def _is_cut_point(message: object) -> bool:
    """A cut may not land on a tool result: its call would be left behind."""
    return not isinstance(message, ToolResultMessage)


def _suffix_tokens(token_counts: Sequence[int]) -> list[int]:
    suffix = [0] * (len(token_counts) + 1)
    for index in range(len(token_counts) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + token_counts[index]
    return suffix


def _usage_checkpoints(
    rows: Sequence[ContextRow],
    token_counts: Sequence[int],
) -> list[tuple[int, int]]:
    """Collect (raw estimate, provider-reported context) pairs in row order.

    Aborted and errored responses are skipped: their usage describes a request
    the provider did not complete, so it cannot calibrate anything.
    """
    cumulative = 0
    checkpoints: list[tuple[int, int]] = []
    for index, (_entry_id, message) in enumerate(rows):
        cumulative += token_counts[index]
        if not isinstance(message, AssistantMessage):
            continue
        if message.stop_reason in ("aborted", "error"):
            continue
        context_tokens = provider_context_tokens(message)
        if context_tokens > 0:
            checkpoints.append((cumulative, context_tokens))
    return checkpoints


def _calibrate_tokens(
    rows: Sequence[ContextRow],
    token_counts: Sequence[int],
    tokens_before: int,
    raw_tokens_before: int,
) -> tuple[float, int]:
    """Fit an upper affine envelope ``context ≈ fixed + scale × raw``.

    Tau's character estimator sees message bodies only; the provider also bills
    the system prompt, tool schemas, and serialization. Checkpoints turn that
    unknown into two numbers, and taking the maximum slope keeps the envelope
    above every observation rather than averaging away the worst case.
    """
    checkpoints = _usage_checkpoints(rows, token_counts)
    usable = [
        (raw, context)
        for raw, context in checkpoints
        if raw < raw_tokens_before and context <= tokens_before
    ]
    minimum_delta = max(
        _CALIBRATION_MIN_RAW_DELTA,
        floor(raw_tokens_before * _CALIBRATION_MIN_RAW_RATIO),
    )
    slopes = [
        (tokens_before - context) / (raw_tokens_before - raw)
        for raw, context in usable
        if raw_tokens_before - raw >= minimum_delta and tokens_before - context >= 0
    ]
    if not slopes:
        return 1.0, max(0, tokens_before - raw_tokens_before)
    token_scale = max(1.0, *slopes)
    overhead_candidates = [0.0, tokens_before - token_scale * raw_tokens_before]
    overhead_candidates.extend(context - token_scale * raw for raw, context in usable)
    return token_scale, max(0, int(max(overhead_candidates)))


def _continuity_plan(
    rows: Sequence[ContextRow],
    token_counts: Sequence[int],
    *,
    context_window_tokens: int,
    reserve_tokens: int,
    target_occupancy: float,
    keep_recent_tokens: int,
    tokens_before: int | None,
    calibrate: bool,
) -> _Plan:
    """Reduce the model window and provider usage to one raw-tail budget."""
    window = max(1, int(context_window_tokens))
    occupancy = min(MAX_TARGET_OCCUPANCY, max(MIN_TARGET_OCCUPANCY, float(target_occupancy)))
    reserve = max(0, int(reserve_tokens))
    keep = max(0, int(keep_recent_tokens))
    raw_tokens_before = sum(token_counts)
    reported = tokens_before if tokens_before and tokens_before > 0 else raw_tokens_before
    if calibrate and tokens_before and tokens_before > 0:
        token_scale, fixed_overhead = _calibrate_tokens(
            rows,
            token_counts,
            int(tokens_before),
            raw_tokens_before,
        )
    else:
        token_scale, fixed_overhead = 1.0, max(0, reported - raw_tokens_before)

    continuity_target = max(
        1,
        ceil(fixed_overhead + (keep + SUMMARY_RAW_TOKEN_BUDGET) * token_scale),
    )
    occupancy_ceiling = max(1, floor(window * occupancy))
    hard_ceiling = max(1, window - reserve)
    safe_ceiling = max(1, floor(hard_ceiling * HARD_CEILING_SAFETY_RATIO))
    reduction_ceiling = max(1, floor(reported * MAX_PRECOMPACTION_RATIO))

    constraints: list[tuple[BindingConstraint, int]] = [
        ("continuity", continuity_target),
        ("occupancy", occupancy_ceiling),
        ("reserve", safe_ceiling),
        ("reduction", reduction_ceiling),
    ]
    target_context_tokens = min(value for _name, value in constraints)
    binding_constraint = next(name for name, value in constraints if value == target_context_tokens)
    raw_post_budget = max(0, floor((target_context_tokens - fixed_overhead) / token_scale))
    return _Plan(
        context_window_tokens=window,
        target_occupancy=occupancy,
        reserve_tokens=reserve,
        keep_recent_tokens=keep,
        raw_tokens_before=raw_tokens_before,
        token_scale=token_scale,
        fixed_overhead_tokens=fixed_overhead,
        target_context_tokens=target_context_tokens,
        continuity_target_tokens=continuity_target,
        occupancy_ceiling_tokens=occupancy_ceiling,
        safe_ceiling_tokens=safe_ceiling,
        reduction_ceiling_tokens=reduction_ceiling,
        raw_tail_token_budget=max(0, raw_post_budget - SUMMARY_RAW_TOKEN_BUDGET),
        binding_constraint=binding_constraint,
    )


def _first_fitting_cut(
    rows: Sequence[ContextRow],
    spans: dict[str, CallResultSpan],
    suffix: Sequence[int],
    raw_tail_token_budget: int | None,
) -> int | None:
    """Return the earliest cut keeping a closure-safe tail inside the budget.

    Scanning upward stops at the first fit, which retains as much recent raw
    context as the budget allows. ``None`` budget drops the size test and asks
    only for a closure-safe boundary, which is the most aggressive compaction
    that still keeps a provider-valid tail.
    """
    for index in range(1, len(rows)):
        if not _is_cut_point(rows[index][1]):
            continue
        if raw_tail_token_budget is not None and suffix[index] > raw_tail_token_budget:
            continue
        if not closure_safe(spans, index):
            continue
        return index
    return None


def _last_fitting_cut(
    rows: Sequence[ContextRow],
    spans: dict[str, CallResultSpan],
    suffix: Sequence[int],
    raw_tail_token_budget: int | None = None,
) -> int | None:
    """Return the latest closure-safe boundary, keeping the smallest valid tail.

    Used when nothing fits the tail budget -- one turn larger than the whole
    allowance. Backing off to that minimal tail still shrinks the context and
    stays provider-valid, which beats skipping compaction and overflowing.
    """
    for index in range(len(rows) - 1, 0, -1):
        if not _is_cut_point(rows[index][1]):
            continue
        if raw_tail_token_budget is not None and suffix[index] > raw_tail_token_budget:
            continue
        if not closure_safe(spans, index):
            continue
        return index
    return None


def _last_turn_boundary(rows: Sequence[ContextRow]) -> int:
    for index in range(len(rows) - 1, 0, -1):
        if _is_turn_boundary(rows[index][1]):
            return index
    return 0


def _close_cut(rows: Sequence[ContextRow], spans: dict[str, CallResultSpan], index: int) -> int:
    """Walk a turn boundary back until no call/result span straddles it."""
    current = index
    while current > 0:
        earliest_crossing = current
        for span in spans.values():
            if _span_crosses(span, current):
                earliest_crossing = min(earliest_crossing, span.first)
        if earliest_crossing == current:
            return current
        moved = -1
        for candidate in range(earliest_crossing, 0, -1):
            if _is_turn_boundary(rows[candidate][1]):
                moved = candidate
                break
        if moved < 0 or moved >= current:
            return 0
        current = moved
    return 0


def compute_cut(
    rows: Sequence[ContextRow],
    *,
    keep_recent_tokens: int = DEFAULT_COMPACTION_KEEP_RECENT_TOKENS,
    context_window_tokens: int | None = None,
    reserve_tokens: int = DEFAULT_COMPACTION_RESERVE_TOKENS,
    target_occupancy: float = DEFAULT_TARGET_OCCUPANCY,
    tokens_before: int | None = None,
    calibrate: bool = True,
) -> ContextCut | None:
    """Select a closure-safe cut, or ``None`` when no provider-valid cut exists.

    With a known model window the four ceilings decide how much to retain.
    Without one, whole-turn continuity is preferred when it fits the tail budget,
    because a turn kept intact needs no inference about where work stopped.
    """
    if len(rows) < 2:
        return None

    token_counts = [estimate_message_tokens(message) for _entry_id, message in rows]
    suffix = _suffix_tokens(token_counts)
    spans = call_result_spans(rows)

    plan: _Plan | None = None
    if context_window_tokens is not None and context_window_tokens > 0:
        plan = _continuity_plan(
            rows,
            token_counts,
            context_window_tokens=context_window_tokens,
            reserve_tokens=reserve_tokens,
            target_occupancy=target_occupancy,
            keep_recent_tokens=keep_recent_tokens,
            tokens_before=tokens_before,
            calibrate=calibrate,
        )

    index: int | None = None
    if plan is not None:
        index = _first_fitting_cut(rows, spans, suffix, plan.raw_tail_token_budget)
    else:
        boundary = _last_turn_boundary(rows)
        closed = _close_cut(rows, spans, boundary) if boundary > 0 else 0
        if closed > 0 and suffix[closed] <= max(0, keep_recent_tokens):
            index = closed
        else:
            index = _first_fitting_cut(rows, spans, suffix, max(0, keep_recent_tokens))

    # A single turn larger than the tail budget must still be compactable: fall
    # back to the smallest closure-safe tail rather than skipping compaction and
    # letting the context overflow.
    if index is None:
        index = _last_fitting_cut(rows, spans, suffix)
    if index is None or index <= 0 or index >= len(rows):
        return None

    retained_raw_tokens = suffix[index]
    budget: CutBudget | None = None
    if plan is not None:
        budget = CutBudget(
            strategy="continuity",
            context_window_tokens=plan.context_window_tokens,
            target_occupancy=plan.target_occupancy,
            reserve_tokens=plan.reserve_tokens,
            keep_recent_tokens=plan.keep_recent_tokens,
            raw_tokens_before=plan.raw_tokens_before,
            token_scale=round(plan.token_scale, 6),
            fixed_overhead_tokens=plan.fixed_overhead_tokens,
            retained_raw_tokens=retained_raw_tokens,
            raw_tail_token_budget=plan.raw_tail_token_budget,
            target_context_tokens=plan.target_context_tokens,
            continuity_target_tokens=plan.continuity_target_tokens,
            occupancy_ceiling_tokens=plan.occupancy_ceiling_tokens,
            safe_ceiling_tokens=plan.safe_ceiling_tokens,
            reduction_ceiling_tokens=plan.reduction_ceiling_tokens,
            binding_constraint=plan.binding_constraint,
        )
    return ContextCut(
        first_kept_entry_id=rows[index][0],
        summarized_rows=tuple(rows[:index]),
        retained_rows=tuple(rows[index:]),
        retained_raw_tokens=retained_raw_tokens,
        budget=budget,
    )
