"""Structural QA probes that grade a compaction against ground truth.

Probes are generated from the normalized event stream, never from the rendered
summary. A check that reads the text it is grading cannot notice what that text
left out, which is the exact failure mode a compaction engine has: it summarizes,
and a silent omission looks like a shorter summary rather than a lost fact.

Each probe is one of two classes:

``content``
    a fact that must appear in the summary: the request in flight, a file that
    was modified, the signature of an error nobody answered, the last command.
``address``
    a pointer that must appear when a fact was *not* carried: an addressed
    omission marker, an ``(#N)`` transcript reference, the expansion footer.

A summary is allowed to drop content. It is never allowed to drop content and
leave no way back to it, so both classes have to pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from tau_coding.compaction.bounds import omission_line, sample_addressed
from tau_coding.compaction.dialogue import dialogue_excerpts, recent_dialogue
from tau_coding.compaction.events import CompactionEvent, UserEvent, first_line
from tau_coding.compaction.projections import (
    MAX_EARLIER_TURNS,
    MAX_FILES_PER_KIND,
    MAX_UNRESOLVED,
    MAX_USER_ONELINER_BYTES,
    MODIFYING_TOOLS,
    NOUN_EARLIER_TURNS,
    NOUN_FILES,
    NOUN_TRANSCRIPT,
    NOUN_UNRESOLVED,
    TRANSCRIPT_WINDOW,
    collect_operations,
    error_signature,
    error_subject,
    iter_turns,
    path_of,
    truncate,
    unresolved_operations,
)
from tau_coding.compaction.render import POINTER_LINE

ProbeClass = Literal["content", "address"]


@dataclass(frozen=True, slots=True)
class Probe:
    """One reconstruction question and the text that answers it."""

    id: str
    probe_class: ProbeClass
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class ProbeFailure:
    """A probe the summary could not answer, with the reason to show a human."""

    probe: Probe
    reason: str


@dataclass(frozen=True, slots=True)
class QaReport:
    """Scores over one summary. A score of 1.0 means nothing was lost silently."""

    score: float
    content_score: float
    address_score: float
    probe_count: int
    failures: tuple[ProbeFailure, ...]

    @property
    def ok(self) -> bool:
        """Return True when every probe was answered."""
        return not self.failures


@dataclass(frozen=True, slots=True)
class _FileTarget:
    """A path a modifying operation touched, addressed by its entry."""

    entry_id: str
    index: int
    path: str


def _essential_path_token(path: str) -> str:
    """Return the path tail the summary must keep when it collapses a root."""
    trimmed = path.rstrip("/\\")
    separator = max(trimmed.rfind("/"), trimmed.rfind("\\"))
    return trimmed[separator + 1 :] or path


def event_window(
    events: Sequence[CompactionEvent],
    cut_index: int | None,
) -> list[CompactionEvent]:
    """Return the events a cut summarizes.

    ``cut_index`` is the exclusive array boundary, which is the number of rows
    ``ContextCut.summarized_rows`` holds.
    """
    if cut_index is None:
        return []
    end = max(0, min(len(events), cut_index))
    return list(events[:end])


def _dialogue_probes(
    source: Sequence[CompactionEvent],
    retained: Sequence[CompactionEvent],
) -> tuple[list[Probe], frozenset[str]]:
    """Derive excerpt probes from source dialogue, not from rendered sections.

    Reading the projection instead would grade the summary against itself and
    could not notice a decayed later paragraph.
    """
    exchanges = recent_dialogue(source, retained)
    probes = [
        Probe(
            id=f"dialogue:{exchange.user.entry_id}:{excerpt.heading}",
            probe_class="content",
            question=(
                "Which recent user direction or paired assistant response must "
                "stay distinguishable?"
            ),
            answer=f"{excerpt.heading}\n{excerpt.text}",
        )
        for exchange in exchanges
        for excerpt in dialogue_excerpts(exchange)
    ]
    return probes, frozenset(exchange.user.entry_id for exchange in exchanges)


def _file_probes(source: Sequence[CompactionEvent]) -> list[Probe]:
    """Probe every modified path the files fold sampled."""
    grouped: dict[str, list[_FileTarget]] = {}
    seen: dict[str, set[str]] = {}
    for operation in collect_operations(source):
        if operation.tool not in MODIFYING_TOOLS or operation.outcome != "succeeded":
            continue
        path = path_of(operation.arguments)
        if not path:
            continue
        paths = seen.setdefault(operation.tool, set())
        if path in paths:
            continue
        paths.add(path)
        grouped.setdefault(operation.tool, []).append(
            _FileTarget(entry_id=operation.entry_id, index=operation.index, path=path)
        )

    probes: list[Probe] = []
    for targets in grouped.values():
        sampled = sample_addressed(targets, MAX_FILES_PER_KIND)
        for target in sampled.values:
            probes.append(
                Probe(
                    id=f"modified-file:{target.index}:{target.path}",
                    probe_class="content",
                    question=f"Which file-modification address must remain for {target.path}?",
                    answer=_essential_path_token(target.path),
                )
            )
        if sampled.omitted:
            probes.append(
                Probe(
                    id=f"modified-file-omission:{sampled.omitted_first_entry_id}",
                    probe_class="address",
                    question=f"Which source range addresses {sampled.omitted} omitted files?",
                    answer=omission_line(
                        sampled.omitted,
                        sampled.omitted_first_entry_id,
                        sampled.omitted_last_entry_id,
                        NOUN_FILES,
                    ),
                )
            )
    return probes


def _error_probes(source: Sequence[CompactionEvent]) -> list[Probe]:
    """Probe every unanswered failure, by identity rather than by error prose."""
    sampled = sample_addressed(unresolved_operations(source), MAX_UNRESOLVED)
    probes = [
        Probe(
            id=f"unresolved-error:{operation.index}",
            probe_class="content",
            question=(
                f"What is the signature of unresolved {operation.tool} error #{operation.index}?"
            ),
            answer=f"{error_subject(operation)}{error_signature(operation)}",
        )
        for operation in sampled.values
    ]
    if sampled.omitted:
        probes.append(
            Probe(
                id=f"unresolved-error-omission:{sampled.omitted_first_entry_id}",
                probe_class="address",
                question=f"Which source range addresses {sampled.omitted} omitted open errors?",
                answer=omission_line(
                    sampled.omitted,
                    sampled.omitted_first_entry_id,
                    sampled.omitted_last_entry_id,
                    NOUN_UNRESOLVED,
                ),
            )
        )
    return probes


def _turn_probes(source: Sequence[CompactionEvent]) -> list[Probe]:
    """Probe the earlier-turn fold and the pointers that keep dropped turns live."""
    probes: list[Probe] = []
    sampled = sample_addressed(iter_turns(source), MAX_EARLIER_TURNS)
    transcript_ids = {event.entry_id for event in source[-TRANSCRIPT_WINDOW:]}
    for turn in sampled.values:
        if not turn.is_user:
            continue
        probes.append(
            Probe(
                id=f"earlier-turn:{turn.entry_id}",
                probe_class="content",
                question=f"Which sampled one-liner accounts for earlier turn {turn.entry_id}?",
                answer=turn.context_line,
            )
        )
        if turn.entry_id not in transcript_ids:
            probes.append(
                Probe(
                    id=f"earlier-turn-address:{turn.entry_id}",
                    probe_class="address",
                    question=(
                        f"What pointer keeps earlier turn {turn.entry_id} addressable "
                        "outside the brief transcript?"
                    ),
                    answer=turn.context_line,
                )
            )
    if sampled.omitted:
        probes.append(
            Probe(
                id=f"earlier-turn-omission:{sampled.omitted_first_entry_id}",
                probe_class="address",
                question=f"Which source range addresses {sampled.omitted} omitted turns?",
                answer=omission_line(
                    sampled.omitted,
                    sampled.omitted_first_entry_id,
                    sampled.omitted_last_entry_id,
                    NOUN_EARLIER_TURNS,
                ),
            )
        )
    return probes


def _transcript_probes(source: Sequence[CompactionEvent]) -> list[Probe]:
    """Probe the stable ``(#N)`` references and the head they leave behind."""
    if not source:
        return []
    probes = [
        Probe(
            id=f"transcript-pointer:{source[-1].index}",
            probe_class="address",
            question="Which transcript pointer addresses the newest summarized event?",
            answer=f"(#{source[-1].index})",
        )
    ]
    omitted = len(source) - min(len(source), TRANSCRIPT_WINDOW)
    if omitted > 0:
        probes.append(
            Probe(
                id=f"transcript-omission:{source[0].entry_id}",
                probe_class="address",
                question=f"Which source range addresses {omitted} dropped transcript events?",
                answer=omission_line(
                    omitted, source[0].entry_id, source[omitted - 1].entry_id, NOUN_TRANSCRIPT
                ),
            )
        )
    return probes


def generate_probes(
    events: Sequence[CompactionEvent],
    cut_index: int | None,
) -> list[Probe]:
    """Build the probe set for one compaction window.

    Nothing here reads a projection or the rendered summary: every answer comes
    from the events the cut removed, so a fold that forgets one of them fails.
    """
    source = event_window(events, cut_index)
    if not source:
        return []
    retained = list(events[len(source) :])
    probes: list[Probe] = []

    dialogue_probes, protected = _dialogue_probes(source, retained)
    probes.extend(dialogue_probes)

    first_user = next((event for event in source if isinstance(event, UserEvent)), None)
    if first_user is not None and first_user.entry_id not in protected:
        probes.append(
            Probe(
                id="historical-goal",
                probe_class="content",
                question="Which original request must remain as historical evidence?",
                answer=truncate(first_line(first_user.text), MAX_USER_ONELINER_BYTES),
            )
        )

    probes.extend(_file_probes(source))
    probes.extend(_error_probes(source))

    last_change = None
    for operation in collect_operations(source):
        if operation.tool in MODIFYING_TOOLS and path_of(operation.arguments):
            last_change = operation
    if last_change is not None:
        probes.append(
            Probe(
                id=f"last-modification:{last_change.index}",
                probe_class="content",
                question="Which path did the last file-modifying call target?",
                answer=path_of(last_change.arguments) or "",
            )
        )

    probes.extend(_turn_probes(source))
    probes.extend(_transcript_probes(source))
    probes.append(
        Probe(
            id="footer-recall",
            probe_class="address",
            question="What pointer lets a reader expand the omitted entries?",
            answer=POINTER_LINE,
        )
    )
    return probes


def check_probes(summary: str, probes: Sequence[Probe]) -> tuple[list[Probe], list[ProbeFailure]]:
    """Return the answered probes and the failures, each with a reason."""
    passed: list[Probe] = []
    failed: list[ProbeFailure] = []
    for probe in probes:
        if probe.answer and probe.answer in summary:
            passed.append(probe)
        else:
            failed.append(
                ProbeFailure(
                    probe=probe,
                    reason=(
                        "summary does not contain the expected answer "
                        f"{probe.answer!r} for {probe.question!r}"
                    ),
                )
            )
    return passed, failed


def qa_report(
    events: Sequence[CompactionEvent],
    cut_index: int | None,
    summary: str,
) -> QaReport:
    """Grade one summary against the events its cut removed."""
    probes = generate_probes(events, cut_index)
    _, failed = check_probes(summary, probes)
    content = [probe for probe in probes if probe.probe_class == "content"]
    address = [probe for probe in probes if probe.probe_class == "address"]
    content_failed = [item for item in failed if item.probe.probe_class == "content"]
    address_failed = [item for item in failed if item.probe.probe_class == "address"]

    def score(group: Sequence[Probe], failures: Sequence[ProbeFailure]) -> float:
        if not group:
            return 1.0
        return (len(group) - len(failures)) / len(group)

    return QaReport(
        score=score(probes, failed),
        content_score=score(content, content_failed),
        address_score=score(address, address_failed),
        probe_count=len(probes),
        failures=tuple(failed),
    )
