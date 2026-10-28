"""Byte-bounded rendering of projected sections into one summary.

The summary is a fixed-size block inside a token budget, so allocation happens
in UTF-8 bytes and is audited after rendering rather than estimated. Section
ceilings are ceilings, not reservations: what a protected section does not use
is offered to the historical sections, and what the historical sections cannot
use is left unallocated rather than promised to someone else.

Two drop rules keep the output honest:

* A historical section that does not fit is re-rendered with fewer sampled lines
  (:func:`_bounded_block` walks the kept-line count down until the block fits),
  so an omission is always counted and addressed instead of being a mid-line
  truncation.
* A protected section (dialogue, an explicit compaction request, carry-forward
  focus) keeps every item. Only the text inside each item is clipped, and the
  clipped byte count is returned so compaction details can report it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from tau_agent.messages import AgentMessage
from tau_coding.compaction.bounds import (
    MAX_SUMMARY_BYTES,
    bounded_excerpt,
    clip_utf8,
    utf8_bytes,
)
from tau_coding.compaction.projections import Sections

MAX_REQUEST_BYTES = 3072
MAX_CARRY_BYTES = 3072
MAX_TRANSCRIPT_BYTES = 5120
MAX_FOOTER_BYTES = 1536
MAX_INPUT_LINES_PER_SECTION = 128
MAX_RENDERED_LINE_BYTES = 1024

#: Section order and per-section ceilings for a historical (sampled) block.
#: Dialogue is protected content rather than a sampled block, so it keeps its
#: own ceiling instead of an entry in the history table. That ceiling must fit
#: ``dialogue.MAX_RECENT_EXCHANGES`` exchanges at ``dialogue.MAX_EXCHANGE_BYTES``
#: and one heading; a test pins the relationship so the two files cannot drift.
DIALOGUE_HEADER = "[Recent user directions and discussion]"
MAX_DIALOGUE_BYTES = 12288

#: Historical (sampled) sections in render order with their ceilings.
HISTORY_CEILINGS: tuple[tuple[str, str, int], ...] = (
    ("requests", "[Historical requests]", 4096),
    ("files", "[Files And Changes]", 4608),
    ("activity", "[Tool Activity]", 2048),
    ("outstanding", "[Outstanding Context]", 4608),
    ("earlier_turns", "[Earlier Turns]", 3072),
    ("status", "[Current Status]", 2048),
)

REQUEST_HEADER = "[Compaction Request]"
CARRY_FORWARD_HEADER = "[Carry Forward]"
TRANSCRIPT_HEADER = "---"

#: Tells the reader how to recover the exact content the summary left out.
#: Unlike a prose summarizer, this engine omits content by address, so the
#: pointer is only meaningful while the session log stays append-only.
POINTER_LINE = (
    "Exact pre-summary entries stay in the append-only session log; each "
    "[entry ...] id above addresses one record, and /tree branches from it."
)


def format_timestamp(timestamp_ms: int | None) -> str:
    """Render an epoch-millisecond timestamp as a deterministic UTC label."""
    if timestamp_ms is None:
        return "(unknown time)"
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def message_timestamp(message: AgentMessage) -> int | None:
    """Return a message's epoch-millisecond timestamp when it carries one."""
    timestamp = getattr(message, "timestamp", None)
    return timestamp if isinstance(timestamp, int) else None


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Addressing and framing for one render.

    ``first_entry_id``/``last_entry_id`` frame the whole cumulative source, not
    just this compaction's window, so one footer addresses everything the
    summary stands for.
    """

    first_entry_id: str = ""
    last_entry_id: str = ""
    last_timestamp: str = ""
    request_lines: tuple[str, ...] = ()
    carry_lines: tuple[str, ...] = ()
    summary_kind: Literal["compaction", "branch"] = "compaction"


@dataclass(frozen=True, slots=True)
class RenderedSummary:
    """The final summary plus the bytes each protected block had to drop."""

    summary: str
    request_omitted_bytes: int = 0
    dialogue_omitted_bytes: int = 0
    carry_omitted_bytes: int = 0
    section_headers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _HistoryBlock:
    header: str
    lines: tuple[str, ...]
    max_bytes: int
    text: str = ""
    size: int = 0
    minimum: int = 0


def _sampled_lines(lines: Sequence[str], keep: int) -> tuple[str, ...]:
    """Keep a head and a tail of already-rendered lines, marking the middle.

    The marker is a real line in the block, so a reader sees that lines were
    dropped and roughly how many, even though the addresses of those items were
    already recorded by the fold that produced them.
    """
    if len(lines) <= keep:
        return tuple(lines)
    earliest = -(-keep // 2)
    latest = keep // 2
    head = list(lines[:earliest])
    tail = list(lines[len(lines) - latest :]) if latest else []
    return (*head, f"… omitted {len(lines) - keep} rendered lines", *tail)


def _bounded_block(header: str, source_lines: Sequence[str], max_bytes: int) -> str:
    """Render one historical block no larger than ``max_bytes``.

    Lines are clipped individually first, then the kept-line count is walked
    down from the top until the block fits. The last resort is the bare header,
    which keeps the section visible (and therefore visible as an omission)
    instead of silently deleting it.
    """
    clipped = [clip_utf8(line, MAX_RENDERED_LINE_BYTES) for line in source_lines]
    capped = _sampled_lines(clipped, min(len(clipped), MAX_INPUT_LINES_PER_SECTION))
    for keep in range(len(capped), -1, -1):
        block = "\n".join((header, *_sampled_lines(capped, keep)))
        if utf8_bytes(block) <= max_bytes:
            return block
    return clip_utf8(header, max_bytes)


def _bounded_history(blocks: Sequence[_HistoryBlock], max_bytes: int) -> tuple[str, ...]:
    """Fit sampled history blocks together inside one shared ceiling.

    When total demand exceeds the ceiling, each block first keeps its minimum
    (header plus omission marker, or a complete single-line block) and the
    remaining space is divided in proportion to each block's excess demand.
    Sampling slack is not reserved: space a block did not use is not promised to
    content that was never rendered.
    """
    candidates: list[_HistoryBlock] = []
    for block in blocks:
        text = _bounded_block(block.header, block.lines, block.max_bytes)
        size = utf8_bytes(text)
        # The smallest legal block is the header plus its omission line slot:
        # a section that cannot show content still shows where its content went.
        header_only = utf8_bytes(f"{block.header}\n")
        minimum = size if len(block.lines) == 1 else min(size, header_only)
        candidates.append(
            _HistoryBlock(
                header=block.header,
                lines=block.lines,
                max_bytes=block.max_bytes,
                text=text,
                size=size,
                minimum=minimum,
            )
        )

    total = sum(block.size for block in candidates)
    if total <= max_bytes:
        return tuple(block.text for block in candidates)

    minimum_total = sum(block.minimum for block in candidates)
    remaining = max_bytes - minimum_total
    demand = total - minimum_total
    fitted: list[str] = []
    for block in candidates:
        wanted = block.size - block.minimum
        extra = 0 if demand <= 0 else min(wanted, (remaining * wanted) // demand)
        text = _bounded_block(block.header, block.lines, block.minimum + extra)
        fitted.append(text)
        remaining -= utf8_bytes(text) - block.minimum
        demand -= wanted
    return tuple(fitted)


def _protected_block(
    header: str,
    items: Sequence[str],
    max_bytes: int,
) -> tuple[str, int]:
    """Render a protected block, shortening items but never dropping one.

    Dialogue, an explicit compaction request, and carry-forward focus are each
    commitments: dropping one item silently would change what the summary tells
    the model to do. Every item is therefore kept, with its text clipped and the
    clipped amount counted.
    """
    text = "\n\n".join((header, *items))
    if utf8_bytes(text) <= max_bytes:
        return text, 0
    available = max(0, max_bytes - utf8_bytes(header) - 2 * len(items))
    share = available // len(items) if items else 0
    budgets = [min(utf8_bytes(item), share) for item in items]
    remaining = available - sum(budgets)
    omitted_bytes = 0
    excerpts: list[str] = []
    for index, item in enumerate(items):
        extra = min(remaining, utf8_bytes(item) - budgets[index])
        remaining -= extra
        excerpt = bounded_excerpt(item, budgets[index] + extra)
        omitted_bytes += excerpt.omitted_bytes
        excerpts.append(excerpt.text)
    return "\n\n".join((header, *excerpts)), omitted_bytes


def _footer_lines(options: RenderOptions) -> tuple[str, ...]:
    label = "branch summarized" if options.summary_kind == "branch" else "compacted"
    source_kind = (
        "structural source entries"
        if options.summary_kind == "branch"
        else "cumulative source entries"
    )
    if options.first_entry_id or options.last_entry_id:
        first = options.first_entry_id or "(start)"
        last = options.last_entry_id or "(end)"
        source_range = f"{first} → {last}"
    else:
        source_range = "(no entries)"
    timestamp = options.last_timestamp or "(unknown time)"
    return (f"[{label} {timestamp}; {source_kind} {source_range}]", POINTER_LINE)


def render_summary_with_metadata(
    sections: Sections,
    options: RenderOptions | None = None,
) -> RenderedSummary:
    """Render every non-empty section into one byte-bounded summary.

    Protected blocks are rendered first because their actual size is known and
    must not be squeezed later; the historical sections then share whatever the
    ceiling still has, and the footer is allocated before them so the expansion
    pointer is never the thing that gets dropped.
    """
    options = options or RenderOptions()
    blocks: list[str] = []
    history: list[_HistoryBlock] = []
    headers: list[str] = []
    request_omitted_bytes = 0
    dialogue_omitted_bytes = 0
    carry_omitted_bytes = 0

    def add_protected(header: str, items: Sequence[str], max_bytes: int) -> int:
        text, omitted = _protected_block(header, items, max_bytes)
        blocks.append(text)
        headers.append(header)
        return omitted

    def add_requests() -> None:
        nonlocal request_omitted_bytes, carry_omitted_bytes
        if options.request_lines:
            request_omitted_bytes = add_protected(
                REQUEST_HEADER, options.request_lines, MAX_REQUEST_BYTES
            )
        if options.carry_lines:
            carry_omitted_bytes = add_protected(
                CARRY_FORWARD_HEADER, options.carry_lines, MAX_CARRY_BYTES
            )

    if sections.dialogue:
        dialogue_omitted_bytes = add_protected(
            DIALOGUE_HEADER, sections.dialogue, MAX_DIALOGUE_BYTES
        )
    # A compaction request or carry-forward focus is a commitment even when the
    # session has no dialogue left to protect, so these never depend on the
    # dialogue section being present.
    add_requests()

    for name, header, max_bytes in HISTORY_CEILINGS:
        lines: tuple[str, ...] = getattr(sections, name)
        if not lines:
            continue
        history.append(_HistoryBlock(header=header, lines=lines, max_bytes=max_bytes))

    if sections.transcript:
        history.append(
            _HistoryBlock(
                header=TRANSCRIPT_HEADER,
                lines=sections.transcript,
                max_bytes=MAX_TRANSCRIPT_BYTES,
            )
        )

    footer_block = _bounded_block(TRANSCRIPT_HEADER, _footer_lines(options), MAX_FOOTER_BYTES)
    # Every block costs a blank-line separator, plus the final newline.
    framing_bytes = utf8_bytes(footer_block) + 2 * (len(blocks) + len(history)) + 1
    protected_bytes = sum(utf8_bytes(block) for block in blocks)
    budget = max(0, MAX_SUMMARY_BYTES - framing_bytes - protected_bytes)
    blocks.extend(_bounded_history(history, budget))
    blocks.append(footer_block)

    summary = "\n\n".join(blocks) + "\n"
    if utf8_bytes(summary) > MAX_SUMMARY_BYTES:
        # Defensive: allocation should already guarantee the ceiling. Clipping
        # the tail is the only safe overflow response, and it keeps the header
        # and protected blocks that come first.
        summary = f"{clip_utf8(summary, MAX_SUMMARY_BYTES - 1, '')}\n"
    return RenderedSummary(
        summary=summary,
        request_omitted_bytes=request_omitted_bytes,
        dialogue_omitted_bytes=dialogue_omitted_bytes,
        carry_omitted_bytes=carry_omitted_bytes,
        section_headers=tuple(headers)
        + tuple(block.header for block in history if block.header != TRANSCRIPT_HEADER),
    )


def render_summary(
    sections: Sections,
    options: RenderOptions | None = None,
) -> str:
    """Render the summary text alone."""
    return render_summary_with_metadata(sections, options).summary
