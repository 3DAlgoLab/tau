"""Deterministic history for a recursion: collapse old turns, never lose count.

A child relays its own past stdout into every turn's prompt. Head-clipping that
history breaks in two ways at once: the line a child needs is usually not the
first line of a big print, and the prompt still grows with turn count because
the clip is per observation. This module applies the Phase 1 compaction rule
instead:

* the newest observation stays verbatim,
* older ones collapse to their first line, their last line, and the exact number
  of bytes between them,
* the whole block is bounded whatever the turn count, and entries dropped to
  meet that bound are counted rather than silenced.

No model call, no prose classification: the same input renders the same way,
which is what makes a regression test on it possible.
"""

from __future__ import annotations

from dataclasses import dataclass

from tau_coding.compaction.bounds import clip_utf8, utf8_bytes

DEFAULT_MAX_HISTORY_BYTES = 6 * 1024
"""The whole history block, not one observation."""

DEFAULT_VERBATIM_TURNS = 1
"""How many of the newest observations are pasted uncollapsed."""

DEFAULT_COLLAPSED_BYTES = 768
"""Room for one collapsed observation: two surviving lines plus the notice."""


@dataclass(frozen=True, slots=True)
class HistoryBlock:
    """The rendered block plus an account of what the rendering cost."""

    text: str
    bytes: int
    collapsed: int
    dropped: int
    omitted_bytes: int


def clip_tail_utf8(text: str, max_bytes: int, prefix: str = "…") -> str:
    """Return a whole-character suffix of ``text`` fitting ``max_bytes``.

    The mirror of :func:`clip_utf8`. A tail slice has to be walked from the end:
    taking a prefix of a prefix is how a "here is the end of the file" claim ends
    up showing the beginning again.
    """
    if max_bytes <= 0:
        return ""
    if utf8_bytes(text) <= max_bytes:
        return text
    prefix_bytes = utf8_bytes(prefix)
    if prefix_bytes >= max_bytes:
        return ""
    available = max_bytes - prefix_bytes
    parts: list[str] = []
    used = 0
    for character in reversed(text):
        width = utf8_bytes(character)
        if used + width > available:
            break
        parts.append(character)
        used += width
    return f"{prefix}{''.join(reversed(parts))}"


def keep_ends(text: str, budget_bytes: int) -> tuple[str, int]:
    """Return ``text`` holding its head *and* its tail, and the bytes cut out between them.

    One observation is the child's view of a file it just dumped. A head slice tells
    it what the file starts like; the row the child went looking for is usually at
    the other end. Keeping both ends, with the middle counted rather than hidden, is
    what makes "it is not in what you were shown" a claim the child can act on.
    """
    body = text.strip()
    if not body:
        return "", 0
    source_bytes = utf8_bytes(body)
    if source_bytes <= budget_bytes:
        return body, 0

    room = max(0, budget_bytes - 128)  # the marker has to fit inside the budget too
    head_budget = max(1, (room * 3) // 5)
    tail_budget = max(1, room - head_budget)
    head = clip_utf8(body, head_budget, "…")
    tail = clip_tail_utf8(body, tail_budget, "…")
    omitted = max(0, source_bytes - utf8_bytes(head) - utf8_bytes(tail))
    middle = "\n".join(
        [
            head,
            f"[middle omitted: {omitted:,} UTF-8 bytes; what follows is the end of the output]",
            tail,
        ]
    )
    return middle, omitted


def collapse(text: str, budget_bytes: int) -> tuple[str, int]:
    """Return ``text`` reduced to its two ends, and the bytes that were dropped.

    First line and last line survive. That is the whole design: the failure this
    module exists for was a value sitting in the tail of a 2,401-row dump,
    invisible to a clip that keeps only the head.
    """
    body = text.strip()
    if not body:
        return "", 0
    source_bytes = utf8_bytes(body)
    if source_bytes <= budget_bytes:
        return body, 0

    lines = body.splitlines()
    room = max(0, budget_bytes - 96)  # keep the notice itself affordable
    if len(lines) == 1:
        head = clip_utf8(lines[0], room, "…")
        omitted = source_bytes - utf8_bytes(head)
        return f"{head}\n[omitted {omitted:,} UTF-8 bytes of this line]", max(0, omitted)

    head_budget = max(1, room // 2)
    tail_budget = max(1, room - head_budget)
    head = clip_utf8(lines[0], head_budget, "…")
    tail = clip_utf8(lines[-1], tail_budget, "…")
    omitted = max(0, source_bytes - utf8_bytes(head) - utf8_bytes(tail))
    middle = "\n".join(
        [
            head,
            f"[middle omitted: {omitted:,} UTF-8 bytes across {len(lines) - 2} lines]",
            tail,
        ]
    )
    return middle, omitted


def render_history(
    observations: list[str],
    *,
    budget_bytes: int = DEFAULT_MAX_HISTORY_BYTES,
    verbatim: int = DEFAULT_VERBATIM_TURNS,
    collapsed_bytes: int = DEFAULT_COLLAPSED_BYTES,
    numbered: bool = True,
) -> HistoryBlock:
    """Render a turn history inside ``budget_bytes``, spending the budget newest-first.

    The newest ``verbatim`` observations are kept whole (each still clipped to its
    own share, so one giant print cannot evict everything else); older ones are
    collapsed; and if even that overshoots, the *oldest* entries are the ones
    dropped -- counted and reported, never silently removed.
    """
    if not observations:
        return HistoryBlock(text="(nothing ran)", bytes=0, collapsed=0, dropped=0, omitted_bytes=0)

    keep = min(max(0, verbatim), len(observations))
    verbatim_share = max(collapsed_bytes, budget_bytes // (keep + 1))

    # Oldest-first eviction, retried until the block -- marker included -- fits. The
    # marker costs bytes, so a budget met before saying it is not met after.
    for dropped in range(0, len(observations)):
        body, collapsed, omitted = _render_tail(
            observations[dropped:],
            first_index=dropped + 1,
            keep=keep,
            verbatim_share=verbatim_share,
            collapsed_bytes=collapsed_bytes,
            numbered=numbered,
        )
        dropped_bytes = sum(utf8_bytes(text) for text in observations[:dropped])
        if dropped:
            plural = "" if dropped == 1 else "s"
            marker = (
                f"[{dropped} older cell{plural} dropped from this history: "
                f"{dropped_bytes:,} bytes not shown]"
            )
            body = f"{marker}\n{body}" if body else marker
        if utf8_bytes(body) <= budget_bytes or dropped == len(observations) - 1:
            return HistoryBlock(
                text=body,
                bytes=utf8_bytes(body),
                collapsed=collapsed,
                dropped=dropped,
                omitted_bytes=omitted + dropped_bytes,
            )

    return HistoryBlock(text="(nothing ran)", bytes=0, collapsed=0, dropped=0, omitted_bytes=0)


def _render_tail(
    observations: list[str],
    *,
    first_index: int,
    keep: int,
    verbatim_share: int,
    collapsed_bytes: int,
    numbered: bool,
) -> tuple[str, int, int]:
    """Render ``observations`` oldest-first, newest last. Return text, collapsed, omitted."""
    parts: list[str] = []
    collapsed = 0
    omitted_total = 0
    for offset, text in enumerate(observations):
        index = first_index + offset
        is_verbatim = offset >= len(observations) - keep
        if is_verbatim:
            clipped = clip_utf8(text.strip(), verbatim_share, "…")
            omitted_total += max(0, utf8_bytes(text) - utf8_bytes(clipped))
        else:
            clipped, omitted = collapse(text, collapsed_bytes)
            omitted_total += omitted
            collapsed += 1
        parts.append(f"--- cell {index} ---\n{clipped}" if numbered else clipped)
    return "\n".join(parts), collapsed, omitted_total


__all__ = [
    "DEFAULT_COLLAPSED_BYTES",
    "DEFAULT_MAX_HISTORY_BYTES",
    "DEFAULT_VERBATIM_TURNS",
    "HistoryBlock",
    "collapse",
    "keep_ends",
    "render_history",
]
