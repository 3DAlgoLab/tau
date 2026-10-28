"""Byte ceilings and conservation-checked sampling for compaction output.

A compaction summary is a fixed-size block competing for room inside a token
budget, so its size is accounted in UTF-8 bytes rather than characters or
lines: output that is deterministic in content but unbounded in size cannot be
budgeted at all. Every sampler here is conservation-checked -- each dropped
item is counted and addressed by the session entry it came from, so the summary
states what it left out instead of quietly shrinking.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import ceil
from typing import Protocol

MAX_SUMMARY_BYTES = 32 * 1024
MAX_REQUEST_SOURCE_BYTES = 8 * 1024

# Room reserved so the "[omitted N UTF-8 bytes]" marker always survives the
# final clip; without it the marker itself would be clipped away and the
# omission would become invisible again.
_OMISSION_MARKER_RESERVE_BYTES = 64


def utf8_bytes(text: str) -> int:
    """Return the UTF-8 encoded length of ``text`` in bytes."""
    return len(text.encode("utf-8"))


def clip_utf8(text: str, max_bytes: int, suffix: str = "…") -> str:
    """Return a whole-character prefix of ``text`` fitting ``max_bytes``.

    Clipping walks characters, never byte offsets, so a multibyte character can
    never be cut into an invalid sequence.
    """
    if max_bytes <= 0:
        return ""
    if utf8_bytes(text) <= max_bytes:
        return text
    suffix_bytes = utf8_bytes(suffix)
    if suffix_bytes >= max_bytes:
        return ""
    available = max_bytes - suffix_bytes
    parts: list[str] = []
    used = 0
    for character in text:
        width = utf8_bytes(character)
        if used + width > available:
            break
        parts.append(character)
        used += width
    return f"{''.join(parts)}{suffix}"


@dataclass(frozen=True, slots=True)
class BoundedExcerpt:
    """Text clipped to a byte ceiling plus the exact amount that was lost."""

    text: str
    omitted_bytes: int


def bounded_excerpt(text: str, max_bytes: int) -> BoundedExcerpt:
    """Clip ``text`` to ``max_bytes`` while naming the loss inside the text.

    Prose that readers must not mistake for a complete statement (dialogue
    excerpts, error signatures) is clipped this way: the marker tells the model
    the excerpt is partial and how much is missing, and the count can be
    reported again in compaction details.
    """
    source_bytes = utf8_bytes(text)
    if source_bytes <= max_bytes:
        return BoundedExcerpt(text=text, omitted_bytes=0)
    prefix = clip_utf8(text, max(0, max_bytes - _OMISSION_MARKER_RESERVE_BYTES), "")
    omitted_bytes = source_bytes - utf8_bytes(prefix)
    clipped = clip_utf8(f"{prefix}\n[omitted {omitted_bytes} UTF-8 bytes]", max_bytes, "")
    return BoundedExcerpt(text=clipped, omitted_bytes=omitted_bytes)


@dataclass(frozen=True, slots=True)
class CanonicalText:
    """Whitespace-normalized text with its truncation state recorded."""

    text: str
    truncated: bool
    source_bytes: int


def canonicalize_text(text: str, max_bytes: int = MAX_REQUEST_SOURCE_BYTES) -> CanonicalText:
    """Collapse whitespace and clip to ``max_bytes``.

    Prose runs through this before it is compared, sampled, or echoed, so
    formatting differences in the source cannot change a deterministic result.
    """
    canonical = " ".join(text.split())
    source_bytes = utf8_bytes(canonical)
    return CanonicalText(
        text=clip_utf8(canonical, max_bytes),
        truncated=source_bytes > max_bytes,
        source_bytes=source_bytes,
    )


class Addressed(Protocol):
    """A sampled item that can be named by the session entry it came from.

    Declared as a read-only property so frozen dataclasses satisfy it: the
    samplers only ever read the address, and a settable protocol member would
    exclude exactly the immutable value types the folds build.
    """

    @property
    def entry_id(self) -> str: ...


@dataclass(frozen=True, slots=True)
class AddressedSample[T: Addressed]:
    """The kept head and tail of a stream, plus an accounting of the middle.

    ``split_index`` is where the tail begins, which is where a caller renders
    the omission marker. Keeping both ends preserves chronology: the earliest
    items show how the work started and the latest show where it stands.
    """

    values: tuple[T, ...]
    omitted: int
    omitted_first_entry_id: str | None
    omitted_last_entry_id: str | None
    split_index: int


def _accept_sample_accounting(total: int, kept: int, omitted: int, max_values: int) -> bool:
    """Verify a sampler conserved items and respected its ceiling."""
    return kept + omitted == total and kept <= max(0, max_values)


def sample_addressed_from[T: Addressed](
    values: Iterable[T],
    max_values: int,
) -> AddressedSample[T]:
    """Sample ``values`` as a head plus a tail, counting what fell between.

    Streams of unknown length are accepted so a projection never has to buffer
    its source first. A failed conservation check raises instead of returning a
    sample whose counts disagree with its contents.
    """
    earliest_limit = ceil(max_values / 2)
    latest_limit = max_values // 2
    earliest: list[T] = []
    latest: list[T] = []
    omitted = 0
    total = 0
    omitted_first_entry_id: str | None = None
    omitted_last_entry_id: str | None = None

    for value in values:
        total += 1
        if len(earliest) < earliest_limit:
            earliest.append(value)
            continue
        latest.append(value)
        if len(latest) <= latest_limit:
            continue
        displaced = latest.pop(0)
        omitted += 1
        omitted_first_entry_id = omitted_first_entry_id or displaced.entry_id
        omitted_last_entry_id = displaced.entry_id

    if not _accept_sample_accounting(total, len(earliest) + len(latest), omitted, max_values):
        raise RuntimeError("addressed sample failed conservation accounting")
    return AddressedSample(
        values=(*earliest, *latest),
        omitted=omitted,
        omitted_first_entry_id=omitted_first_entry_id,
        omitted_last_entry_id=omitted_last_entry_id,
        split_index=len(earliest),
    )


def sample_addressed[T: Addressed](
    values: Iterable[T],
    max_values: int,
) -> AddressedSample[T]:
    """Sample a materialized iterable; see :func:`sample_addressed_from`."""
    return sample_addressed_from(values, max_values)


def omission_line(
    count: int,
    first_entry_id: str | None,
    last_entry_id: str | None,
    noun: str,
) -> str:
    """Render an omission marker that addresses the omitted source range."""
    if first_entry_id or last_entry_id:
        first = first_entry_id or "(start)"
        last = last_entry_id or "(end)"
        source_range = f"{first} → {last}"
    else:
        source_range = "(unknown range)"
    return f"… omitted {count} {noun}; source entries {source_range}"
