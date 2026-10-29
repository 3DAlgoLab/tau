"""Session context as handles the kernel opens, not text the prompt carries.

A code-mode cell that needs its own conversation history asks for it by address
and receives a path. The bytes of that history never enter a frame, a tool
result, or a prompt -- they enter a file the cell slices with ordinary Python.
This is the same handle discipline as ``attach``: the host does the I/O it
already owns, and the kernel gets a reference to it.

The addressing is the 1-based ``(#N)`` a compaction summary already cites, on
purpose. When a summary says "3 of 31 events [omitted]", the cell can open the
omitted range and read it; the omission marker and the handle speak one
currency.

Everything that crosses is accounted. Each row says how many bytes of itself
were clipped, and a sampled slice names the entry range it dropped, using the
compaction module's conservation-checked samplers rather than a second
implementation of the same idea.

"""

from __future__ import annotations

import json
import shutil
import tempfile
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tau_agent.messages import AgentMessage, message_text
from tau_coding.compaction.bounds import (
    bounded_excerpt,
    clip_utf8,
    omission_line,
    sample_addressed,
    utf8_bytes,
)

MAX_INDEX_ROWS = 120
MAX_INDEX_PREVIEW_BYTES = 200
MAX_SLICE_ROWS = 400
MAX_ROW_BYTES = 16 * 1024
MAX_SLICE_BYTES = 2 * 1024 * 1024
MIN_ROW_BYTES = 256

# Addressing metadata (entry id, address, role, byte counts) around each row of
# text. The slice budget is spent on the whole file, so this is subtracted
# before the per-row text budget is solved.
ROW_METADATA_BYTES = 128


@dataclass(frozen=True, slots=True)
class _AddressedRow:
    """One context entry, positioned so the samplers can address what they drop."""

    entry_id: str
    address: int
    role: str
    text: str


class ContextEntries:
    """Answer ``context_index()`` / ``context_open()`` for one session.

    ``rows`` is a live view of the *active* branch, not a snapshot: a cell that
    runs after a compaction or a branch switch sees the context the next model
    call would see, which is the only context worth searching.
    """

    def __init__(self, rows: Callable[[], Sequence[tuple[str, AgentMessage]]]) -> None:
        self._rows = rows
        self._dir: Path | None = None
        self._slices = 0
        self._finalizer: Callable[[], None] | None = None

    # -- kernel-facing surface ---------------------------------------------

    async def index(self, query: str | None) -> dict[str, Any]:
        """Describe the active context in bounded rows; full text stays here."""
        rows = self._addressed_rows()
        matched = rows
        if query:
            needle = query.casefold()
            matched = [
                row
                for row in rows
                if needle in row.role.casefold() or needle in row.text.casefold()
            ]
        sample = sample_addressed(matched, MAX_INDEX_ROWS)
        payload = [
            {
                "address": row.address,
                "entry_id": row.entry_id,
                "role": row.role,
                "bytes": utf8_bytes(row.text),
                "preview": clip_utf8(_one_line(row.text), MAX_INDEX_PREVIEW_BYTES),
            }
            for row in sample.values
        ]
        reply: dict[str, Any] = {
            "rows": payload,
            "total": len(rows),
            "matched": len(matched),
            "omitted": sample.omitted,
        }
        if query:
            reply["query"] = query
        if sample.omitted:
            reply["omitted_note"] = omission_line(
                sample.omitted,
                sample.omitted_first_entry_id,
                sample.omitted_last_entry_id,
                "entries",
            )
        return reply

    async def open(self, refs: tuple[str, ...]) -> dict[str, Any]:
        """Write the addressed entries to a JSONL file and return its path."""
        rows = self._addressed_rows()
        by_address = {row.address: row for row in rows}
        entry_to_address = {row.entry_id: row.address for row in rows}
        wanted: list[int] = []
        unknown: list[str] = []
        for ref in refs:
            addresses, unresolved = _resolve_ref(ref, rows, entry_to_address)
            wanted.extend(addresses)
            unknown.extend(unresolved)
        selected = sorted(dict.fromkeys(wanted))
        picked = [by_address[address] for address in selected]

        # The file is what is bounded, not the row text alone: a row also costs
        # its addressing envelope, so the row count and the per-row budget are
        # solved together. Whatever does not fit is sampled away head+tail and
        # counted against the entries it came from, rather than truncated away.
        capacity = max(1, len(picked))
        per_row = MAX_ROW_BYTES
        while True:
            per_row = min(
                MAX_ROW_BYTES,
                max(MIN_ROW_BYTES, MAX_SLICE_BYTES // capacity - ROW_METADATA_BYTES),
            )
            room = max(1, MAX_SLICE_BYTES // (per_row + ROW_METADATA_BYTES))
            if room >= capacity or capacity == 1:
                break
            capacity = room
        sample = sample_addressed(picked, min(MAX_SLICE_ROWS, capacity))
        kept = list(sample.values)

        written: list[dict[str, Any]] = []
        clipped = 0
        for row in kept:
            excerpt = bounded_excerpt(row.text, per_row)
            clipped += excerpt.omitted_bytes
            written.append(
                {
                    "address": row.address,
                    "entry_id": row.entry_id,
                    "role": row.role,
                    "bytes": utf8_bytes(row.text),
                    "omitted_bytes": excerpt.omitted_bytes,
                    "text": excerpt.text,
                }
            )

        path = self._slice_path()
        body = "\n".join(json.dumps(item, ensure_ascii=False) for item in written)
        path.write_text(f"{body}\n" if body else "", encoding="utf-8")
        reply: dict[str, Any] = {
            "path": str(path),
            "written": len(written),
            "requested": len(selected),
            "omitted": sample.omitted,
            "file_bytes": utf8_bytes(path.read_text(encoding="utf-8")),
            "refs": list(refs),
        }
        notes: list[str] = []
        if unknown:
            reply["unknown_refs"] = unknown
            notes.append(
                f"unresolved refs: {', '.join(unknown[:5])} (context_index() lists addresses)"
            )
        if sample.omitted:
            reply["omitted_note"] = omission_line(
                sample.omitted,
                sample.omitted_first_entry_id,
                sample.omitted_last_entry_id,
                "entries",
            )
            notes.append(reply["omitted_note"])
        if clipped:
            notes.append(f"{clipped} bytes clipped from row text at a {per_row}-byte row budget")
        if sample.omitted or clipped:
            reply["per_row_bytes"] = per_row
            reply["file_budget_bytes"] = MAX_SLICE_BYTES
        if notes:
            reply["note"] = "; ".join(notes)
        return reply

    def dispose(self) -> None:
        """Remove the scratch directory holding this session's materialized slices."""
        if self._dir is not None:
            if self._finalizer is not None:
                self._finalizer()  # runs the cleanup and detaches it
                self._finalizer = None
            self._dir = None

    # -- internals ----------------------------------------------------------

    def _addressed_rows(self) -> list[_AddressedRow]:
        return [
            _AddressedRow(
                entry_id=entry_id,
                address=index,
                role=message.role,
                text=message_text(message),
            )
            for index, (entry_id, message) in enumerate(self._rows(), start=1)
        ]

    def _slice_path(self) -> Path:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="tau-context-"))
            # ``dispose()`` is the polite path; this is the survivable one. A
            # session that dies without closing (crash, cancelled task) still
            # must not leave a copy of its transcript readable in /tmp.
            self._finalizer = weakref.finalize(self, _remove_tree, str(self._dir))
        self._slices += 1
        return self._dir / f"context-{self._slices}.jsonl"


def _remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _resolve_ref(
    ref: str,
    rows: Sequence[_AddressedRow],
    entry_to_address: dict[str, int],
) -> tuple[list[int], list[str]]:
    """Turn one typed reference into addresses, or report it as unresolved.

    The grammar is deliberately small and entirely numeric: ``#N``, ``A-B``,
    ``last:N``, ``first:N``, ``all``, or an entry id. A ref that resolves to
    nothing comes back to the cell as an ``unknown_refs`` entry rather than an
    empty file, because the model needs to know which of its addresses was
    wrong, not that it got nothing.
    """

    count = len(rows)
    token = ref.strip()
    if count == 0:
        return [], [token]
    lowered = token.casefold()
    if lowered == "all":
        return list(range(1, count + 1)), []
    if lowered.startswith(("last:", "first:")):
        head, _, raw = token.partition(":")
        size = _positive(raw)
        if size is None:
            return [], [token]
        size = min(size, count)
        if head.casefold() == "last":
            return list(range(count - size + 1, count + 1)), []
        return list(range(1, size + 1)), []
    if "-" in token:
        raw_first, _, raw_last = token.partition("-")
        first, last = _positive(raw_first), _positive(raw_last)
        if first is None or last is None or last < first:
            return [], [token]
        return [address for address in range(first, min(last, count) + 1)], []
    address = _positive(token.lstrip("#"))
    if address is not None:
        if 1 <= address <= count:
            return [address], []
        return [], [token]
    mapped = entry_to_address.get(token)
    if mapped is not None:
        return [mapped], []
    return [], [token]


def _positive(text: str) -> int | None:
    stripped = text.strip().lstrip("#")
    return int(stripped) if stripped.isdigit() else None


__all__ = [
    "ContextEntries",
    "MAX_INDEX_ROWS",
    "MAX_ROW_BYTES",
    "MAX_SLICE_BYTES",
    "MAX_SLICE_ROWS",
    "MIN_ROW_BYTES",
    "ROW_METADATA_BYTES",
]
