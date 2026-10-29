"""Session context as handles: addressing, budgets, and what the cell is told.

These tests are about the policy layer only -- no child process. The frame path
is pinned separately in ``test_kernel_process.py``; here the question is what a
``ContextEntries`` source is allowed to say about a session's history, because
that answer is what a model will act on. Every refusal must name the address it
refused and the amount it dropped.

"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import pytest

from tau_agent.messages import AssistantMessage, UserMessage
from tau_coding import context_entries as ce
from tau_coding.context_entries import ContextEntries

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _rows(count: int, *, pad: int = 0) -> list[tuple[str, object]]:
    rows: list[tuple[str, object]] = []
    for index in range(1, count + 1):
        text = f"entry {index}" + (" y" * pad if pad else "")
        message = UserMessage(content=text) if index % 2 else AssistantMessage(content=text)
        rows.append((f"e{index:03d}", message))
    return rows


def _source(rows: list[tuple[str, object]]) -> ContextEntries:
    return ContextEntries(rows=lambda: rows)


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


async def test_index_addresses_entries_the_way_a_summary_does() -> None:
    source = _source(_rows(4))
    index = await source.index(None)
    assert index["total"] == 4
    assert index["matched"] == 4
    assert [row["address"] for row in index["rows"]] == [1, 2, 3, 4]
    assert [row["entry_id"] for row in index["rows"]] == ["e001", "e002", "e003", "e004"]
    assert [row["role"] for row in index["rows"]] == ["user", "assistant", "user", "assistant"]
    assert index["rows"][0]["preview"] == "entry 1"
    assert index["omitted"] == 0


async def test_a_query_filters_without_hiding_the_total() -> None:
    source = _source(_rows(6))
    index = await source.index("ENTRY 3")
    assert index["matched"] == 1
    assert index["total"] == 6
    assert index["rows"][0]["address"] == 3
    assert index["query"] == "ENTRY 3"


async def test_an_oversized_index_is_sampled_and_says_what_it_dropped() -> None:
    source = _source(_rows(ce.MAX_INDEX_ROWS + 30))
    index = await source.index(None)
    assert len(index["rows"]) == ce.MAX_INDEX_ROWS
    assert index["omitted"] == 30
    # Head and tail, never a middle-only slice: chronology survives the sample.
    assert index["rows"][0]["address"] == 1
    assert index["rows"][-1]["address"] == ce.MAX_INDEX_ROWS + 30
    assert "omitted 30 entries" in index["omitted_note"]


async def test_previews_are_single_line_and_byte_bounded() -> None:
    bulky = "line one\nline two\n" + "z" * 500
    rows: list[tuple[str, object]] = [("e1", UserMessage(content=bulky))]
    index = await _source(rows).index(None)
    preview = index["rows"][0]["preview"]
    assert "\n" not in preview
    assert len(preview.encode("utf-8")) <= ce.MAX_INDEX_PREVIEW_BYTES
    assert preview.endswith("…")
    assert index["rows"][0]["bytes"] > ce.MAX_INDEX_PREVIEW_BYTES


async def test_refs_accept_every_typed_form(tmp_path: Path) -> None:
    source = _source(_rows(10))
    reply = await source.open(("#2", "4-6", "last:2", "first:1", "e010"))
    rows = _lines(Path(reply["path"]))
    assert [row["address"] for row in rows] == [1, 2, 4, 5, 6, 9, 10]
    assert reply["written"] == 7
    assert reply["omitted"] == 0
    assert "unknown_refs" not in reply


async def test_an_unresolved_ref_comes_back_named(tmp_path: Path) -> None:
    source = _source(_rows(5))
    reply = await source.open(("#4", "#9", "nonsense"))
    assert reply["unknown_refs"] == ["#9", "nonsense"]
    assert [row["address"] for row in _lines(Path(reply["path"]))] == [4]
    assert "unresolved refs: #9, nonsense" in reply["note"]


async def test_all_is_deduplicated_and_ordered(tmp_path: Path) -> None:
    source = _source(_rows(4))
    reply = await source.open(("all", "#2", "#2"))
    assert [row["address"] for row in _lines(Path(reply["path"]))] == [1, 2, 3, 4]
    assert reply["requested"] == 4


async def test_a_slice_over_the_row_ceiling_names_the_omitted_range(tmp_path: Path) -> None:
    source = _source(_rows(ce.MAX_SLICE_ROWS + 25))
    reply = await source.open(("all",))
    rows = _lines(Path(reply["path"]))
    assert len(rows) == ce.MAX_SLICE_ROWS
    assert reply["omitted"] == 25
    assert "omitted 25 entries" in reply["note"]
    assert rows[0]["address"] == 1
    assert rows[-1]["address"] == ce.MAX_SLICE_ROWS + 25


async def test_row_text_is_clipped_with_the_loss_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ce, "MAX_ROW_BYTES", 512)
    huge = "hundreds of bytes of tool output\n" * 60
    source = _source([("ebig", UserMessage(content=huge))])
    reply = await source.open(("all",))
    row = _lines(Path(reply["path"]))[0]
    assert row["omitted_bytes"] > 0
    assert row["bytes"] == len(huge.encode("utf-8"))
    assert f"[omitted {row['omitted_bytes']} UTF-8 bytes]" in str(row["text"])
    assert "bytes clipped" in reply["note"]


async def test_the_file_budget_bounds_the_file_and_accounts_for_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ce, "MAX_SLICE_BYTES", 4096)
    source = _source(_rows(20, pad=400))
    reply = await source.open(("all",))
    rows = _lines(Path(reply["path"]))

    # The file stays inside budget by dropping rows -- counted, head and tail --
    # and clipping the rest to one shared row budget, never by truncating the
    # file itself.
    assert len(rows) < 20
    assert reply["written"] == len(rows)
    assert reply["omitted"] == 20 - len(rows)
    assert rows[0]["address"] == 1
    assert rows[-1]["address"] == 20
    assert len({int(str(row["omitted_bytes"])) for row in rows}) <= 2
    assert reply["per_row_bytes"] == 4096 // len(rows) - ce.ROW_METADATA_BYTES
    assert reply["file_bytes"] <= 4096
    assert f"omitted {reply['omitted']} entries" in reply["note"]
    assert "bytes clipped" in reply["note"]


async def test_the_view_is_live_not_a_snapshot() -> None:
    held = {"rows": _rows(2)}
    source = ContextEntries(rows=lambda: held["rows"])
    assert (await source.index(None))["total"] == 2
    held["rows"] = _rows(5)
    index = await source.index(None)
    assert index["total"] == 5
    assert index["rows"][-1]["address"] == 5


async def test_dispose_removes_the_scratch_directory() -> None:
    source = _source(_rows(3))
    reply = await source.open(("all",))
    path = Path(reply["path"])
    assert path.is_file()
    scratch = path.parent
    source.dispose()
    assert not scratch.exists()


async def test_a_discarded_context_does_not_leak_its_transcript() -> None:
    # dispose() is the polite path. A session that dies without one still must
    # not leave its transcript materialized in the scratch directory, so the
    # object's own collection has to remove it.
    source = _source(_rows(2))
    reply = await source.open(("all",))
    scratch = Path(reply["path"]).parent
    assert scratch.is_dir()
    del source
    gc.collect()
    assert not scratch.exists()


async def test_an_empty_context_answers_without_raising(tmp_path: Path) -> None:
    source = _source([])
    index = await source.index(None)
    assert index["total"] == 0 and index["rows"] == []
    reply = await source.open(("all", "#3"))
    assert reply["written"] == 0
    assert Path(reply["path"]).read_text(encoding="utf-8") == ""
    assert reply["unknown_refs"] == ["all", "#3"]
