"""Code mode tests: the ``python_repl`` tool, the kernel behind it, its session scope.

The kernel's own contract is pinned in ``test_kernel_*.py``. These tests drive the
tool the model actually sees -- through ``AgentTool.execute`` -- and then check the
session rules: code mode is on by default, an explicit tool set owns its tools, the
namespace survives ``/reload``, and closing the session closes the process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start, text_delta
from tau_agent.harness import SimpleCancellationToken
from tau_agent.messages import AssistantMessage
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentToolResult, ToolCancellationToken
from tau_agent.types import JSONValue
from tau_ai import FakeProvider
from tau_coding.code_mode import (
    MAX_ATTACHED_FILE_BYTES,
    MAX_CODE_BYTES,
    PYTHON_REPL_TOOL_NAME,
    CodeMode,
)
from tau_coding.kernel.scaffold import RESERVED
from tau_coding.kernel.subcall import SubcallBridge
from tau_coding.session import CodingSession, CodingSessionConfig
from tau_coding.session_manager import SessionManager
from tau_coding.tools import ToolInputError

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@asynccontextmanager
async def _mode(**kwargs: object) -> AsyncIterator[CodeMode]:
    """A code-mode handle that never leaves a kernel process behind."""
    mode = CodeMode(**kwargs)  # type: ignore[arg-type]
    try:
        yield mode
    finally:
        await mode.dispose()


async def _cell(
    mode: CodeMode,
    code: str,
    *,
    signal: ToolCancellationToken | None = None,
    **extra: JSONValue,
) -> AgentToolResult:
    arguments: dict[str, JSONValue] = {"code": code, "description": "Exercise code mode", **extra}
    return await mode.tool().execute("call-1", arguments, signal)


def _text(result: AgentToolResult) -> str:
    return result.text


# -- the tool contract ------------------------------------------------------


def test_the_tool_is_named_and_runs_sequentially(tmp_path: Path) -> None:
    tool = CodeMode(cwd=tmp_path).tool()
    assert tool.name == PYTHON_REPL_TOOL_NAME
    assert tool.label == "python"
    # One namespace: two cells cannot interleave inside it.
    assert tool.execution_mode == "sequential"
    assert tool.parameters["required"] == ["code", "description"]
    assert "timeout" in tool.parameters["properties"]
    assert "attach" in tool.parameters["properties"]
    assert tool.prompt_guidelines


async def test_the_namespace_persists_between_calls(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        first = await _cell(mode, "rows = ['a', 'b', 'c']\nprint(len(rows))")
        assert "3" in _text(first)
        second = await _cell(mode, "print(rows[1].upper())")
        assert "B" in _text(second)
        assert mode.started is True


async def test_ret_returns_data_rather_than_prose(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        result = await _cell(mode, "ret({'count': 2, 'note': 'ok'})")
        assert result.details["returned"] is True
        assert result.details["ret"] == {"count": 2, "note": "ok"}
        assert "[ret]" in _text(result)


async def test_a_non_data_ret_is_rejected_in_words(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        result = await _cell(mode, "import io\nret(io.BytesIO())")
        assert result.details["returned"] is True
        assert "JSON" in str(result.details["ret_error"])
        assert "[ret rejected" in _text(result)


async def test_a_raising_cell_reports_the_error_and_survives(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        bad = await _cell(mode, "raise ValueError('boom')")
        assert bad.details["ok"] is False
        assert "boom" in str(bad.details["error"])
        assert bad.details["kernel"]["alive"] is True
        after = await _cell(mode, "print('still here')")
        assert "still here" in _text(after)


async def test_show_vars_lists_names_without_values(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        result = await _cell(mode, "secret_number = 123456\nprint(SHOW_VARS())")
        text = _text(result)
        assert "secret_number" in text
        assert "llm_query" in text
        assert "123456" not in text


async def test_output_is_capped_and_says_what_was_dropped(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        result = await _cell(mode, "for i in range(5000):\n    print(i)")
        assert "[showing first 400 of 5000 lines]" in _text(result)


async def test_an_overrun_cell_is_killed_with_its_namespace(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        await _cell(mode, "answer = 7")
        slow = await _cell(mode, "import time\ntime.sleep(5)", timeout=0.5)
        assert slow.details["timed_out"] is True
        assert slow.details["state_lost"] is True
        assert "namespace is gone" in _text(slow)
        after = await _cell(mode, "print(answer)")
        assert "NameError" in str(after.details["error"])
        assert after.details["kernel"]["restarts"] >= 1


async def test_a_cancelled_cell_is_killed_and_says_so(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        token = SimpleCancellationToken()
        running = asyncio.create_task(_cell(mode, "import time\ntime.sleep(5)", signal=token))
        await asyncio.sleep(0.3)
        token.cancel()
        result = await asyncio.wait_for(running, timeout=15)
        assert result.details["cancelled"] is True
        assert result.details["kernel"]["alive"] is False
        assert "[cell cancelled]" in _text(result)


async def test_a_cell_cancelled_while_the_kernel_boots_is_reported(tmp_path: Path) -> None:
    # The watch task can fire before the child finishes its handshake; the cell has
    # to come back cancelled rather than wait out the startup timeout.
    async with _mode(cwd=tmp_path) as mode:
        token = SimpleCancellationToken()
        running = asyncio.create_task(_cell(mode, "print('never runs')", signal=token))
        # One yield: the cell has passed its pre-flight check and is inside kernel
        # startup, which is where the abort has to land.
        await asyncio.sleep(0)
        token.cancel()
        result = await asyncio.wait_for(running, timeout=15)
        assert result.details["cancelled"] is True
        assert "never runs" not in _text(result)


async def test_a_call_cancelled_before_the_cell_starts_is_refused(tmp_path: Path) -> None:
    token = SimpleCancellationToken()
    token.cancel()
    with pytest.raises(ToolInputError, match="cancelled"):
        await _cell(CodeMode(cwd=tmp_path), "print(1)", signal=token)


async def test_sub_calls_are_reported_with_their_cost(tmp_path: Path) -> None:
    answer = "two words"
    stream = [
        assistant_start(),
        text_delta(answer),
        assistant_done(AssistantMessage(model="fake", content=answer)),
    ]
    provider = FakeProvider([stream])  # type: ignore[arg-type]
    async with _mode(
        cwd=tmp_path,
        subcall=SubcallBridge(provider=lambda: provider, model=lambda: "fake"),
    ) as mode:
        result = await _cell(mode, 'print(llm_query("count these words"))')
        assert answer in _text(result)
        subcalls = result.details["subcalls"]
        assert subcalls["calls"] == 1
        assert subcalls["models"] == ["fake"]
        assert result.details["kernel"]["subcalls_used"] == 1
        assert "1 sub-call via fake" in _text(result)


async def test_a_closed_mode_refuses_to_run_again(tmp_path: Path) -> None:
    mode = CodeMode(cwd=tmp_path)
    try:
        await _cell(mode, "print(1)")
        await mode.dispose()
        assert mode.started is False
        closed = await _cell(mode, "print(2)")
        assert closed.details["ok"] is False
        assert "code mode is closed" in _text(closed)
    finally:
        await mode.dispose()


async def test_empty_and_oversized_cells_are_refused_before_the_kernel(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        with pytest.raises(ToolInputError, match="non-empty"):
            await _cell(mode, "   ")
        with pytest.raises(ToolInputError, match="limit"):
            await _cell(mode, "x = 1\n" * (MAX_CODE_BYTES // 6 + 2))
        assert mode.started is False


async def test_a_bad_timeout_is_refused(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        with pytest.raises(ToolInputError, match="timeout"):
            await _cell(mode, "print(1)", timeout=0)


# -- file handles (attach) --------------------------------------------------


async def test_attach_binds_a_path_not_a_payload(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("line one\nneedle here\n", encoding="utf-8")
    async with _mode(cwd=tmp_path) as mode:
        result = await _cell(
            mode,
            "with open(corpus, encoding='utf-8') as handle:\n"
            "    text = handle.read()\n"
            "print([line for line in text.splitlines() if 'needle' in line])",
            attach={"corpus": str(corpus)},
        )
        assert "needle here" in _text(result)
        assert "[attached as paths: corpus (" in _text(result)
        assert result.details["attachments"] == [
            {"name": "corpus", "bytes": len(corpus.read_bytes())}
        ]


async def test_a_handle_can_be_rebound_after_the_namespace_dies(tmp_path: Path) -> None:
    # The handle is a name in the namespace, not a durable resource, so a kernel
    # killed by an overrun loses it and the next call must restate it.
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("kept\n", encoding="utf-8")
    async with _mode(cwd=tmp_path) as mode:
        first = await _cell(
            mode, "print(open(corpus, encoding='utf-8').read())", attach={"corpus": str(corpus)}
        )
        assert "kept" in _text(first)
        killed = await _cell(mode, "import time\ntime.sleep(5)", timeout=0.5)
        assert killed.details["state_lost"] is True
        lost = await _cell(mode, "print(open(corpus, encoding='utf-8').read())")
        assert "NameError" in str(lost.details["error"])
        rebound = await _cell(
            mode, "print(open(corpus, encoding='utf-8').read())", attach={"corpus": str(corpus)}
        )
        assert "kept" in _text(rebound)


async def test_a_refused_handle_fails_the_call_before_the_cell_runs(tmp_path: Path) -> None:
    async with _mode(cwd=tmp_path) as mode:
        with pytest.raises(ToolInputError, match="not a regular file"):
            await _cell(mode, "print(1)", attach={"where": str(tmp_path)})
        with pytest.raises(ToolInputError, match="identifier"):
            await _cell(mode, "print(1)", attach={"not an id": str(tmp_path / "x.txt")})
        with pytest.raises(ToolInputError, match="kernel binding"):
            await _cell(mode, "print(1)", attach={RESERVED[0]: str(tmp_path / "x.txt")})
        with pytest.raises(ToolInputError, match="could not be read"):
            await _cell(mode, "print(1)", attach={"gone": str(tmp_path / "gone.txt")})
        # Refused before any cell ran: the kernel object exists but executed nothing.
        assert mode._kernel is not None and mode._kernel.cells_run == 0


async def test_an_oversized_handle_is_named_not_loaded(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    big.write_bytes(b"x" * (MAX_ATTACHED_FILE_BYTES + 1))
    async with _mode(cwd=tmp_path) as mode:
        with pytest.raises(ToolInputError, match="Slice the file"):
            await _cell(mode, "print(1)", attach={"big": str(big)})
        assert mode._kernel is None or mode._kernel.cells_run == 0


# -- session scope ----------------------------------------------------------


def _session_config(tmp_path: Path, **kwargs: object) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=FakeProvider([]),  # type: ignore[arg-type]
        model="fake",
        system="You are Tau.",
        storage=JsonlSessionStorage(tmp_path / "sessions"),
        session_manager=SessionManager(),
        cwd=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )


async def test_python_repl_is_part_of_the_default_tool_set(tmp_path: Path) -> None:
    session = await CodingSession.load(_session_config(tmp_path))
    try:
        assert PYTHON_REPL_TOOL_NAME in {tool.name for tool in session.tools}
        assert session.code_mode is not None
        # The kernel is lazy: a session that never runs a cell never pays for a process.
        assert session.code_mode.started is False
    finally:
        await session.aclose()


async def test_an_explicit_tool_set_owns_its_tools(tmp_path: Path) -> None:
    session = await CodingSession.load(_session_config(tmp_path, tools=[]))
    try:
        assert PYTHON_REPL_TOOL_NAME not in {tool.name for tool in session.tools}
        assert session.code_mode is None
    finally:
        await session.aclose()


async def test_code_mode_can_be_turned_off(tmp_path: Path) -> None:
    session = await CodingSession.load(_session_config(tmp_path, code_mode_enabled=False))
    try:
        assert PYTHON_REPL_TOOL_NAME not in {tool.name for tool in session.tools}
        assert session.code_mode is None
    finally:
        await session.aclose()


async def test_the_allow_list_can_drop_the_tool(tmp_path: Path) -> None:
    session = await CodingSession.load(
        _session_config(tmp_path, allowed_tool_names=frozenset({"read", "bash"}))
    )
    try:
        assert PYTHON_REPL_TOOL_NAME not in {tool.name for tool in session.tools}
    finally:
        await session.aclose()


async def test_reload_keeps_the_same_namespace(tmp_path: Path) -> None:
    session = await CodingSession.load(_session_config(tmp_path))
    try:
        mode = session.code_mode
        assert mode is not None
        await _cell(mode, "carried = 'kept'")
        await session.reload()
        assert session.code_mode is mode
        after = await _cell(session.code_mode, "print(carried)")
        assert "kept" in _text(after)
    finally:
        await session.aclose()


async def test_closing_the_session_closes_the_kernel(tmp_path: Path) -> None:
    session = await CodingSession.load(_session_config(tmp_path))
    mode = session.code_mode
    assert mode is not None
    await _cell(mode, "print('one cell')")
    assert mode.started is True
    await session.aclose()
    assert mode.started is False


async def test_a_new_session_closes_the_old_kernel(tmp_path: Path) -> None:
    # A new conversation is a new namespace: the old kernel dies rather than
    # lingering as a process whose working memory belongs to a dead transcript.
    session = await CodingSession.load(_session_config(tmp_path))
    try:
        old = session.code_mode
        assert old is not None
        await _cell(old, "carried = 'old conversation'")
        assert old.started is True
        await session.new_session()
        assert old.started is False
        fresh = session.code_mode
        assert fresh is not None and fresh is not old
        result = await _cell(fresh, "print('fresh namespace')")
        assert "fresh namespace" in _text(result)
    finally:
        await session.aclose()
