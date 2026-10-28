"""End-to-end tests for the deterministic compaction extension.

These drive a real :class:`CodingSession` with the shipped example extension
loaded, so they cover the whole chain -- loader, ``before_compact`` seam,
engine, session append, replay -- instead of the engine in isolation. The engine
itself is unit-tested in ``test_compaction_*.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pi_event_helpers import (
    assistant_done,
    assistant_start,
    text_delta,
    thinking_delta,
    tool_call_end,
)
from tau_agent.messages import AssistantMessage, TextContent, ThinkingContent, ToolCall
from tau_agent.session import JsonlSessionStorage
from tau_agent.session.entries import CompactionEntry
from tau_ai import FakeProvider
from tau_coding.compaction import utf8_bytes
from tau_coding.session import CodingSession, CodingSessionConfig

# async tests run on asyncio, matching the rest of the suite
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


ENGINE_EXTENSION = (
    Path(__file__).resolve().parents[1] / "examples" / "extensions" / "deterministic_compaction.py"
)

_THINKING = "Weighing whether to mention the retry budget."
_REQUEST = "Fix the retry budget in gateway.py; 한국어 메모도 남기세요."


def _config(tmp_path: Path, provider: FakeProvider, storage: JsonlSessionStorage):
    return CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Tau.",
        storage=storage,
        cwd=tmp_path,
        extension_paths=(ENGINE_EXTENSION,),
    )


def _provider() -> FakeProvider:
    read_call = ToolCall(id="call-read", name="read", arguments={"path": "gateway.py"})
    bash_call = ToolCall(id="call-bash", name="bash", arguments={"command": "test -f nope.py"})
    first = AssistantMessage(
        content=[
            ThinkingContent(thinking=_THINKING),
            TextContent(text="Reading the gateway."),
            read_call,
        ],
        model="fake",
    )
    second = AssistantMessage(
        content=[
            TextContent(text="The retry budget is three; raise it to four."),
            bash_call,
        ],
        model="fake",
    )
    final = AssistantMessage(content="Done.", model="fake")
    return FakeProvider(
        [
            [
                assistant_start(),
                thinking_delta(_THINKING),
                text_delta("Reading the gateway."),
                tool_call_end(read_call),
                assistant_done(first, "toolUse"),
            ],
            [
                assistant_start(),
                text_delta("The retry budget is three; raise it to four."),
                tool_call_end(bash_call),
                assistant_done(second, "toolUse"),
            ],
            [assistant_start(), text_delta("Done."), assistant_done(final)],
        ]
    )


async def _compacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run a small but structured session, compact it, and hand back the parts."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "gateway.py").write_text("RETRY_BUDGET = 3\n", encoding="utf-8")
    storage = JsonlSessionStorage(tmp_path / "sessions")
    provider = _provider()
    session = await CodingSession.load(_config(tmp_path, provider, storage))
    assert "deterministic_compaction" in session.extension_names

    async for _ in session.prompt(_REQUEST):
        pass
    calls_before_compaction = len(provider.calls)
    # The core keeps whatever fits this budget; one token keeps nothing, so the
    # engine has to plan a real cut rather than inherit a trivial one.
    monkeypatch.setattr("tau_coding.session.DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", 1)
    await session.compact_detailed()

    entries = await storage.read_all()
    compactions = [entry for entry in entries if isinstance(entry, CompactionEntry)]
    assert len(compactions) == 1
    return session, provider, compactions[0], calls_before_compaction


def _normalized(summary: str) -> str:
    """Mask the two things the engine does not control: the clock and entry ids."""
    summary = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", "<clock>", summary)
    return re.sub(r"\b[0-9a-f]{32}\b", "<id>", summary)


async def test_the_engine_writes_the_summary_without_a_second_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, provider, entry, calls_before = await _compacted(tmp_path, monkeypatch)

    assert entry.details is not None
    assert entry.details["engine"] == "deterministic"
    assert entry.details["decision"] == "replace"
    # Compaction added no provider call at all, which is the point of the engine:
    # the agent loop's own turns are the only model traffic in this session.
    assert calls_before >= 1
    assert len(provider.calls) == calls_before
    # Replay: the active context now starts at the engine's boundary, with the
    # summary standing in for every entry it replaced.
    active = session.state.context_entry_ids
    assert entry.first_kept_entry_id in active
    assert active[0] in {entry.id, entry.first_kept_entry_id}


async def test_the_summary_is_typed_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, _provider_, entry, _calls = await _compacted(tmp_path, monkeypatch)
    details = entry.details
    assert details is not None

    assert "Fix the retry budget in gateway.py" in entry.summary
    assert "한국어 메모도 남기세요." in entry.summary
    assert "[Current Status]" in entry.summary
    assert "/tree branches from it" in entry.summary
    assert details["summary_bytes"] == utf8_bytes(entry.summary)
    assert details["summarized_entries"] >= 1


async def test_assistant_thinking_is_erased_and_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, _provider_, entry, _calls = await _compacted(tmp_path, monkeypatch)
    details = entry.details
    assert details is not None

    assert _THINKING not in entry.summary
    assert details["thinking_blocks_erased"] == 1


async def test_the_engine_qas_its_own_summary_before_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _session, _provider_, entry, _calls = await _compacted(tmp_path, monkeypatch)
    details = entry.details
    assert details is not None

    qa = details["qa"]
    assert isinstance(qa, dict)
    assert qa["probes"] >= 3
    # A summary is only written when nothing went silently missing.
    assert qa["content_score"] == 1.0
    assert qa["address_score"] == 1.0


async def test_the_same_entries_compile_to_the_same_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _first, _first_provider, first, _calls_a = await _compacted(tmp_path / "a", monkeypatch)
    _second, _second_provider, second, _calls_b = await _compacted(tmp_path / "b", monkeypatch)

    # Entry ids are random per session and the footer prints a timestamp, so those
    # are the only degrees of freedom. With them masked, two independent runs of
    # the same turns produce the same bytes -- an LLM summarizer cannot promise
    # that, and it is why the summary can be diffed and QA-checked at all.
    assert _normalized(first.summary) == _normalized(second.summary)
    assert first.details is not None and second.details is not None
    assert first.details["summary_bytes"] == second.details["summary_bytes"]


async def test_the_status_command_reports_the_last_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, _provider_, _entry, _calls = await _compacted(tmp_path, monkeypatch)
    registry = session.extension_runtime.build_command_registry()

    result = registry.execute(session, "/compact-engine")

    assert result.handled
    message = result.message or ""
    assert "decision=replace" in message
    assert "qa:" in message
