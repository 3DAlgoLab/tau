"""Session-level tests for the `session_before_compact` extension hook.

These cover the contract the transparent-compact port depends on: an extension
replaces Tau's model-generated compaction summary with deterministic text plus
canonical `details`, sees the raw branch (not the projected context), and can
cancel a compaction outright.
"""

from pathlib import Path

import pytest

from conftest import isolate_home
from pi_event_helpers import assistant_done, assistant_start
from tau_agent import AssistantMessage
from tau_agent.session import CompactionEntry, CustomEntry, JsonlSessionStorage
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig
from tau_coding import session as coding_session_module

CHECKPOINT_SUMMARY = "CHECKPOINT verbatim: [User]: Earlier work."
CHECKPOINT_DETAILS: dict[str, object] = {
    "kind": "mech-compact",
    "span": {"lines": ["[User]: Earlier work."]},
}

_OVERRIDE_EXTENSION = """
from tau_coding.extensions import SessionBeforeCompactResult


def setup(tau):
    async def before_compact(event, context):
        first = event.messages_to_summarize[0] if event.messages_to_summarize else None
        await tau.append_entry(
            "hook:record",
            {
                "reason": event.reason,
                "first_kept_entry_id": event.first_kept_entry_id,
                "replaced": list(event.replaced_entry_ids),
                "summarized_count": len(event.messages_to_summarize),
                "first_message": first.text if first is not None else "",
                "tokens_before": event.tokens_before,
                "context_window_tokens": event.context_window_tokens,
                "instructions": event.custom_instructions,
                "branch_entries": len(context.entries),
                "projected_messages": len(context.transcript),
            },
        )
        return SessionBeforeCompactResult(
            summary=__SUMMARY__,
            details=__DETAILS__,
        )

    tau.on("session_before_compact", before_compact)
""".lstrip()

_CANCEL_EXTENSION = """
from tau_coding.extensions import SessionBeforeCompactResult


def setup(tau):
    tau.on("session_before_compact", lambda event, context: SessionBeforeCompactResult(cancel=True))
""".lstrip()

_RAISING_EXTENSION = """
def setup(tau):
    def before_compact(event, context):
        raise RuntimeError("checkpoint builder exploded")

    tau.on("session_before_compact", before_compact)
""".lstrip()


def _write_extension(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    return path


def _checkpoint_extension(tmp_path: Path) -> Path:
    """The override hook, with the checkpoint text/details spliced in."""
    source = _OVERRIDE_EXTENSION.replace("__SUMMARY__", repr(CHECKPOINT_SUMMARY)).replace(
        "__DETAILS__", repr(CHECKPOINT_DETAILS)
    )
    return _write_extension(tmp_path, "checkpoint.py", source)


def _config(
    tmp_path: Path,
    provider: FakeProvider,
    storage: JsonlSessionStorage,
    extension: Path,
    **overrides: object,
) -> CodingSessionConfig:
    return CodingSessionConfig(
        provider=provider,
        model="fake",
        system="You are Tau.",
        storage=storage,
        cwd=tmp_path,
        extension_paths=(extension,),
        extensions_enabled=False,
        **overrides,
    )


async def _drain(stream: object) -> list[object]:
    return [event async for event in stream]  # type: ignore[attr-defined]


def _records(entries: list[CustomEntry]) -> list[dict[str, object]]:
    return [entry.data for entry in entries if isinstance(entry, CustomEntry)]


def _answer(text: str) -> list[object]:
    return [
        assistant_start(model="fake"),
        assistant_done(message=AssistantMessage(content=text)),
    ]


def _compactions(entries: list[object]) -> list[CompactionEntry]:
    return [entry for entry in entries if isinstance(entry, CompactionEntry)]


@pytest.mark.anyio
async def test_extension_summary_replaces_manual_compaction_without_model_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr(coding_session_module, "DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", 1)
    extension = _checkpoint_extension(tmp_path)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    # Only the prompt itself is scripted: any summarization request would find
    # an exhausted FakeProvider and fail the test.
    provider = FakeProvider([_answer("Answer")])
    session = await CodingSession.load(_config(tmp_path, provider, storage, extension))
    await _drain(session.prompt("Earlier work."))

    result = await session.compact_detailed("Focus on the port.")

    assert result.summary == CHECKPOINT_SUMMARY
    assert provider.calls and len(provider.calls) == 1
    # The summary lands at the boundary of the replaced span, not at the tail.
    assert session.messages[0].text == f"Previous conversation summary:\n{CHECKPOINT_SUMMARY}"
    assert [message.role for message in session.messages] == ["user", "assistant"]

    entries = await storage.read_all()
    compactions = _compactions(entries)
    assert len(compactions) == 1
    assert compactions[0].summary == CHECKPOINT_SUMMARY
    assert compactions[0].details == CHECKPOINT_DETAILS
    # No provider spend belongs to a mechanical summary.
    assert compactions[0].provider is None
    assert compactions[0].model is None
    assert compactions[0].usage is None

    record = next(data for data in _records(entries) if "reason" in data)
    assert record["reason"] == "manual"
    assert record["instructions"] == "Focus on the port."
    assert record["first_kept_entry_id"] == compactions[0].first_kept_entry_id
    assert record["replaced"]  # the plan's replaced active-context entries
    assert record["first_message"] == "Earlier work."
    assert record["tokens_before"] > 0
    assert record["context_window_tokens"] == 128_000
    # Nothing is compacted yet, so the branch is only slightly ahead of the
    # projection (bookkeeping entries are not model messages).
    assert record["branch_entries"] >= record["projected_messages"]

    await _drain(session.prompt("Continue."))
    assert len(provider.calls) == 2
    assert provider.calls[1][2][0].content == (
        f"Previous conversation summary:\n{CHECKPOINT_SUMMARY}"
    )

    # Durability: a reopened session replays the same projection from the
    # extension-authored summary plus its canonical details.
    await session.aclose()
    reopened = await CodingSession.load(_config(tmp_path, FakeProvider([]), storage, extension))
    assert reopened.messages[0].text == f"Previous conversation summary:\n{CHECKPOINT_SUMMARY}"
    assert [entry.details for entry in _compactions(await storage.read_all())] == [
        CHECKPOINT_DETAILS
    ]
    await reopened.aclose()


@pytest.mark.anyio
async def test_extension_can_cancel_a_manual_compaction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr(coding_session_module, "DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", 1)
    extension = _write_extension(tmp_path, "canceller.py", _CANCEL_EXTENSION)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = FakeProvider([_answer("Answer")])
    session = await CodingSession.load(_config(tmp_path, provider, storage, extension))
    await _drain(session.prompt("Earlier work."))
    context_entries = set(session.state.context_entry_ids)

    with pytest.raises(ValueError, match="Compaction cancelled by an extension"):
        await session.compact_detailed()

    assert set(session.state.context_entry_ids) == context_entries
    assert _compactions(await storage.read_all()) == []
    await session.aclose()


@pytest.mark.anyio
async def test_raising_hook_falls_back_to_tau_summarization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr(coding_session_module, "DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", 1)
    extension = _write_extension(tmp_path, "raiser.py", _RAISING_EXTENSION)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = FakeProvider([_answer("Answer"), _answer("Model summary")])
    session = await CodingSession.load(_config(tmp_path, provider, storage, extension))
    await _drain(session.prompt("Earlier work."))

    result = await session.compact_detailed()

    assert result.summary == "Model summary"
    assert any(
        "session_before_compact" in diagnostic.message
        for diagnostic in session.resource_diagnostics
    )
    # The fallback summarizer is a model call, so it is attributed as usual.
    assert provider.calls[1][1].startswith("You are a context summarization assistant.")
    compactions = _compactions(await storage.read_all())
    assert compactions[0].details is None
    assert compactions[0].model == "fake"
    await session.aclose()


@pytest.mark.anyio
async def test_auto_compaction_override_tiles_without_model_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Threshold compaction routes through the hook, and the second checkpoint
    sees the previous checkpoint as its base plus the still-lossless branch."""
    isolate_home(monkeypatch, tmp_path)
    monkeypatch.setattr(coding_session_module, "DEFAULT_COMPACTION_KEEP_RECENT_TOKENS", 1)
    extension = _checkpoint_extension(tmp_path)
    storage = JsonlSessionStorage(tmp_path / "session.jsonl")
    provider = FakeProvider(
        [_answer("First answer"), _answer("Second answer"), _answer("Third answer")]
    )
    session = await CodingSession.load(
        _config(tmp_path, provider, storage, extension, auto_compact_token_threshold=1)
    )
    await _drain(session.prompt("Earlier work."))
    await _drain(session.prompt("Continue."))
    await _drain(session.prompt("Next."))

    # Three prompts, three requests: no request was ever spent summarizing.
    assert len(provider.calls) == 3
    assert not any(
        call[1].startswith("You are a context summarization assistant.") for call in provider.calls
    )

    entries = await storage.read_all()
    compactions = _compactions(entries)
    assert len(compactions) >= 2
    assert all(entry.summary == CHECKPOINT_SUMMARY for entry in compactions)

    records = [data for data in _records(entries) if "reason" in data]
    assert len(records) == len(compactions)
    assert {str(record["reason"]) for record in records} == {"threshold"}
    assert records[0]["branch_entries"] >= records[0]["projected_messages"]

    # At least one later compaction runs against an already-compacted context:
    # its projected context is shorter than the still-lossless branch, and its
    # base is the checkpoint Tau rendered for the previous one.
    carried = [
        record
        for record in records[1:]
        if str(record["first_message"]).startswith(
            f"Previous conversation summary:\n{CHECKPOINT_SUMMARY}"
        )
    ]
    assert carried
    for record in carried:
        assert record["projected_messages"] < record["branch_entries"]
    # Every compaction keeps a real boundary in the live context.
    for record, compaction in zip(records, compactions, strict=True):
        assert record["first_kept_entry_id"] == compaction.first_kept_entry_id
        assert compaction.first_kept_entry_id not in record["replaced"]
    await session.aclose()
