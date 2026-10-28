"""Tests for the deterministic section folds over a compaction event stream."""

from __future__ import annotations

import dataclasses

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_coding.compaction.bounds import utf8_bytes
from tau_coding.compaction.dialogue import MAX_EXCHANGE_BYTES, MAX_RECENT_EXCHANGES
from tau_coding.compaction.events import CompactionEvent, ContextRow, normalize_rows
from tau_coding.compaction.projections import (
    MAX_FILES_PER_KIND,
    MAX_REQUESTS,
    TRANSCRIPT_WINDOW,
    collect_operations,
    project,
    project_with_metadata,
    truncate,
)


def _rows(*pairs: tuple[str, AgentMessage]) -> list[ContextRow]:
    return [(entry_id, message) for entry_id, message in pairs]


def _events(*pairs: tuple[str, AgentMessage]) -> list[CompactionEvent]:
    return list(normalize_rows(_rows(*pairs)))


def _assistant(*text: str, calls: tuple[ToolCall, ...] = ()) -> AssistantMessage:
    content: list[object] = [TextContent(text=body) for body in text]
    content.extend(calls)
    return AssistantMessage(content=content)  # type: ignore[arg-type]


def _call(call_id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))  # type: ignore[arg-type]


def _result(
    call_id: str,
    name: str,
    text: str,
    *,
    is_error: bool = False,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name=name,
        content=[TextContent(text=text)],
        is_error=is_error,
    )


def _tool_turn(
    rows: list[ContextRow],
    entry_id: str,
    call_id: str,
    name: str,
    arguments: dict[str, object],
    output: str,
    *,
    is_error: bool = False,
) -> None:
    rows.append((f"{entry_id}a", _assistant(calls=(_call(call_id, name, **arguments),))))
    rows.append((f"{entry_id}r", _result(call_id, name, output, is_error=is_error)))


def test_operations_pair_results_with_the_call_that_requested_them() -> None:
    events = _events(
        ("e1", _assistant(calls=(_call("c1", "read", path="a.py"),))),
        ("e2", _result("c1", "read", "contents")),
        ("e3", _result("ghost", "read", "orphan result")),
    )

    operations = collect_operations(events)

    assert [
        (operation.tool, operation.entry_id, operation.outcome) for operation in operations
    ] == [
        ("read", "e1", "succeeded"),
    ]


def test_bash_operations_take_their_command_from_the_call() -> None:
    events = _events(
        ("e1", _assistant(calls=(_call("c1", "bash", command="pytest -q"),))),
        ("e2", _result("c1", "bash", "2 failed", is_error=True)),
    )

    operation = collect_operations(events)[0]

    assert operation.tool == "bash"
    assert operation.arguments == {"command": "pytest -q"}
    assert operation.outcome == "failed"
    assert operation.error == "2 failed"


def test_files_section_labels_kinds_and_drops_reads_of_changed_files() -> None:
    rows: list[ContextRow] = [
        ("e1", _assistant(calls=(_call("c1", "read", path="src/app.py"),))),
        ("e2", _result("c1", "read", "def main(): ...")),
        ("e3", _assistant(calls=(_call("c2", "edit", path="src/app.py"),))),
        ("e4", _result("c2", "edit", "replaced 1 block")),
        ("e5", _assistant(calls=(_call("c3", "write", path="src/new.py"),))),
        ("e6", _result("c3", "write", "wrote file")),
        ("e7", _assistant(calls=(_call("c4", "read", path="src/other.py"),))),
        ("e8", _result("c4", "read", "contents")),
    ]

    lines = project(_events(*rows)).files

    assert "(under src/)" in lines
    assert "Written:" in lines
    assert "  new.py [entry e5]" in lines
    assert "Modified:" in lines
    assert "  app.py [entry e3]" in lines
    assert "Read:" in lines
    assert "  other.py [entry e7]" in lines
    assert "app.py [entry e1]" not in lines


def test_failed_file_operations_are_not_reported_as_changes() -> None:
    lines = project(
        _events(
            ("e1", _assistant(calls=(_call("c1", "write", path="src/denied.py"),))),
            ("e2", _result("c1", "write", "permission denied", is_error=True)),
        )
    ).files

    assert lines == ()


def test_files_section_addresses_omitted_paths() -> None:
    rows: list[ContextRow] = []
    for index in range(MAX_FILES_PER_KIND + 8):
        _tool_turn(rows, f"t{index}", f"c{index}", "write", {"path": f"src/f{index}.py"}, "ok")

    projection = project_with_metadata(_events(*rows))
    lines = projection.sections.files

    assert projection.omitted.files == 8
    assert "  … omitted 8 file addresses; source entries t12a → t19a" in lines, (
        "the omitted middle must stay addressed"
    )


def test_activity_fold_skips_file_tools_and_keeps_shell_work() -> None:
    rows: list[ContextRow] = [
        ("e1", _assistant(calls=(_call("c1", "read", path="a.py"),))),
        ("e2", _result("c1", "read", "text")),
        ("e3", _assistant(calls=(_call("c2", "bash", command="uv run pytest -q"),))),
        ("e4", _result("c2", "bash", "all passed")),
    ]

    lines = project(_events(*rows)).activity

    assert lines == ("- bash (uv run pytest -q) -> succeeded [entry e4]",)


def test_a_later_success_resolves_only_the_same_typed_identity() -> None:
    rows: list[ContextRow] = [
        ("e1", _assistant(calls=(_call("c1", "bash", command="uv run pytest -q"),))),
        ("e2", _result("c1", "bash", "collection error", is_error=True)),
        ("e3", _assistant(calls=(_call("c2", "bash", command="uv run pytest -q"),))),
        ("e4", _result("c2", "bash", "all passed")),
        ("e5", _assistant(calls=(_call("c3", "edit", path="src/a.py"),))),
        ("e6", _result("c3", "edit", "boom", is_error=True)),
        ("e7", _assistant(calls=(_call("c4", "edit", path="src/b.py"),))),
        ("e8", _result("c4", "edit", "ok")),
    ]

    lines = project(_events(*rows)).outstanding

    assert any("bash: uv run pytest -q: collection error [RESOLVED]" in line for line in lines)
    assert any("edit src/a.py: boom" in line and "[RESOLVED]" not in line for line in lines)


def test_earlier_turns_collapse_to_one_addressed_line_each() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="first request")),
        ("e2", _assistant(calls=(_call("c1", "read", path="a.py"),))),
        ("e3", _result("c1", "read", "text")),
        ("e4", UserMessage(content="second request")),
        ("e5", _assistant(calls=(_call("c2", "bash", command="ls"),))),
        ("e6", _result("c2", "bash", "a.py")),
        ("e7", UserMessage(content="third request")),
    ]

    lines = project(_events(*rows)).earlier_turns

    assert lines[0] == '"first request" | read:1 [entry e1]'
    assert lines[1] == '"second request" | bash:1 [entry e4]'


def test_custom_messages_open_turns_and_keep_their_payload_visible() -> None:
    rows: list[ContextRow] = [
        ("e1", CustomMessage(custom_type="myext.note", content="context", display=False)),
        ("e2", UserMessage(content="next request")),
    ]

    lines = project(_events(*rows)).earlier_turns

    assert 'custom "myext.note" (hidden): "context"' in lines[0]
    assert lines[0].endswith("[entry e1]")


def test_prior_summary_events_stay_as_addresses_not_content() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="first")),
        ("e2", UserMessage(content="Previous conversation summary:\nolder session facts")),
        ("e3", UserMessage(content="latest")),
    ]

    lines = project(_events(*rows)).earlier_turns

    assert "- prior compaction summary [entry e2]" in lines
    assert "older session facts" not in "\n".join(lines)


def test_status_bridges_the_summarized_window_into_the_kept_tail() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="port the renderer")),
        ("e2", _assistant(calls=(_call("c1", "write", path="src/render.py"),))),
        ("e3", _result("c1", "write", "wrote it")),
        ("e4", _assistant(calls=(_call("c2", "bash", command="uv run pytest -q"),))),
        ("e5", _result("c2", "bash", "passed")),
        ("e6", _assistant("Renderer is in place.")),
    ]

    lines = project(_events(*rows)).status

    assert lines == (
        "Last request: port the renderer",
        "Last change: write src/render.py",
        "Last command: bash uv run pytest -q -> succeeded",
        "Last note: Renderer is in place.",
    )


def test_transcript_window_keeps_stable_pointers_and_addresses_the_head() -> None:
    rows: list[ContextRow] = [
        (f"u{index}", UserMessage(content=f"request {index}"))
        for index in range(TRANSCRIPT_WINDOW + 5)
    ]

    projection = project_with_metadata(_events(*rows))
    lines = projection.sections.transcript

    assert projection.omitted.transcript == 5
    assert lines[0] == "… omitted 5 transcript events; source entries u0 → u4"
    assert lines[1].startswith(f"(#{6}) user: request 5")
    assert lines[-1].startswith(f"(#{TRANSCRIPT_WINDOW + 5}) user: request {TRANSCRIPT_WINDOW + 4}")


def test_requests_fold_protects_entries_the_dialogue_section_keeps() -> None:
    total = MAX_RECENT_EXCHANGES + 2
    rows: list[ContextRow] = []
    for index in range(total):
        rows.append((f"e{index}", UserMessage(content=f"request {index}")))
        rows.append((f"r{index}", _assistant(f"reply {index}")))

    projection = project_with_metadata(_events(*rows))

    dialogue = "\n".join(projection.sections.dialogue)
    requests = "\n".join(projection.sections.requests)
    kept = range(total - MAX_RECENT_EXCHANGES, total)
    dropped = range(total - MAX_RECENT_EXCHANGES)
    assert all(f"request {index}" in dialogue for index in kept)
    assert all(f"request {index} [entry e{index}]" in requests for index in dropped)
    assert all(f"request {index}" not in requests for index in kept)


def test_requests_are_sampled_head_and_tail_with_an_addressed_gap() -> None:
    total = MAX_REQUESTS + MAX_RECENT_EXCHANGES + 4
    rows: list[ContextRow] = [
        (f"e{index}", UserMessage(content=f"request number {index}")) for index in range(total)
    ]

    projection = project_with_metadata(_events(*rows))
    lines = projection.sections.requests
    source_count = total - MAX_RECENT_EXCHANGES
    dropped = source_count - MAX_REQUESTS
    head = -(-MAX_REQUESTS // 2)

    assert projection.omitted.requests == dropped
    assert lines[0] == "- request number 0 [entry e0]"
    assert lines[head] == (
        f"… omitted {dropped} user scope changes; "
        f"source entries e{head} → e{source_count - head - 1}"
    )
    assert lines[-1] == f"- request number {source_count - 1} [entry e{source_count - 1}]"


def test_projection_is_deterministic_and_line_bounded() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="요약 알고리즘을 포팅하세요" * 40)),
        ("e2", _assistant("읽는 중입니다" * 60)),
        ("e3", _assistant(calls=(_call("c1", "bash", command="ls " + "x" * 500),))),
        ("e4", _result("c1", "bash", "y" * 4000, is_error=True)),
        ("e5", UserMessage(content="확인" * 30)),
    ]

    events = _events(*rows)
    first = project_with_metadata(events)
    second = project_with_metadata(events)

    assert first == second
    for field in dataclasses.fields(first.sections):
        ceiling = MAX_EXCHANGE_BYTES + 256 if field.name == "dialogue" else 1024
        for line in getattr(first.sections, field.name):
            assert utf8_bytes(line) <= ceiling, line


def test_truncate_clips_on_bytes_without_splitting_a_character() -> None:
    text = "한국어 콘텐츠가 있습니다"

    clipped = truncate(text, 12)

    assert utf8_bytes(clipped) <= 12
    clipped.encode("utf-8").decode("utf-8")
    assert clipped.endswith("…")
