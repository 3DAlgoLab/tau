"""Tests for typed, prose-blind normalization of active-context rows."""

from __future__ import annotations

from tau_agent.messages import (
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_agent.session.entries import BranchSummaryEntry
from tau_agent.session.memory import _format_branch_summary, _format_compaction_summary
from tau_coding.compaction.events import (
    AssistantTextEvent,
    BashEvent,
    ContextRow,
    CustomMessageEvent,
    PriorSummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
    count_erased_thinking_blocks,
    first_line,
    normalize_rows,
)
from tau_coding.context_window import COMPACTION_SUMMARY_PREFIX


def _assistant_message(text: str = "", *, calls: tuple[ToolCall, ...] = ()) -> AssistantMessage:
    blocks: list[object] = []
    if text:
        blocks.append(TextContent(text=text))
    blocks.extend(calls)
    return AssistantMessage(content=blocks)  # type: ignore[arg-type]


def _call(call_id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))  # type: ignore[arg-type]


def test_events_keep_sequential_indices_and_entry_addresses() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="first request")),
        (
            "e2",
            _assistant_message(
                "working on it",
                calls=(_call("c1", "read", path="a.py"), _call("c2", "read", path="b.py")),
            ),
        ),
    ]

    events = normalize_rows(rows)

    assert [event.index for event in events] == [1, 2, 3, 4]
    assert [event.entry_id for event in events] == ["e1", "e2", "e2", "e2"]
    assert [event.source_entry_id for event in events] == ["e1", "e2", "e2", "e2"]
    assert [event.kind for event in events] == ["user", "assistant_text", "tool_call", "tool_call"]
    assert isinstance(events[0], UserEvent)
    assert events[0].text == "first request"
    assert isinstance(events[2], ToolCallEvent)
    assert events[2].arguments == {"path": "a.py"}


def test_thinking_blocks_are_erased_and_counted() -> None:
    rows: list[ContextRow] = [
        (
            "e1",
            AssistantMessage(
                content=[
                    ThinkingContent(thinking="maybe delete everything"),
                    TextContent(text="conclusion only"),
                    ThinkingContent(thinking="second draft"),
                ]
            ),
        ),
    ]

    events = normalize_rows(rows)

    assert [event.kind for event in events] == ["assistant_text"]
    assert isinstance(events[0], AssistantTextEvent)
    assert events[0].text == "conclusion only"
    assert count_erased_thinking_blocks(rows) == 2
    assert all("maybe delete everything" not in getattr(event, "text", "") for event in events)


def test_bash_result_inherits_its_command_from_the_call() -> None:
    rows: list[ContextRow] = [
        (
            "e1",
            _assistant_message(calls=(_call("c1", "bash", command="uv run pytest", cwd="x"),)),
        ),
        (
            "e2",
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="bash",
                content=[TextContent(text="1 failed")],
                is_error=True,
            ),
        ),
    ]

    events = normalize_rows(rows)
    bash = events[-1]

    assert isinstance(bash, BashEvent)
    assert bash.command == "uv run pytest"
    assert bash.is_error is True
    assert bash.error == "1 failed"
    assert bash.tool_call_id == "c1"


def test_non_bash_result_keeps_its_tool_name_and_text() -> None:
    rows: list[ContextRow] = [
        ("e1", _assistant_message(calls=(_call("c1", "read", path="a.py"),))),
        (
            "e2",
            ToolResultMessage(
                tool_call_id="c1",
                tool_name="read",
                content=[TextContent(text="file body")],
            ),
        ),
    ]

    events = normalize_rows(rows)
    result = events[-1]

    assert isinstance(result, ToolResultEvent)
    assert result.tool_name == "read"
    assert result.text == "file body"
    assert result.is_error is False


def test_local_bash_execution_records_exit_code_and_cancellation() -> None:
    rows: list[ContextRow] = [
        ("e1", BashExecutionMessage(command="make", output="boom", exit_code=2)),
        ("e2", BashExecutionMessage(command="sleep 5", output="", cancelled=True)),
    ]

    events = normalize_rows(rows)

    first, second = events
    assert isinstance(first, BashEvent)
    assert first.is_error is True
    assert first.exit_code == 2
    assert first.error == "boom"
    assert isinstance(second, BashEvent)
    assert second.is_error is True
    assert second.exit_code is None


def test_custom_message_keeps_type_visibility_and_sanitized_details() -> None:
    rows: list[ContextRow] = [
        (
            "e1",
            CustomMessage(
                custom_type="my_ext/status",
                content="state note",
                display=False,
                details={"nested": {"flag": True, "count": 3}, "blob": "y" * 4_000},
            ),
        ),
    ]

    events = normalize_rows(rows)
    event = events[0]

    assert isinstance(event, CustomMessageEvent)
    assert event.custom_type == "my_ext/status"
    assert event.display is False
    assert event.details is not None
    assert event.details["nested"] == {"count": 3, "flag": True}
    assert len(event.details["blob"] or "") <= 1_024


def test_oversized_custom_details_are_dropped_entirely() -> None:
    rows: list[ContextRow] = [
        (
            "e1",
            CustomMessage(
                custom_type="my_ext/blob",
                content="note",
                details={f"key{index}": index for index in range(400)},
            ),
        ),
    ]

    events = normalize_rows(rows)
    event = events[0]

    assert isinstance(event, CustomMessageEvent)
    assert event.details is None


def test_compaction_summary_row_becomes_a_prior_summary_event() -> None:
    formatted = _format_compaction_summary("## Goal\nkeep this")
    assert formatted.startswith(COMPACTION_SUMMARY_PREFIX)

    rows: list[ContextRow] = [("cmp1", UserMessage(content=formatted))]
    events = normalize_rows(rows)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PriorSummaryEvent)
    assert event.source == "compaction"
    assert event.summary == "## Goal\nkeep this"


def test_branch_summary_prefix_is_pinned_to_the_replay_formatter() -> None:
    entry = BranchSummaryEntry(summary="branch body")
    formatted = _format_branch_summary(entry)

    rows: list[ContextRow] = [("br1", UserMessage(content=formatted))]
    events = normalize_rows(rows)

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, PriorSummaryEvent)
    assert event.source == "branch"
    assert "branch body" in event.summary


def test_summary_message_roles_normalize_as_prior_summaries() -> None:
    rows: list[ContextRow] = [
        ("c1", CompactionSummaryMessage(summary="compact body", tokens_before=100)),
        ("b1", BranchSummaryMessage(summary="branch body", from_id="root")),
    ]

    events = normalize_rows(rows)

    assert [event.kind for event in events] == ["prior_summary", "prior_summary"]
    assert [event.source for event in events] == ["compaction", "branch"]


def test_first_line_stops_at_the_first_newline() -> None:
    assert first_line("  \n  second\nthird") == "second"
    assert first_line("single") == "single"
    assert first_line("") == ""
