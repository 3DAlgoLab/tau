"""Tests for the protected dialogue fold of the deterministic compactor."""

from __future__ import annotations

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    UserMessage,
)
from tau_coding.compaction.bounds import utf8_bytes
from tau_coding.compaction.dialogue import (
    MAX_EXCHANGE_BYTES,
    MAX_RECENT_EXCHANGES,
    dialogue_excerpts,
    project_dialogue,
    recent_dialogue,
)
from tau_coding.compaction.events import CompactionEvent, ContextRow, normalize_rows


def _user(text: str) -> UserMessage:
    return UserMessage(content=text)


def _assistant(*blocks: str) -> AssistantMessage:
    content: list[object] = [TextContent(text=block) for block in blocks]
    return AssistantMessage(content=content)  # type: ignore[arg-type]


def _rows(*pairs: tuple[str, AgentMessage]) -> list[ContextRow]:
    return [(entry_id, message) for entry_id, message in pairs]


def _events(*pairs: tuple[str, AgentMessage]) -> list[CompactionEvent]:
    return list(normalize_rows(_rows(*pairs)))


def test_dialogue_pairs_a_request_with_the_response_that_followed_it() -> None:
    events = _events(
        ("e1", _user("first request")),
        ("e2", _assistant("first reply")),
        ("e3", _user("second request")),
        ("e4", _assistant("second reply")),
    )

    exchanges = recent_dialogue(events)

    assert [
        (item.user.entry_id, item.assistant.entry_id if item.assistant else None)
        for item in exchanges
    ] == [
        ("e1", "e2"),
        ("e3", "e4"),
    ]


def test_one_assistant_entry_is_one_response_even_across_text_blocks() -> None:
    events = _events(
        ("e1", _user("explain the plan")),
        ("e2", _assistant("part one", "part two")),
    )

    exchanges = recent_dialogue(events)

    assert exchanges[0].assistant is not None
    assert exchanges[0].assistant.entry_id == "e2"
    assert exchanges[0].assistant.text == "part one\npart two"


def test_only_the_most_recent_exchanges_survive() -> None:
    pairs: list[tuple[str, AgentMessage]] = []
    for index in range(MAX_RECENT_EXCHANGES + 3):
        pairs.append((f"u{index}", _user(f"request {index}")))
        pairs.append((f"a{index}", _assistant(f"reply {index}")))

    exchanges = recent_dialogue(_events(*pairs))

    assert len(exchanges) == MAX_RECENT_EXCHANGES
    assert [item.user.entry_id for item in exchanges] == [
        f"u{index}" for index in range(3, MAX_RECENT_EXCHANGES + 3)
    ]


def test_thinking_never_reaches_the_dialogue_fold() -> None:
    message = AssistantMessage(
        content=[ThinkingContent(thinking="private deliberation"), TextContent(text="public note")]
    )

    rows: list[ContextRow] = [("e1", _user("ask")), ("e2", message)]
    exchanges = recent_dialogue(normalize_rows(rows))

    assert exchanges[0].assistant is not None
    assert exchanges[0].assistant.text == "public note"


def test_assistant_prose_without_a_visible_request_is_not_attributed() -> None:
    exchanges = recent_dialogue(_events(("e1", _assistant("orphan note"))))

    assert exchanges == []


def test_retained_request_with_retained_reply_is_dropped() -> None:
    tail = _events(
        ("e9", _user("kept request")),
        ("e10", _assistant("kept reply")),
    )

    exchanges = recent_dialogue([], tail)

    assert exchanges == []


def test_summarized_request_keeps_a_retained_reply_as_an_address_only() -> None:
    summarized = _events(("e1", _user("request that is summarized")))
    tail = _events(("e2", _assistant("reply that stays in context")))

    exchanges = recent_dialogue(summarized, tail)
    excerpts = dialogue_excerpts(exchanges[0])

    assert exchanges[0].assistant is not None
    assert exchanges[0].assistant.retained is True
    assert excerpts[0].text == "request that is summarized"
    assert "retained raw" in excerpts[1].heading
    assert excerpts[1].text == ""


def test_long_reply_is_clipped_without_losing_the_request() -> None:
    long_reply = "reply " + ("x" * (MAX_EXCHANGE_BYTES * 2))
    exchanges = recent_dialogue(
        _events(("e1", _user("short request")), ("e2", _assistant(long_reply)))
    )

    projection = project_dialogue(exchanges)
    rendered = "\n\n".join(projection.lines)

    assert "short request" in rendered
    assert projection.omitted_bytes > 0
    assert utf8_bytes(rendered) <= MAX_EXCHANGE_BYTES


def test_long_request_keeps_at_least_half_the_exchange() -> None:
    long_request = "q" * (MAX_EXCHANGE_BYTES * 3)
    exchanges = recent_dialogue(
        _events(("e1", _user(long_request)), ("e2", _assistant("brief reply")))
    )

    excerpts = dialogue_excerpts(exchanges[0])
    user_bytes = utf8_bytes(excerpts[0].text)

    assert user_bytes >= MAX_EXCHANGE_BYTES // 3
    assert excerpts[1].text == ""
    assert excerpts[1].omitted_bytes == utf8_bytes("brief reply")


def test_project_dialogue_labels_every_side_and_counts_loss() -> None:
    events = _events(
        ("e1", _user("do the thing")),
        ("e2", _assistant("did the thing")),
    )

    projection = project_dialogue(recent_dialogue(events))

    assert projection.lines[0].startswith("User instruction [entry e1]:")
    assert (
        "Historical assistant response (not a verified outcome) [entry e2]:" in projection.lines[0]
    )
    assert projection.omitted_bytes == 0


def test_tool_calls_do_not_count_as_exchanges() -> None:
    message = AssistantMessage(
        content=[ToolCall(id="c1", name="bash", arguments={"command": "ls"})]
    )  # type: ignore[arg-type]

    exchanges = recent_dialogue(_events(("e1", _user("list files")), ("e2", message)))

    assert exchanges[0].assistant is None
