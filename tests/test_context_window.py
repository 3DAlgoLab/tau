from pathlib import Path

from tau_agent import AssistantMessage, TextContent, ToolCall, ToolResultMessage, Usage, UserMessage
from tau_agent.messages import assistant_content
from tau_coding.context_window import (
    COMPACTION_SUMMARY_PREFIX,
    ContextUsageEstimate,
    auto_compaction_threshold_for_context_window,
    build_compaction_summary_prompt,
    estimate_context_tokens,
    estimate_context_usage,
    estimate_message_tokens,
    estimate_text_tokens,
    provider_prompt_floor_tokens,
    serialize_messages_for_compaction,
    summarize_messages_for_compaction,
)
from tau_coding.tools import create_coding_tools


def test_text_token_estimate_is_deterministic() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("a") == 1
    assert estimate_text_tokens("abcd") == 1
    assert estimate_text_tokens("abcde") == 2


def test_message_token_estimate_counts_roles_and_tool_calls() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    user_tokens = estimate_message_tokens(UserMessage(content="hello"))
    assistant_tokens = estimate_message_tokens(
        AssistantMessage(content=assistant_content("using tool", [tool_call]))
    )
    tool_tokens = estimate_message_tokens(
        ToolResultMessage(
            tool_call_id="call-1", tool_name="read", content=[TextContent(text="contents")]
        )
    )

    assert user_tokens > estimate_text_tokens("hello")
    assert assistant_tokens > user_tokens
    assert tool_tokens > estimate_text_tokens("contents")


def test_context_token_estimate_includes_system_messages_and_tools(tmp_path: Path) -> None:
    tools = tuple(create_coding_tools(cwd=tmp_path))

    estimate = estimate_context_tokens(
        system="You are Tau.",
        messages=(UserMessage(content="hello"), AssistantMessage(content="hi")),
        tools=tools,
    )

    assert estimate > estimate_text_tokens("You are Tau.hellohi")


def test_context_usage_uses_latest_provider_report_and_estimates_only_trailing_messages() -> None:
    reported = AssistantMessage(
        content="answer",
        usage=Usage(input=70_000, output=2_000, cache_read=28_000, total_tokens=100_000),
        timestamp=200,
    )
    trailing = UserMessage(content="follow up", timestamp=300)

    usage = estimate_context_usage(
        system="ignored because provider usage includes it",
        messages=(UserMessage(content="large history", timestamp=100), reported, trailing),
        tools=(),
    )

    assert usage.uses_provider_usage is True
    assert usage.provider_tokens == 100_000
    assert usage.trailing_tokens == estimate_message_tokens(trailing)
    assert usage.total_tokens == 100_000 + usage.trailing_tokens


def test_context_usage_ignores_stale_provider_report_after_newer_prefix_is_inserted() -> None:
    stale = AssistantMessage(
        content="old answer",
        usage=Usage(total_tokens=100_000),
        timestamp=100,
    )
    summary = UserMessage(content="Previous conversation summary", timestamp=200)

    usage = estimate_context_usage(system="system", messages=(summary, stale), tools=())

    assert usage.uses_provider_usage is False
    assert usage.total_tokens < 100_000


def test_context_usage_ignores_error_response_usage() -> None:
    failed = AssistantMessage(
        stop_reason="error",
        error_message="failed",
        usage=Usage(total_tokens=100_000),
    )

    usage = estimate_context_usage(system="system", messages=(failed,), tools=())

    assert usage.uses_provider_usage is False
    assert usage.total_tokens < 100_000


def test_auto_compaction_threshold_keeps_pi_style_reserve() -> None:
    assert auto_compaction_threshold_for_context_window(128_000) == 111_616
    assert auto_compaction_threshold_for_context_window(16_384) == 1
    assert auto_compaction_threshold_for_context_window(0) is None


def test_provider_prompt_floor_is_the_accepted_prompt_without_the_reply() -> None:
    older = AssistantMessage(
        content="earlier",
        usage=Usage(input=1_000, cache_read=900, output=50, total_tokens=1_950),
        timestamp=100,
    )
    newest = AssistantMessage(
        content="later",
        usage=Usage(input=40_000, cache_read=70_000, cache_write=1_000, output=500),
        timestamp=200,
    )
    trailing = UserMessage(content="more", timestamp=250)

    floor = provider_prompt_floor_tokens((older, trailing, newest))

    # The prompt the provider accepted, excluding what it wrote back: the next
    # request carries that prefix plus whatever came after.
    assert floor == 111_000


def test_provider_prompt_floor_ignores_rejected_and_unmeasured_responses() -> None:
    measured = AssistantMessage(
        content="answer",
        usage=Usage(input=120_000, cache_read=1_000, total_tokens=121_500),
        timestamp=100,
    )
    rejected = AssistantMessage(
        content="",
        stop_reason="error",
        error_message="prompt exceeds the context window",
        usage=Usage(input=131_000, total_tokens=131_000),
        timestamp=200,
    )
    unmeasured = AssistantMessage(content="no usage", timestamp=300)

    assert provider_prompt_floor_tokens((measured, rejected, unmeasured)) == 121_000


def test_provider_prompt_floor_stops_at_a_compaction_summary() -> None:
    before = AssistantMessage(
        content="answer",
        usage=Usage(input=100_000, cache_read=20_000, total_tokens=120_500),
        timestamp=100,
    )
    summary = UserMessage(
        content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\nKeep going.",
        timestamp=200,
    )

    # Usage measured before a compaction priced context the summary replaced, so it
    # can only overstate what the next request will carry.
    assert provider_prompt_floor_tokens((before, summary)) == 0


def test_provider_prompt_floor_ignores_a_replayed_summary_row_at_the_front() -> None:
    """Replay puts a compaction summary *in front of* the responses it replaced.

    Breaking the scan at the summary row made the responses behind it count again, so
    the floor kept pricing a prefix the compaction had thrown away and could never
    come back down. In the session this was found in, one compaction took the prompt
    from 115,209 tokens to 51,476 and every pressure check afterwards still reported
    115,209: each later trigger was phantom, and one of them paid 56,240 tokens to
    summarize a context that was not near its window.
    """
    kept = AssistantMessage(
        content="old answer",
        usage=Usage(input=100_000, cache_read=20_000, total_tokens=120_500),
        timestamp=100,
    )
    summary = UserMessage(
        content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\nKeep going.",
        timestamp=200,
    )
    after = AssistantMessage(
        content="post-compaction answer",
        usage=Usage(input=50_000, cache_read=1_000, total_tokens=51_976),
        timestamp=300,
    )

    # Only what the provider measured since the boundary counts as a floor.
    assert provider_prompt_floor_tokens((summary, kept, after)) == 51_000
    # Nothing measured since the boundary: no floor, rather than a stale one.
    assert provider_prompt_floor_tokens((summary, kept)) == 0


def test_provider_prompt_floor_uses_the_newest_of_several_summaries() -> None:
    early = UserMessage(content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\nEarly.", timestamp=100)
    between = AssistantMessage(
        content="measured between compactions",
        usage=Usage(input=90_000, cache_read=0, total_tokens=90_500),
        timestamp=200,
    )
    later = UserMessage(content=f"{COMPACTION_SUMMARY_PREFIX}## Goal\nLater.", timestamp=400)

    assert provider_prompt_floor_tokens((early, between, later)) == 0


def test_context_usage_estimate_reports_breakdown(tmp_path: Path) -> None:
    tools = tuple(create_coding_tools(cwd=tmp_path))
    messages = (UserMessage(content="hello"), AssistantMessage(content="hi"))

    usage = estimate_context_usage(system="You are Tau.", messages=messages, tools=tools)

    assert isinstance(usage, ContextUsageEstimate)
    assert usage.message_count == 2
    assert usage.tool_count == len(tools)
    assert usage.system_tokens == estimate_text_tokens("You are Tau.")
    assert usage.message_tokens == sum(estimate_message_tokens(message) for message in messages)
    assert usage.total_tokens == usage.system_tokens + usage.message_tokens + usage.tool_tokens
    assert estimate_context_tokens(system="You are Tau.", messages=messages, tools=tools) == (
        usage.total_tokens
    )


def test_summarize_messages_for_compaction_is_deterministic() -> None:
    tool_call = ToolCall(id="call-1", name="read", arguments={"path": "README.md"})

    summary = summarize_messages_for_compaction(
        (
            UserMessage(content="Read README.md"),
            AssistantMessage(content=assistant_content("I'll inspect it.", [tool_call])),
            ToolResultMessage(
                tool_call_id="call-1",
                tool_name="read",
                content=[TextContent(text="README contents")],
            ),
        )
    )

    assert summary == "\n".join(
        [
            "Automatically compacted 3 prior message(s).",
            "1. user: Read README.md",
            "2. assistant: I'll inspect it. [tool calls: read]",
            "3. tool: read ok: README contents",
        ]
    )


def test_compaction_summary_prompt_uses_pi_format_and_custom_instructions() -> None:
    prompt = build_compaction_summary_prompt(
        (
            UserMessage(content="Refactor src/app.py"),
            AssistantMessage(content="Updated src/app.py"),
        ),
        custom_instructions="Focus on files changed.",
    )

    assert "<conversation>" in prompt
    assert "Use this EXACT format:" in prompt
    assert "## Goal" in prompt
    assert "Preserve exact file paths" in prompt
    assert "Additional focus: Focus on files changed." in prompt
    assert "Refactor src/app.py" in prompt


def test_compaction_summary_prompt_updates_previous_summary() -> None:
    prompt = build_compaction_summary_prompt(
        (
            UserMessage(content="Previous conversation summary:\n## Goal\nShip compaction."),
            UserMessage(content="Now add tests."),
        )
    )

    assert "<previous-summary>\n## Goal\nShip compaction.\n</previous-summary>" in prompt
    assert "NEW conversation messages" in prompt
    assert "Now add tests." in prompt
    assert "Previous conversation summary" not in serialize_messages_for_compaction(
        (UserMessage(content="Now add tests."),)
    )
