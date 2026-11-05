"""A compacted session has to mean the same thing replayed as it did live.

`SessionState.from_entries` rebuilds the provider-visible message list from stored
entries. Two kinds of row in that list have no message entry behind them — a
compaction summary and a branch summary — so their timestamps came from the wall
clock of whoever was replaying. Context accounting reads message order to decide
whether a provider's usage block still describes the active prefix, so a single row
stamped in the future invalidates every usage number in the session at once.

That is how a session ended up with two different sizes: 131k tokens while it ran,
97k for the same bytes on disk. The automatic compaction trigger compares that
number to a threshold, so the file that was already at its context window looked
comfortably under it for hundreds of turns and was never compacted.
"""

from tau_agent import AssistantMessage, Usage, UserMessage
from tau_agent.session import BranchSummaryEntry, CompactionEntry, MessageEntry, SessionState
from tau_coding.context_window import COMPACTION_SUMMARY_PREFIX, estimate_context_usage


def _branch() -> list[object]:
    """A short session with one compaction in the middle and work after it."""
    return [
        MessageEntry(
            id="u1",
            timestamp=1000,
            message=UserMessage(content="Why is the sky blue?", timestamp=1_000_000),
        ),
        MessageEntry(
            id="a1",
            parent_id="u1",
            timestamp=1100,
            message=AssistantMessage(
                content="Rayleigh scattering.",
                usage=Usage(input=50_000, cache_read=30_000, total_tokens=80_500),
                timestamp=1_100_000,
            ),
        ),
        MessageEntry(
            id="u2",
            parent_id="a1",
            timestamp=1200,
            message=UserMessage(content="And sunsets?", timestamp=1_200_000),
        ),
        MessageEntry(
            id="a2",
            parent_id="u2",
            timestamp=1300,
            message=AssistantMessage(
                content="Longer paths.",
                usage=Usage(input=60_000, cache_read=30_000, total_tokens=90_600),
                timestamp=1_300_000,
            ),
        ),
        CompactionEntry(
            id="c1",
            parent_id="a2",
            timestamp=1400,
            summary="## Goal\nExplain scattering.",
            first_kept_entry_id="u1",
            tokens_before=90_600,
        ),
        MessageEntry(
            id="a3",
            parent_id="c1",
            timestamp=1500,
            message=AssistantMessage(
                content="Clouds help.",
                usage=Usage(input=95_000, cache_read=10_000, total_tokens=105_700),
                timestamp=1_500_000,
            ),
        ),
    ]


def test_a_rebuilt_compaction_summary_carries_its_entry_time() -> None:
    state = SessionState.from_entries(_branch())  # type: ignore[arg-type]

    summary = state.messages[0]
    assert isinstance(summary, UserMessage)
    assert summary.content.startswith(COMPACTION_SUMMARY_PREFIX)
    # The compaction entry's own clock (seconds), not the moment of replay: nothing
    # else in the file can be dated after the work that followed it.
    assert summary.timestamp == 1_400_000


def test_a_rebuilt_branch_summary_carries_its_entry_time() -> None:
    entries = [
        *(_branch()[:1]),
        BranchSummaryEntry(
            id="b1",
            parent_id="u1",
            timestamp=1234,
            summary="A branch that was abandoned.",
        ),
    ]
    state = SessionState.from_entries(entries)  # type: ignore[arg-type]

    branch_row = state.messages[-1]
    assert isinstance(branch_row, UserMessage)
    assert "branch" in branch_row.content
    assert branch_row.timestamp == 1_234_000


def test_a_replayed_compaction_keeps_the_provider_usage_anchor() -> None:
    state = SessionState.from_entries(_branch())  # type: ignore[arg-type]

    usage = estimate_context_usage(system="You are Tau.", messages=state.messages, tools=())

    # The last response after the compaction still prices the active prefix. Before
    # the summary row was dated from the replay, it sat after every message in the
    # list, and this fell back to counting characters — here, a quarter of the real
    # size.
    assert usage.uses_provider_usage is True
    assert usage.provider_tokens == 105_700


def test_two_replays_of_one_session_report_one_context_size() -> None:
    first = estimate_context_usage(
        system="You are Tau.",
        messages=SessionState.from_entries(_branch()).messages,  # type: ignore[arg-type]
        tools=(),
    )
    second = estimate_context_usage(
        system="You are Tau.",
        messages=SessionState.from_entries(_branch()).messages,  # type: ignore[arg-type]
        tools=(),
    )

    assert first.total_tokens == second.total_tokens
    assert first.uses_provider_usage is second.uses_provider_usage
