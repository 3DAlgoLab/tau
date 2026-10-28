"""Tests for closure-safe compaction cut selection and token ceilings."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from tau_agent.messages import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from tau_coding.compaction.cut import (
    SUMMARY_RAW_TOKEN_BUDGET,
    CallResultSpan,
    ContextCut,
    _span_crosses,
    call_result_spans,
    closure_safe,
    compute_cut,
)
from tau_coding.compaction.events import ContextRow
from tau_coding.context_window import estimate_message_tokens


def _pad(chars: int) -> str:
    return "x" * chars


def _user(entry_id: str, text: str) -> ContextRow:
    return (entry_id, UserMessage(content=text))


def _call_row(entry_id: str, *calls: ToolCall) -> ContextRow:
    return (entry_id, AssistantMessage(content=list(calls)))


def _text_row(entry_id: str, text: str) -> ContextRow:
    return (entry_id, AssistantMessage(content=[TextContent(text=text)]))


def _result_row(entry_id: str, call_id: str, *, text: str = "", tool: str = "read") -> ContextRow:
    return (
        entry_id,
        ToolResultMessage(
            tool_call_id=call_id,
            tool_name=tool,
            content=[TextContent(text=text)] if text else [],
        ),
    )


def _call(call_id: str, name: str = "read", **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))  # type: ignore[arg-type]


def _suffix(rows: Sequence[ContextRow]) -> list[int]:
    counts = [estimate_message_tokens(message) for _entry_id, message in rows]
    suffix = [0] * (len(counts) + 1)
    for index in range(len(counts) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + counts[index]
    return suffix


def _assert_provider_valid(cut: ContextCut, rows: Sequence[ContextRow]) -> None:
    """Every call/result pair lies wholly on one side, and the tail is valid."""
    boundary = len(cut.summarized_rows)
    spans = call_result_spans(rows)
    assert cut.first_kept_entry_id == rows[boundary][0]
    assert closure_safe(spans, boundary)
    assert not isinstance(rows[boundary][1], ToolResultMessage)
    assert cut.summarized_rows + cut.retained_rows == tuple(rows)


def test_spans_pair_calls_with_their_results() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1"), _call("c2")),
        _result_row("r2", "c1"),
        _result_row("r3", "c2", tool="edit"),
    ]

    spans = call_result_spans(rows)

    assert spans["c1"] == CallResultSpan(first=1, last=2, has_call=True, has_result=True)
    assert spans["c2"] == CallResultSpan(first=1, last=3, has_call=True, has_result=True)


def test_closure_safe_flags_a_boundary_straddling_a_pair() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1")),
        _user("r2", "steer"),
        _result_row("r3", "c1"),
    ]
    spans = call_result_spans(rows)

    assert _span_crosses(spans["c1"], 2) is True
    assert closure_safe(spans, 2) is False
    assert closure_safe(spans, 1) is True
    assert closure_safe(spans, 3) is False
    assert closure_safe(spans, 4) is True


def test_call_without_a_result_never_blocks_a_cut() -> None:
    rows = [_user("r0", "start"), _call_row("r1", _call("c1"))]
    spans = call_result_spans(rows)

    assert closure_safe(spans, 1) is True


def test_cut_refuses_a_boundary_that_straddles_a_call_and_result() -> None:
    rows = [
        _user("r0", _pad(200_000)),
        _call_row("r1", _call("c1", path="a.py")),
        _user("r2", "steer: also check the tests"),
        _result_row("r3", "c1", text=_pad(80_000)),
        _user("r4", "next step"),
        _text_row("r5", _pad(60_000)),
    ]
    suffix = _suffix(rows)
    raw_before = suffix[0]
    # A tail allowance just under the span's call row makes r2 the boundary the
    # budget wants: r3's result would be kept while r1's call was summarized.
    tail_budget = suffix[2] + 3

    straddling = compute_cut(
        rows,
        keep_recent_tokens=tail_budget,
        context_window_tokens=200_000,
        reserve_tokens=0,
        tokens_before=raw_before,
        calibrate=False,
    )
    generous = compute_cut(
        rows,
        keep_recent_tokens=suffix[1] + 3,
        context_window_tokens=200_000,
        reserve_tokens=0,
        tokens_before=raw_before,
        calibrate=False,
    )

    assert straddling is not None and straddling.budget is not None
    assert straddling.budget.binding_constraint == "continuity"
    assert straddling.budget.raw_tail_token_budget == tail_budget
    assert straddling.first_kept_entry_id == "r4"
    _assert_provider_valid(straddling, rows)

    assert generous is not None
    assert generous.first_kept_entry_id == "r1"
    _assert_provider_valid(generous, rows)


def test_cut_never_lands_on_a_tool_result_row() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1")),
        _result_row("r2", "c1", text=_pad(2_000)),
        _result_row("r3", "c2", text=_pad(2_000)),
        _user("r4", "next"),
    ]

    cut = compute_cut(rows, keep_recent_tokens=1_000)

    assert cut is not None
    _assert_provider_valid(cut, rows)


def test_tail_strategy_prefers_whole_turn_continuity() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1")),
        _result_row("r2", "c1", text="ok"),
        _text_row("r3", "in progress"),
        _user("r4", "new instruction"),
        _text_row("r5", "working"),
    ]

    cut = compute_cut(rows, keep_recent_tokens=10**9)

    assert cut is not None
    assert cut.first_kept_entry_id == "r4"
    _assert_provider_valid(cut, rows)


def test_tail_budget_limits_retention_and_keeps_the_open_turn() -> None:
    rows = [
        _user("r0", "start"),
        _text_row("r1", _pad(4_000)),
        _user("r2", "keep this turn"),
        _text_row("r3", _pad(2_000)),
    ]
    budget = estimate_message_tokens(rows[3][1]) + 10

    cut = compute_cut(rows, keep_recent_tokens=budget)

    assert cut is not None
    assert cut.retained_raw_tokens <= budget
    assert cut.first_kept_entry_id == "r2"
    _assert_provider_valid(cut, rows)


def test_oversized_single_turn_still_yields_a_valid_cut() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1", path="big.py")),
        _result_row("r2", "c1", text=_pad(40_000)),
        _user("r3", "next"),
    ]

    cut = compute_cut(rows, keep_recent_tokens=1)

    assert cut is not None
    assert cut.summarized_rows
    assert cut.retained_rows
    _assert_provider_valid(cut, rows)


def test_returns_none_when_no_valid_cut_exists() -> None:
    rows = [_call_row("r0", _call("c1")), _result_row("r1", "c1")]

    assert compute_cut(rows, keep_recent_tokens=1) is None


def test_returns_none_for_contexts_too_small_to_compact() -> None:
    assert compute_cut([_user("r0", "only")], keep_recent_tokens=10) is None
    assert compute_cut([], keep_recent_tokens=10) is None


def test_continuity_plan_reports_the_binding_constraint() -> None:
    rows = [
        _user("r0", "start"),
        _text_row("r1", _pad(40_000)),
        _user("r2", "next"),
        _text_row("r3", _pad(20_000)),
    ]
    tokens_before = 120_000

    cut = compute_cut(
        rows,
        context_window_tokens=131_072,
        tokens_before=tokens_before,
        calibrate=False,
    )

    assert cut is not None
    budget = cut.budget
    assert budget is not None
    assert budget.strategy == "continuity"
    assert budget.target_context_tokens == min(
        budget.continuity_target_tokens,
        budget.occupancy_ceiling_tokens,
        budget.safe_ceiling_tokens,
        budget.reduction_ceiling_tokens,
    )
    assert budget.binding_constraint == "reserve"
    assert budget.raw_tokens_before == sum(estimate_message_tokens(m) for _id, m in rows)
    # A provider-anchored reading of this small context leaves no room for a
    # tail at all, so selection falls back to the smallest valid tail.
    assert budget.raw_tail_token_budget == 0
    assert len(cut.retained_rows) == 1
    assert cut.retained_raw_tokens < budget.raw_tokens_before
    _assert_provider_valid(cut, rows)


def test_fallback_keeps_a_valid_tail_when_nothing_fits_the_budget() -> None:
    rows = [
        _user("r0", "start"),
        _call_row("r1", _call("c1", path="a.py")),
        _result_row("r2", "c1", text=_pad(60_000)),
        _user("r3", "next"),
        _text_row("r4", _pad(20_000)),
    ]

    cut = compute_cut(
        rows,
        keep_recent_tokens=1_000,
        context_window_tokens=4_000,
        reserve_tokens=0,
        calibrate=False,
    )

    assert cut is not None and cut.budget is not None
    assert cut.budget.raw_tail_token_budget == 0
    assert len(cut.retained_rows) == 1
    _assert_provider_valid(cut, rows)


def test_continuity_plan_scales_estimator_against_provider_usage() -> None:
    rows: list[ContextRow] = [
        _user("r0", "start"),
        (
            "r1",
            AssistantMessage(
                content=[TextContent(text=_pad(80_000))],
                usage=Usage(input=20_000, output=0, total_tokens=20_000),
            ),
        ),
        _text_row("r2", _pad(80_000)),
    ]
    raw_before = sum(estimate_message_tokens(message) for _id, message in rows)
    usage_raw_tokens = estimate_message_tokens(rows[1][1]) + estimate_message_tokens(rows[0][1])
    tokens_before = raw_before + 40_000

    cut = compute_cut(
        rows,
        context_window_tokens=200_000,
        reserve_tokens=0,
        keep_recent_tokens=2_000,
        tokens_before=tokens_before,
    )

    assert cut is not None
    budget = cut.budget
    assert budget is not None
    expected_scale = (tokens_before - 20_000) / (raw_before - usage_raw_tokens)
    assert budget.token_scale == pytest.approx(expected_scale)
    assert budget.token_scale > 2.0
    assert budget.fixed_overhead_tokens == 0
    assert budget.binding_constraint == "continuity"
    assert budget.raw_tail_token_budget == max(
        0,
        int((budget.target_context_tokens - budget.fixed_overhead_tokens) // budget.token_scale)
        - SUMMARY_RAW_TOKEN_BUDGET,
    )
    _assert_provider_valid(cut, rows)


def test_uncalibrated_plan_treats_unexplained_tokens_as_fixed_overhead() -> None:
    rows = [
        _user("r0", "start"),
        _text_row("r1", _pad(20_000)),
    ]
    raw_before = sum(estimate_message_tokens(message) for _id, message in rows)
    tokens_before = raw_before + 12_000

    cut = compute_cut(
        rows,
        context_window_tokens=200_000,
        tokens_before=tokens_before,
        calibrate=False,
    )

    assert cut is not None
    budget = cut.budget
    assert budget is not None
    assert budget.token_scale == 1.0
    assert budget.fixed_overhead_tokens == 12_000


def test_cut_budget_serializes_to_json() -> None:
    rows = [
        _user("r0", "start"),
        _text_row("r1", _pad(20_000)),
        _user("r2", "next"),
    ]

    cut = compute_cut(rows, context_window_tokens=64_000, tokens_before=40_000)

    assert cut is not None and cut.budget is not None
    payload = json.loads(json.dumps(cut.budget.to_json()))
    assert payload["contextWindow"] == 64_000
    assert payload["strategy"] == "continuity"
    assert set(payload) >= {
        "bindingConstraint",
        "retainedRawTokens",
        "rawTailTokenBudget",
        "tokenScale",
    }


def test_occupancy_ceiling_shrinks_the_retained_tail() -> None:
    rows = [
        _user("r0", "start"),
        _text_row("r1", _pad(40_000)),
        _text_row("r2", _pad(40_000)),
        _user("r3", "middle"),
        _text_row("r4", _pad(40_000)),
        _text_row("r5", _pad(40_000)),
        _user("r6", "latest"),
    ]

    roomy = compute_cut(
        rows,
        keep_recent_tokens=50_000,
        context_window_tokens=300_000,
        reserve_tokens=0,
        calibrate=False,
    )
    tight = compute_cut(
        rows,
        keep_recent_tokens=50_000,
        context_window_tokens=20_000,
        reserve_tokens=0,
        target_occupancy=0.5,
        calibrate=False,
    )

    assert roomy is not None and tight is not None
    assert len(tight.summarized_rows) > len(roomy.summarized_rows)
    _assert_provider_valid(tight, rows)
    assert roomy.budget is not None and tight.budget is not None
    assert roomy.budget.binding_constraint == "reduction"
    assert tight.budget.binding_constraint == "occupancy"
    assert tight.budget.occupancy_ceiling_tokens == 10_000
