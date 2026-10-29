"""Sub-call bridge tests: what ``llm_query`` asks, and what it costs.

The bridge is the only path from model-written code to a provider, so these tests
pin the properties that make it safe to hand a model a completion function: it
asks the session's current model, it never hands out tools, it bounds what a cell
may spend, and it keeps a ledger that session usage analytics cannot see.
"""

from __future__ import annotations

import pytest

from pi_event_helpers import assistant_done, assistant_error, assistant_start, text_delta
from tau_agent.harness import SimpleCancellationToken
from tau_agent.messages import AssistantMessage, Usage, message_text
from tau_ai import FakeProvider
from tau_coding.kernel.subcall import SubcallBridge, SubcallTotals

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _done(text: str, *, usage: Usage | None = None) -> list[object]:
    return [
        assistant_start(),
        text_delta(text),
        assistant_done(AssistantMessage(model="m", content=text, usage=usage or Usage())),
    ]


def _bridge(
    streams: list[list[object]], model: str = "alpha"
) -> tuple[SubcallBridge, FakeProvider]:
    provider = FakeProvider(streams)  # type: ignore[arg-type]
    bridge = SubcallBridge(provider=lambda: provider, model=lambda: model)
    return bridge, provider


async def test_a_sub_call_answers_with_the_session_model_and_its_usage() -> None:
    bridge, provider = _bridge([_done("42", usage=Usage(input=5, output=7))])
    bridge.begin_cell(None)
    outcome = await bridge("what is it?")
    assert outcome.text == "42"
    assert outcome.model == "alpha"
    assert outcome.metadata.get("empty") is False
    model, system, messages, tools = provider.calls[0]
    assert model == "alpha"
    assert system
    assert len(messages) == 1
    assert message_text(messages[0]) == "what is it?"
    assert tools == []
    totals = bridge.totals()
    assert totals.calls == 1
    assert totals.requested_tokens == 5
    assert totals.output_tokens == 7


async def test_the_model_is_resolved_per_call_so_a_switch_takes_effect() -> None:
    provider = FakeProvider([_done("one"), _done("two")])  # type: ignore[arg-type]
    current = ["alpha"]
    bridge = SubcallBridge(provider=lambda: provider, model=lambda: current[0])
    bridge.begin_cell(None)
    assert (await bridge("first")).model == "alpha"
    current[0] = "beta"
    second = await bridge("second")
    assert second.model == "beta"
    assert [call[0] for call in provider.calls] == ["alpha", "beta"]
    assert bridge.totals().models == ("alpha", "beta")


async def test_an_explicit_model_argument_overrides_the_session_model() -> None:
    bridge, provider = _bridge([_done("sure")])
    bridge.begin_cell(None)
    outcome = await bridge("q", model="gamma")
    assert outcome.model == "gamma"
    assert provider.calls[0][0] == "gamma"


async def test_a_provider_error_raises_and_is_counted_as_a_failed_call() -> None:
    bridge, _ = _bridge([[assistant_error("upstream exploded")]])
    bridge.begin_cell(None)
    with pytest.raises(RuntimeError, match="upstream exploded"):
        await bridge("q")
    totals = bridge.totals()
    assert (totals.calls, totals.failures) == (1, 1)


async def test_a_cancelled_cell_refuses_to_keep_paying() -> None:
    bridge, _ = _bridge([_done("late")])
    token = SimpleCancellationToken()
    token.cancel()
    bridge.begin_cell(token)
    with pytest.raises(RuntimeError, match="cancelled"):
        await bridge("q")
    assert bridge.totals().failures == 1


async def test_the_response_is_byte_bounded_and_says_what_it_cut() -> None:
    bridge, _ = _bridge([_done("x" * 5_000)])
    bridge.max_response_bytes = 256
    bridge.begin_cell(None)
    outcome = await bridge("q")
    assert len(outcome.text.encode("utf-8")) <= 256
    assert outcome.metadata["omitted_bytes"] == 5_000 - 256


async def test_an_empty_answer_is_flagged_instead_of_looked_through() -> None:
    bridge, _ = _bridge([_done("   ")])
    bridge.begin_cell(None)
    outcome = await bridge("q")
    assert outcome.metadata["empty"] is True


async def test_a_cell_cannot_exceed_its_sub_call_budget() -> None:
    bridge, _ = _bridge([_done("a"), _done("b"), _done("c")])
    bridge.max_calls_per_cell = 2
    bridge.begin_cell(None)
    await bridge("one")
    await bridge("two")
    with pytest.raises(RuntimeError, match="budget of 2"):
        await bridge("three")


async def test_failed_calls_count_against_the_budget_too() -> None:
    bridge, _ = _bridge([[assistant_error("nope")], [_done("ok")], [_done("again")]])
    bridge.max_calls_per_cell = 2
    bridge.begin_cell(None)
    with pytest.raises(RuntimeError, match="nope"):
        await bridge("one")
    await bridge("two")
    with pytest.raises(RuntimeError, match="budget"):
        await bridge("three")


async def test_the_ledger_resets_at_the_cell_boundary() -> None:
    bridge, _ = _bridge(
        [_done("a", usage=Usage(input=3, output=2)), _done("b", usage=Usage(input=4, output=1))]
    )
    bridge.begin_cell(None)
    await bridge("one")
    first = bridge.take_totals()
    assert first.calls == 1 and first.requested_tokens == 3
    assert bridge.totals() == SubcallTotals()
    bridge.end_cell()
    bridge.begin_cell(None)
    await bridge("two")
    second = bridge.take_totals()
    assert second.calls == 1 and second.output_tokens == 1


async def test_a_sub_call_outside_a_cell_still_answers() -> None:
    bridge, _ = _bridge([_done("no scope")])
    outcome = await bridge("q")
    assert outcome.text == "no scope"
