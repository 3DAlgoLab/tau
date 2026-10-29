"""Recursion tests: what a child is allowed to do, and what it costs.

A real child kernel runs the model's cells here, because the guarantee under test
is that turns, calls, and depth stay bounded around something that genuinely
executes code. The model is scripted: the behaviour worth pinning is the
recursion's, not the provider's.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import Any

import pytest

from pi_event_helpers import assistant_done, assistant_start, text_delta
from tau_agent.messages import AssistantMessage
from tau_coding.kernel.process import ProcessKernel
from tau_coding.kernel.subcall import SubcallBridge
from tau_coding.rlm import CHILD_SYSTEM, RlmBudgetError, RlmLimits, RlmQuery

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _ScriptedModel:
    """Answers with canned cells in order and remembers what it was asked."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.prompts: list[str] = []
        self.systems: list[str | None] = []

    def stream_response(
        self, *, model: str, system: str, messages: list[Any], tools: list[Any], signal: Any
    ) -> AsyncIterator[Any]:
        self.prompts.append(str(messages[0].content))
        self.systems.append(system)
        if not self.script:
            raise AssertionError("the recursion asked for a turn nobody scripted")
        return self._stream(model, self.script.pop(0))

    async def _stream(self, model: str, text: str) -> AsyncIterator[Any]:
        yield assistant_start(model)
        yield text_delta(text)
        yield assistant_done(AssistantMessage(model=model, content=text))


def _query(tmp_path, script: list[str], *, limits: RlmLimits | None = None, depth: int = 0):
    """Wire a recursion over a real child kernel and a scripted model."""
    provider = _ScriptedModel(script)
    bridge = SubcallBridge(provider=lambda: provider, model=lambda: "fake")
    made: list[ProcessKernel] = []

    def factory(rlm):
        kernel = ProcessKernel(cwd=tmp_path, subcall=bridge, rlm=rlm, max_concurrent_subcalls=1)
        made.append(kernel)
        return kernel

    query = RlmQuery(bridge=bridge, kernel_factory=factory, limits=limits, depth=depth)
    query.begin_cell()
    return query, provider, bridge, made


async def test_a_child_solves_its_task_by_writing_code(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path, ["print(sum(range(101)))", "ret({'answer': 5050})"]
    )

    outcome = await query("add up 0..100")

    assert outcome.text == '{"answer": 5050}'
    assert outcome.metadata["rlm_returned"] is True
    assert outcome.metadata["rlm_depth"] == 1
    totals = query.take_totals()
    assert (totals.children, totals.turns, totals.cells, totals.deepest) == (1, 2, 2, 1)
    # The child's turns are sub-calls on the parent's bridge, so one ledger sees
    # them: recursion cannot spend its way outside the cell's ceiling.
    assert bridge.totals().calls == 2
    assert provider.systems == [CHILD_SYSTEM, CHILD_SYSTEM]
    assert "5050" in provider.prompts[1], "the second turn must see the first cell"
    assert not made[0].alive


async def test_the_depth_ceiling_answers_with_one_plain_call(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path, ["a direct answer, no child"], limits=RlmLimits(max_depth=0)
    )

    outcome = await query("something that looks like a task")

    assert outcome.text == "a direct answer, no child"
    assert outcome.metadata["rlm_fallback"] is True
    assert query.take_totals().fallbacks == 1
    assert made == []
    assert bridge.totals().calls == 1


async def test_a_child_at_the_ceiling_cannot_recurse_for_real(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path,
        # The child writes a cell that asks for a recursion of its own; the depth
        # guard turns that into the second scripted line: one plain completion.
        ["ret({'inner': rlm_query('nested task')})", "the nested answer"],
    )

    outcome = await query("outer")

    assert outcome.text == '{"inner": "the nested answer"}'
    totals = query.take_totals()
    assert totals.children == 1
    assert totals.fallbacks == 1, "the nested call must degrade, not nest"
    assert provider.prompts[-1] == "nested task"
    assert len(made) == 1, "only one child kernel should ever exist"


async def test_children_share_the_parents_call_budget(tmp_path) -> None:
    provider = _ScriptedModel(["print(1)", "print(2)", "print(3)", "print(4)", "ret(1)"])
    bridge = SubcallBridge(provider=lambda: provider, model=lambda: "fake")
    bridge.max_calls_per_cell = 3
    query = RlmQuery(
        bridge=bridge,
        kernel_factory=lambda rlm: ProcessKernel(
            cwd=tmp_path, subcall=bridge, rlm=rlm, max_concurrent_subcalls=1
        ),
        limits=RlmLimits(max_turns=9),
    )
    query.begin_cell()

    with pytest.raises(RuntimeError, match="sub-call budget"):
        await query("keep going")

    assert bridge.totals().calls == 3, "recursion must not spend past the cell's ceiling"


async def test_the_per_cell_child_budget_refuses_extra_recursions(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path, ["ret(1)", "ret(2)"], limits=RlmLimits(max_depth=2, max_children_per_cell=1)
    )

    await query("first")
    with pytest.raises(RlmBudgetError, match="recursion budget"):
        await query("second")

    assert query.take_totals().refusals == 1


async def test_out_of_turns_the_child_answers_in_plain_text(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path, ["print('partial')", "the answer from what ran"], limits=RlmLimits(max_turns=1)
    )

    outcome = await query("a task that takes too long")

    assert outcome.text == "the answer from what ran"
    assert outcome.metadata["rlm_answer"] == "final turn"
    totals = query.take_totals()
    assert totals.turns == 2 and totals.cells == 1


async def test_a_cancelled_cell_stops_recursing(tmp_path) -> None:
    class _Cancelled:
        def is_cancelled(self) -> bool:
            return True

    query, provider, bridge, made = _query(tmp_path, ["print(1)"], limits=RlmLimits(max_depth=2))
    query.begin_cell(_Cancelled())

    with pytest.raises(RlmBudgetError, match="cancelled"):
        await query("do work")

    assert bridge.totals().calls == 0
    assert made == []


async def test_a_child_kernel_is_closed_when_its_model_call_fails(tmp_path) -> None:
    class _Broken:
        def stream_response(self, **kwargs: Any) -> AsyncIterator[Any]:
            async def empty():
                if False:
                    yield None

            return empty()

    bridge = SubcallBridge(provider=lambda: _Broken(), model=lambda: "fake")
    made: list[ProcessKernel] = []

    def factory(rlm):
        kernel = ProcessKernel(cwd=tmp_path, subcall=bridge, rlm=rlm)
        made.append(kernel)
        return kernel

    query = RlmQuery(bridge=bridge, kernel_factory=factory, limits=RlmLimits(max_depth=2))
    query.begin_cell()

    await query("anything")

    assert made and not made[0].alive, "a finished recursion must not leak a kernel"
    assert query.take_totals().children == 1


async def test_a_child_that_prints_nothing_is_told_so(tmp_path) -> None:
    query, provider, bridge, made = _query(
        tmp_path, ["x = 1", "ret(x + 1)"], limits=RlmLimits(max_turns=3)
    )

    outcome = await query("compute two")

    assert outcome.text == "2"
    assert "(no output)" in provider.prompts[1]


async def test_an_old_turn_collapses_but_its_tail_survives(tmp_path) -> None:
    # The other half of the same failure. The warning tells the child the print was
    # cut; it does not give back the row past the cut. A dump older than the newest
    # turn collapses to its first line, its last line, and a byte count, so the row
    # at the end of a file is still on screen when the child plans its next cell.
    rows = [f"{i},a-value-wide-enough-to-cost-bytes" for i in range(2000)]
    (tmp_path / "ledger.csv").write_text("id,value\n" + "\n".join(rows) + "\n")
    query, provider, bridge, made = _query(
        tmp_path,
        [
            "print(open('ledger.csv').read())",
            "print('count', sum(1 for _ in open('ledger.csv')))",
            "ret('last row is 1999')",
        ],
        limits=RlmLimits(max_turns=3, max_observation_bytes=256, max_history_bytes=1200),
    )

    outcome = await query("find the last row of ledger.csv")

    assert outcome.text == "last row is 1999"
    third = provider.prompts[2]
    assert "middle omitted" in third, "the old turn must say it was collapsed"
    assert "1999,a-value" in third, "the tail of a collapsed turn has to survive"
    assert "--- cell 1 ---" in third and "--- cell 2 ---" in third


async def test_the_history_block_does_not_grow_with_turns(tmp_path) -> None:
    # Three chatty turns used to be re-pasted in full every turn: the clip was per
    # observation and the prompt still scaled with turn count.
    cells = [f"print('x' * 4000); print({i})" for i in range(3)]
    query, provider, bridge, made = _query(
        tmp_path,
        cells + ["ret('done')"],
        limits=RlmLimits(max_turns=4, max_history_bytes=1500, max_observation_bytes=4200),
    )

    outcome = await query("count things")

    assert outcome.text == "done"
    sizes = [len(prompt) for prompt in provider.prompts]
    # Every turn printed ~4 KB. With no history budget the prompts grow by roughly
    # 4 KB per turn; the block is capped, so the last prompt may not be much bigger
    # than the second one.
    assert sizes[-1] <= sizes[1] + 1900, f"history scaled with turns: {sizes}"


async def test_a_cut_observation_warns_first_and_keeps_both_ends(tmp_path) -> None:
    # Measured on a real model: a child printed a 49 KB ledger, was shown 2 KiB of
    # it with the drop count at the bottom, and concluded the missing row did not
    # exist. Two things were wrong -- the warning arrived after the wall of text, and
    # what it kept was only the head, so the row the child wanted was exactly the part
    # that was gone. Both are pinned here.
    rows = "\n".join(f"row-{i},a-value-wide-enough-to-cost-bytes" for i in range(2000))
    (tmp_path / "ledger.csv").write_text(rows + "\n")
    query, provider, bridge, made = _query(
        tmp_path,
        ["print(open('ledger.csv').read())", "ret({'done': True})"],
        limits=RlmLimits(max_observation_bytes=512),
    )

    outcome = await query("find the last row of ledger.csv")

    assert outcome.metadata["rlm_returned"] is True
    second = provider.prompts[1]
    notice = second.index("bytes were cut out of the middle")
    first_row = second.index("row-0,")
    assert notice < first_row, "the child must be warned before it reads the head"
    totals = re.search(r"\[(\d[\d,]*) of (\d[\d,]*) bytes were cut out", second)
    assert totals, "the notice states both totals"
    cut, total = (int(x.replace(",", "")) for x in totals.groups())
    assert total > cut > 0, "the notice states both totals"
    kernel_cut = re.search(r"middle omitted: ([\d,]+)", second)
    assert kernel_cut, "the kernel's own cut is visible too, not just the recursion's"
    assert "middle omitted" in second
    assert "row-1999," in second, "the end of the output has to survive the cut"
    assert "row-999," not in second, "the middle stays out: both ends, not everything"
    assert "search or count it in code" in second, "the child is told how to get the rest"


def test_the_child_is_told_a_clipped_print_proves_nothing() -> None:
    assert "clipped" in CHILD_SYSTEM
    assert "absent" in CHILD_SYSTEM
