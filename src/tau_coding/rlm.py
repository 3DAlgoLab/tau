"""Recursion: hand a task to a child that solves it by writing code.

``llm_query`` asks the model a question. ``rlm_query`` hands the model a *task*
and lets it work at that task: a fresh kernel, a bounded number of turns in which
it writes code, sees what the code printed, and tries again, until it answers
through ``ret(...)`` or runs out of turns. That is the recursion primitive from
the upstream recursion-as-code-mode line (``rlm/rlm/core/rlm.py``) expressed with
Tau's own parts -- the same :class:`ProcessKernel`, the same
:class:`SubcallBridge`, the same per-cell ledger.

Three properties make the recursion safe to expose to model-written code:

* **A depth ceiling, checked before anything starts.** Mirroring upstream, a
  child asked for at or past ``max_depth`` is answered by one plain completion
  instead of another kernel. Degrading rather than raising is deliberate: model
  code that calls ``rlm_query`` at any depth still gets an answer, and the
  metadata says the recursion was not real.
* **One ledger.** Every turn of a child is a call on the parent's
  :class:`SubcallBridge`, so children are bound by the same per-cell call ceiling
  and appear in the same totals. Recursion cannot spend its way outside the
  budget, which is the property that makes it shippable at all.
* **A child is a kernel, not a session.** No tools, no transcript, no approval
  surface: it sees the task, the turns it has taken, and its own output. It is
  disposed when the answer comes back, scratch directory and all.

The budget knobs are here, not in the kernel package: the kernel forwards an
``rlm`` request the same way it forwards a sub-call and never looks at a depth.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tau_agent.tools import ToolCancellationToken
from tau_agent.types import JSONValue
from tau_coding.kernel.history import (
    DEFAULT_COLLAPSED_BYTES,
    DEFAULT_MAX_HISTORY_BYTES,
    DEFAULT_VERBATIM_TURNS,
    keep_ends,
    render_history,
)
from tau_coding.kernel.protocol import Kernel, RlmFn, SubcallOutcome, bounded_text

if TYPE_CHECKING:
    from tau_coding.kernel.subcall import SubcallBridge

DEFAULT_MAX_DEPTH = 1
"""How many generations of children one cell may start.

``1`` -- upstream's number, read the way its README describes the default --
means a cell gets a real child, and that child's own ``rlm_query`` calls degrade
to plain completions. A grandchild needs an explicit ``max_depth=2``: each
generation costs a process and a turn budget, which is why the default stops at
one.
"""

DEFAULT_MAX_TURNS = 3
"""Code cells one child may run before it has to answer."""

DEFAULT_MAX_CHILDREN_PER_CELL = 4
"""Recursions one cell may start, whatever depth they are at.

The sub-call ceiling bounds tokens; this bounds processes. A cell that starts
more children than this has stopped delegating work and started enumerating.
"""

DEFAULT_MAX_CONCURRENT_CHILDREN = 2
"""Children running at once, across the whole recursion tree."""

DEFAULT_CHILD_CELL_TIMEOUT_SECONDS = 20.0
"""Per-cell budget inside a child; overrunning kills that child kernel."""

DEFAULT_DEADLINE_SECONDS = 120.0
"""Wall-clock ceiling for one recursion, counted from the request."""

DEFAULT_MAX_ANSWER_BYTES = 8 * 1024
DEFAULT_MAX_OBSERVATION_BYTES = 2 * 1024

CHILD_SYSTEM = (
    "You are a sub-agent working inside a fresh Python kernel. You get one task "
    "and the output of the cells you have already run. Reply with exactly one "
    "Python cell and nothing else: no prose, no fences, no explanation. "
    "Statements run as written, so print() what you need to see -- but only the "
    "part you need: a print of a large file is clipped and the rest is gone for "
    "good, so filter, count and search in code instead of dumping. Never claim "
    "something is absent because it is missing from a clipped print. Finish by "
    "calling ret(value) with the answer as data -- that call ends the task. If "
    "the task cannot be done in code, ret({'answer': ...}) with what you have."
)

FINAL_TASK = (
    "You are out of turns. Answer the task in plain text, using only what the "
    "cells below already established. No code."
)


@dataclass(frozen=True, slots=True)
class RlmLimits:
    """Every ceiling one recursion tree obeys, in one place."""

    max_depth: int = DEFAULT_MAX_DEPTH
    max_turns: int = DEFAULT_MAX_TURNS
    max_children_per_cell: int = DEFAULT_MAX_CHILDREN_PER_CELL
    max_concurrent_children: int = DEFAULT_MAX_CONCURRENT_CHILDREN
    child_cell_timeout_seconds: float = DEFAULT_CHILD_CELL_TIMEOUT_SECONDS
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS
    max_answer_bytes: int = DEFAULT_MAX_ANSWER_BYTES
    max_observation_bytes: int = DEFAULT_MAX_OBSERVATION_BYTES
    max_history_bytes: int = DEFAULT_MAX_HISTORY_BYTES
    verbatim_turns: int = DEFAULT_VERBATIM_TURNS
    collapsed_turn_bytes: int = DEFAULT_COLLAPSED_BYTES

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative; 0 means no children at all")
        if self.max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if self.max_children_per_cell < 1:
            raise ValueError("max_children_per_cell must be at least 1")


@dataclass(frozen=True, slots=True)
class RlmTotals:
    """What one window of recursion cost, separate from the sub-call ledger."""

    children: int = 0
    fallbacks: int = 0
    turns: int = 0
    cells: int = 0
    refusals: int = 0
    deepest: int = 0

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "children": self.children,
            "fallbacks": self.fallbacks,
            "turns": self.turns,
            "cells": self.cells,
            "refusals": self.refusals,
            "deepest": self.deepest,
        }


class RlmBudgetError(RuntimeError):
    """A cell asked for more recursions than its budget allows."""


class _Meter:
    """Counters shared by a whole recursion tree, reset per cell."""

    def __init__(self, limits: RlmLimits) -> None:
        self.limits = limits
        self.semaphore = asyncio.Semaphore(limits.max_concurrent_children)
        self.children = 0
        self.fallbacks = 0
        self.turns = 0
        self.cells = 0
        self.refusals = 0
        self.deepest = 0
        self.deadline: float | None = None

    def reset(self) -> None:
        self.children = 0
        self.fallbacks = 0
        self.turns = 0
        self.cells = 0
        self.refusals = 0
        self.deepest = 0
        self.deadline = None

    def totals(self) -> RlmTotals:
        return RlmTotals(
            children=self.children,
            fallbacks=self.fallbacks,
            turns=self.turns,
            cells=self.cells,
            refusals=self.refusals,
            deepest=self.deepest,
        )


class RlmQuery:
    """Run one task as a child with a kernel of its own.

    Construct one instance per session at depth 0 and hand its
    :meth:`~__call__` to the kernel as the ``rlm`` handler; the children it
    creates get their own handler at ``depth + 1`` sharing the same meter, so the
    per-cell ceilings hold across the whole tree rather than per kernel.

    The factory is a callable taking the child's own ``rlm`` handler: the thing
    that owns kernel construction (the code-mode tool) decides what a child
    kernel is, and this module only decides when one is worth making.
    """

    def __init__(
        self,
        *,
        bridge: SubcallBridge,
        kernel_factory: Callable[[RlmFn], Kernel],
        limits: RlmLimits | None = None,
        depth: int = 0,
        meter: _Meter | None = None,
    ) -> None:
        self._bridge = bridge
        self._kernel_factory = kernel_factory
        self._limits = limits or RlmLimits()
        self._depth = depth
        self._meter = meter or _Meter(self._limits)
        self._signal: ToolCancellationToken | None = None

    # -- per-cell scope -----------------------------------------------------

    def begin_cell(self, signal: ToolCancellationToken | None = None) -> None:
        """Open the window this cell's recursions belong to.

        ``signal`` is the cell's cancellation token, kept so a cancelled cell
        stops recursing instead of spawning the next child: the bridge aborts the
        in-flight turn, and the disposal below still runs.
        """
        self._meter.reset()
        self._signal = signal
        self._meter.deadline = asyncio.get_running_loop().time() + self._limits.deadline_seconds

    def take_totals(self) -> RlmTotals:
        """Return the open window's totals and reset them."""
        totals = self._meter.totals()
        self._meter.reset()
        self._signal = None
        return totals

    # -- one recursion ------------------------------------------------------

    async def __call__(self, task: str, model: str | None = None) -> SubcallOutcome:
        meter = self._meter
        next_depth = self._depth + 1

        self._check_deadline(meter)
        if next_depth > self._limits.max_depth:
            # Upstream's ceiling behaviour, kept on purpose: past the ceiling the
            # RLM *is* an LM. The cell still gets an answer; the metadata says no
            # child ran, so nothing about the recursion is hidden.
            meter.fallbacks += 1
            outcome = await self._bridge(task, model)
            merged: dict[str, JSONValue] = {
                **dict(outcome.metadata),
                "rlm_depth": self._depth,
                "rlm_fallback": True,
            }
            return SubcallOutcome(text=outcome.text, model=outcome.model, metadata=merged)

        if meter.children >= self._limits.max_children_per_cell:
            meter.refusals += 1
            raise RlmBudgetError(
                f"this cell used its recursion budget of "
                f"{self._limits.max_children_per_cell}; ask fewer, larger questions "
                "or run the rest in a new cell"
            )

        meter.children += 1
        meter.deepest = max(meter.deepest, next_depth)
        child_rlm = RlmQuery(
            bridge=self._bridge,
            kernel_factory=self._kernel_factory,
            limits=self._limits,
            depth=next_depth,
            meter=meter,
        )
        kernel = self._kernel_factory(child_rlm)
        observations: list[str] = []
        answer: str | None = None
        turns = 0
        returned = False
        try:
            async with meter.semaphore:
                while turns < self._limits.max_turns:
                    self._check_deadline(meter)
                    turns += 1
                    meter.turns += 1
                    prompt = self._task_prompt(task, observations)
                    reply = await self._bridge.complete(prompt, model, system=CHILD_SYSTEM)
                    cell = _cell(reply.text)
                    if not cell:
                        observations.append("(the last turn returned no code)")
                        continue
                    result = await kernel.execute(
                        cell, timeout=self._limits.child_cell_timeout_seconds
                    )
                    meter.cells += 1
                    if result.returned:
                        answer = _render(result.ret)
                        returned = True
                        break
                    observations.append(self._observe(result))
                    if not kernel.alive:
                        observations.append("(the child kernel died; its namespace is gone)")
                        break
                if answer is None:
                    # Out of turns without an answer: ask once in plain text so
                    # the parent gets the child's best reading of the evidence
                    # rather than an error and a wasted budget.
                    meter.turns += 1
                    final = await self._bridge(self._final_prompt(task, observations), model)
                    answer = final.text
        finally:
            await kernel.close()

        text, omitted = bounded_text(answer or "", self._limits.max_answer_bytes)
        used_model = self._bridge.model_name(model)
        metadata: dict[str, JSONValue] = {
            "rlm_depth": next_depth,
            "rlm_turns": turns,
            "rlm_cells": meter.cells,
            "rlm_returned": returned,
            "rlm_answer": "ret" if returned else "final turn",
        }
        if omitted:
            metadata["omitted_bytes"] = omitted
        return SubcallOutcome(text=text, model=used_model, metadata=metadata)

    # -- prompts ------------------------------------------------------------

    def _task_prompt(self, task: str, observations: list[str]) -> str:
        if not observations:
            return f"Task: {task}\n\nWrite the first cell."
        block = self._history(observations, numbered=True)
        return f"Task: {task}\n\nOutput so far:\n{block}\n\nWrite the next cell."

    def _final_prompt(self, task: str, observations: list[str]) -> str:
        block = self._history(observations, numbered=False)
        return f"Task: {task}\n\n{FINAL_TASK}\n\nEvidence:\n{block}"

    def _history(self, observations: list[str], *, numbered: bool) -> str:
        """Collapse the child's old turns instead of pasting every one of them.

        A child that prints a big file relays that print into every later turn.
        Head-clipping each observation fixes the size of one entry and leaves the
        prompt growing with turns, and it hides the tail -- which is where the
        value a child is looking for usually sits. So the newest turn stays whole,
        older turns collapse to first line, last line, and a byte count, and the
        block as a whole is bounded whatever the turn count.
        """
        block = render_history(
            observations,
            budget_bytes=self._limits.max_history_bytes,
            verbatim=self._limits.verbatim_turns,
            collapsed_bytes=self._limits.collapsed_turn_bytes,
            numbered=numbered,
        )
        return block.text

    def _observe(self, result: object) -> str:
        stdout = getattr(result, "stdout", "") or ""
        stderr = getattr(result, "stderr", "") or ""
        error = getattr(result, "error", None)
        parts = []
        if stdout.strip():
            parts.append(stdout.rstrip())
        if stderr.strip():
            parts.append(f"[stderr] {stderr.rstrip()}")
        if error:
            parts.append(f"[error] {error}")
        if not parts:
            parts.append("(no output)")
        raw = "\n".join(parts)
        limit = self._limits.max_observation_bytes
        text, omitted = keep_ends(raw, limit)
        if not omitted:
            return text
        # The notice goes first. A child that printed a 49 KB file reads 2 KB of
        # it, and a note at the bottom of that wall of text arrives after the child
        # has already begun treating the text as the file. The failure measured on a
        # real model was exactly this: one print(f.read()) over a 2,401-row ledger,
        # then "row not found", because the row sat past the clip.
        return (
            f"[{omitted:,} of {len(raw.encode('utf-8')):,} bytes were cut out of the middle: "
            "you are reading the beginning and the end. If what you need is not in either, "
            "search or count it in code and print the matches instead of the whole thing.]"
            f"\n{text}"
        )

    def _check_deadline(self, meter: _Meter) -> None:
        """Refuse to start another turn once the cell is out of time or cancelled."""
        signal = self._signal
        if signal is not None and signal.is_cancelled():
            meter.refusals += 1
            raise RlmBudgetError("the cell was cancelled, so the recursion stopped")
        deadline = meter.deadline
        if deadline is not None and asyncio.get_running_loop().time() >= deadline:
            meter.refusals += 1
            raise RlmBudgetError(
                f"this recursion passed its {self._limits.deadline_seconds:g}s ceiling"
            )


def _cell(text: str) -> str:
    """Pull one code cell out of a model reply, tolerating fences."""
    body = (text or "").strip()
    if body.startswith("```"):
        lines = body.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    return body


def _render(value: object) -> str:
    """Render a child's ``ret()`` value as the answer text."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "CHILD_SYSTEM",
    "DEFAULT_DEADLINE_SECONDS",
    "DEFAULT_MAX_ANSWER_BYTES",
    "DEFAULT_MAX_CHILDREN_PER_CELL",
    "DEFAULT_MAX_CONCURRENT_CHILDREN",
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_MAX_OBSERVATION_BYTES",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_COLLAPSED_BYTES",
    "DEFAULT_MAX_HISTORY_BYTES",
    "DEFAULT_VERBATIM_TURNS",
    "FINAL_TASK",
    "RlmBudgetError",
    "RlmLimits",
    "RlmQuery",
    "RlmTotals",
]
