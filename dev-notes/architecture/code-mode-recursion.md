---
title: "Recursion inside python_repl (Phase 2c-iii)"
date: "2026-10-29"
---



# Recursion inside `python_repl`


`llm_query` asks the current model a question. `rlm_query(task)` hands it a *task*
and lets the model work: a fresh kernel, up to three cells in which it writes
code, reads what the code printed, and tries again, until it answers through
`ret(...)` or runs out of turns. That is the recursion primitive from the upstream
recursion-as-code-mode line (`rlm/rlm/core/rlm.py`) rebuilt out of Tau's own parts
-- the same `ProcessKernel`, the same `SubcallBridge`, the same per-cell ledger.


## Mechanism and policy, in different files


The kernel layer is transport and stays ignorant of recursion semantics:
`OP_RLM` / `OP_RLM_BATCHED` / `OP_RLM_REPLY` frames, one `RlmFn` handler installed
at start, `recursions_used` counted for the same kind of runaway loop that
`subcalls_used` counts. `scaffold.py` owns the two names and their `HELP()` text;
`child.py` posts the frame and blocks, exactly like a sub-call.


All the decisions live in `tau_coding/rlm.py`: how many turns, how deep, how many
children, what counts as an observation, when to give up. `RlmLimits` is one
frozen dataclass so the ceilings have one owner, and `RlmQuery` carries a `_Meter`
shared down the whole tree -- children are constructed with the parent's meter, so
a per-cell budget is per *cell*, not per kernel.


## The depth guard, read the way upstream reads it


The request is refused by *degrading*, not by raising:


```python
if next_depth > self._limits.max_depth:      # default max_depth = 1
    meter.fallbacks += 1
    return await self._bridge(task, model)   # one plain completion
```


`max_depth=1` means a cell gets one real generation: the child's own
`rlm_query` calls are answered by the plain sub-call path instead of another
kernel. Mirroring upstream's `max_depth=1` here means the same *behaviour* the
upstream default describes, not the same integer compared at the same place: at
the ceiling model-written code still gets an answer, and the outcome metadata says
`rlm_fallback: true` so nothing about the recursion is hidden. `max_depth=0`
disables children entirely; negative is refused at construction.


## One ledger, two budgets


Every child turn is a call on the parent's bridge, so children are bound by the
sub-call ceiling (`max_subcalls_per_cell`) and appear in the same totals: an
RLM cannot spend its way outside the budget, which is the property that makes it
shippable next to a tool the model calls freely. Tokens and processes are
separate costs, so there is a second knob: `max_children_per_cell` (4) bounds
kernels whatever the token ceiling allows, and going past it raises
`RlmBudgetError` rather than quietly starting a fifth process. `max_concurrent_children`
(2) is a semaphore held by the whole tree.


A recursion also has a wall-clock deadline (`deadline_seconds`, 120) checked
before every turn, and it obeys the cell's cancellation token: a cancelled cell
stops spawning the next child instead of finishing its tree.


## What the answer is worth


A child answers by `ret(value)`; the value is rendered as the answer text (JSON
for non-strings). Out of turns without a `ret`, one final plain-text completion is
asked so the parent gets the child's best reading of its evidence rather than an
error and a spent budget. Observations are capped at 2 KB each and the answer at
8 KB, both with the byte count reported when something was dropped -- the same
currency as everywhere else in code mode.


`CodeMode` reports the window next to stdout: `[1 recursion: 3 turns, 2 child
cells, depth 1]`, plus fallback and refusal counts when they happened, and puts
`RlmTotals.to_json()` in the result details. Recursion totals are kept out of the
sub-call sentence on purpose: the turns *are* counted there, and repeating them
would double-state the cost.


## Wiring, and why it is a config decision


`Session._wire_code_mode_recursion` builds the `RlmQuery` and the child-kernel
factory together, because a child answers through the *same* bridge as the
parent's sub-calls -- that is what makes its turns land in one ledger -- and only
the session can supply a bridge that resolves the live provider and model. The
tool description, on the other hand, has to promise `rlm_query` before any harness
exists, so `CodeMode(recursion_enabled=...)` carries the promise and
`bind_recursion()` delivers the runner; the two cannot drift, and a cell that
meets an advertised-but-unwired state gets a failure in words instead of a
`NameError`.


Off by default at three levels: `CodingSessionConfig.code_mode_recursion_enabled`,
`ShellSettings.code_mode_recursion` (`"codeModeRecursion": true` in
`~/.tau/settings.json`), and `CodeMode(recursion_enabled=False)`. A recursion costs
a process and a turn budget, which is a cost decision the user makes, not a
capability Tau assumes. `code_mode_rlm_max_depth` (1) is a
session-config knob, not a `settings.json` key: how deep to go is a host decision,
which ceilings to allow at all is the user's.


## Deliberately not here


No tool access inside a child, no transcript inheritance, no retrieval, no shared
namespace: a child sees the task, the cells it ran, and its own output, and is
disposed with its scratch directory when the answer comes back. A child can read
what its parent materialized (2c-ii slices are files on disk in the same cwd), so
passing a path in the task is the way history travels down; passing the transcript
as text would be the way this whole track exists to avoid.
