# Code Mode As A Session Tool (Phase 2b)

Phase 2a built a kernel nothing could reach. Phase 2b makes it reachable by the
model: one `python_repl` tool, a host-side answerer for the sub-calls that code
inside the kernel makes, and the session rules that decide when a kernel exists at
all.

Policy lives in `tau_coding/code_mode.py`. The kernel package stays mechanism, and
`tau_coding/kernel/subcall.py` is the only place that knows how to spend a provider
request on behalf of model-written code.

## What was added

```text
src/tau_coding/code_mode.py       the python_repl tool, one kernel per session, result shape
src/tau_coding/kernel/subcall.py  SubcallBridge: kernel prompt -> provider completion, and its ledger
src/tau_coding/kernel/protocol.py Kernel.abort_cell, CellResult.cancelled
src/tau_coding/kernel/process.py  abort-by-kill, and a stop that answers a cell still starting
src/tau_coding/shell_config.py    codeMode setting (default on)
src/tau_coding/session.py         tool composition, kernel lifetime
src/tau_coding/cli.py             settings -> session config on the RPC and print paths
src/tau_coding/tui/app.py         settings -> session config on the TUI path
```

## One tool, three names in the schema

`python_repl` takes `code`, `description`, and an optional per-cell `timeout`. Its
label is `python`, and it is declared `execution_mode="sequential"`: one namespace
cannot safely interleave two cells, whatever the executor would otherwise prefer.

The prompt guidance is the part that decides whether the tool pays for itself, so
it states the invariants rather than leaving them to be discovered:

- use it when the work is shaped like data -- parsing, slicing, counting,
  comparing, or one question over many chunks;
- the namespace persists, so bind once and reuse instead of re-reading;
- do not print large objects to inspect them; end the cell with `ret({...})`;
- keep the corpus in kernel variables and bring back answers, not the corpus;
- a cell that overruns its timeout dies with its namespace, so split long work
  across cells.

`bash` stays the tool for project commands. The difference is not "shell versus
Python"; it is that a shell forces every intermediate through a text pipe, while a
namespace can hold a 5 MB list and return only the slice the model decided
mattered. That difference is the whole economic argument for recursion, so it is
also the argument for this tool existing at all.

## Sub-calls: the model in the cell is the session's model

`llm_query` and `llm_query_batched` reach the host over the kernel's control
channel, and `SubcallBridge` answers them. Three properties are load-bearing:

**The model is resolved per call.** The bridge holds thunks --
`provider=lambda: harness.config.provider`, `model=lambda: harness.config.model` --
not the values. `/model` and `reload_plugins` rebuild the harness; a kernel that
outlives that switch must ask its question of the model the session is using now,
not the one that was current when the kernel started.

**A sub-call is a leaf.** The request carries `tools=[]`, so recursion is
structurally impossible until an explicit `rlm_query` arrives with a depth guard.
Without that, one cell could start an agent that starts a kernel that starts an
agent, with nothing counting.

**Every sub-call is tallied.** Sub-calls are not session messages, so session usage
analytics cannot see them; the ledger in `SubcallTotals` is what keeps their cost
from vanishing. It reports calls, failures, models, requested and output tokens,
and estimated cost, and the tool result carries it: `[2 sub-calls via gpt-x, 3410
in / 118 out tokens]`. A cell that fans one question over forty chunks should see
what that cost, in the same place it sees the answer.

`max_calls_per_cell` (default 32) bounds total provider requests per cell, and a
failed attempt counts against it, because the failed attempt cost the request that
failed. Concurrency is bounded separately in the kernel (`max_concurrent_subcalls`,
default 4) so a batch cannot open an unbounded number of streams. `begin_cell`
hands the bridge the tool's cancellation token: a cell the user stops stops paying
for in-flight requests rather than merely stopping to read their answers.

There is deliberately no `rlm_query` yet. That name arrives in Phase 2c with a
depth guard, and until then the only recursion is the model calling itself once
per line of code it wrote.

## Lifetime: one kernel per conversation

The kernel is created lazily -- constructing `CodeMode` costs nothing, and a
session that never calls the tool never pays for a process.

- `/reload` rebuilds tools but keeps the same `CodeMode` object, and therefore the
  same namespace. Working memory is not a resource file; dropping it on reload
  would throw away the reason the tool exists.
- A new, resumed, or branched conversation adopts a replacement session and closes
  the previous kernel. A live process whose bound names belong to a dead
  transcript is a leak with a namespace attached.
- `CodingSession.aclose()` disposes code mode before closing providers.
- `CodeMode.dispose()` is permanent. A killed kernel restarts lazily on the next
  cell; a closed one refuses, so a stale handle cannot resurrect an execution
  environment behind its owner's back.

## Default on, with three ways out

Code mode ships in the default tool set, because a tool that is never on is a tool
nobody uses. The escape hatches are ordered by scope:

| Scope | Switch |
| --- | --- |
| Every session on this machine | `"codeMode": false` in `~/.tau/settings.json` |
| One host process | `CodingSessionConfig(code_mode_enabled=False)` |
| One host with its own tools | any explicit `config.tools` list |
| One session's tool set | `allowed_tool_names` without `python_repl` |

An explicit `tools` list owns its tool set: a host that builds one is not silently
handed a Python interpreter. That is why restricted subagents and one-shot RPC
callers need no new flag -- they already name their tools. The settings key is
user-global for the same reason the trust setting is: a project cannot opt itself
into running model-written code.

No CLI flag was added. `main()` dispatches to the runners with positional
argument tuples, and a boolean knob there would mean re-cutting that dispatch for
one feature; `codeMode` in settings plus the config field cover the cases, and a
flag can be added when the dispatch is next restructured.

## Result shape, and the two layers of shrinking

A cell result reports captured stdout and stderr, the `ret()` payload as data, and
what happened to the kernel. Two ceilings apply, and each says how much of itself
it cut:

- the kernel caps what it *captures* (default 64 KB per stream) and counts the
  omitted bytes;
- the tool caps what it *hands to the model* (400 lines / 16 KB) and appends
  `[showing first 400 of 5000 lines]`.

Silent truncation is the failure mode to avoid here: an agent that cannot tell its
output was cut will reason about a partial world.

Timeout, cancellation, and restart are each named in the result text, because each
one changes what the next cell can assume:

```text
[cell killed after exceeding its 30s budget; the namespace is gone; split the work into smaller cells]
[cell cancelled]
[the kernel restarted: names bound by this cell did not survive]
[12.4 KB of this cell's output was capped and not returned]
```

## The abort race, fixed in the kernel

Cancelling a cell while its process is still booting used to hang: the watch task
killed the kernel, `start()` kept waiting for a handshake that would never arrive,
and the caller sat until the startup timeout. Three changes close it:

- `_stop` wakes startup waiters, not only pending cells;
- `start()` refuses rather than proceeding with no channel;
- `execute()` registers its cell buffer *before* checking the process, so a stop
  that lands during startup answers that cell instead of raising past a caller that
  asked the cell to stop.

Option A has no interrupt, so cancellation is a kill and the namespace goes with
it. The result says so; an IPython-backed kernel (`Kernel` protocol, Phase 2a's
seam) can later answer the same call with a real interrupt and keep state.

## Still not here

- `bind()`, the file-handle path into the namespace, is on `ProcessKernel` but not
  exposed through the tool. That is Phase 2c's context-as-variable work.
- No `rlm_query`, no depth guard, no recursion.
- No TUI surface for code mode: no state panel, no per-session toggle.
- No per-conversation namespace isolation beyond "one kernel per session"; two
  branches of the same session share one kernel.

## Tests

`tests/test_kernel_subcall.py` (11) pins the bridge: the session model resolved per
call, an explicit model argument winning, `tools=[]` on every request, provider
errors raised and counted, a cancelled cell refusing to keep paying, byte-bounded
responses with the omission reported, empty answers flagged rather than trusted,
the per-cell budget (including failures against it), and the ledger resetting at
the cell boundary.

`tests/test_code_mode.py` (22) drives the tool through `AgentTool.execute` against
a real child process: namespace persistence, `ret()` as data and the reason a
non-data `ret()` was refused, a raising cell that does not end the kernel,
`SHOW_VARS` listing names without values, the output-cap note, overrun with
`state_lost` proven by the next cell's `NameError`, cancellation mid-cell and
during startup, sub-call totals in the result, a closed mode refusing, and input
refusals before any process is spawned. The session half checks the tool set on
the default path, an explicit tool list, `code_mode_enabled=False`, the allow
list, `/reload` keeping the same namespace, `aclose` closing the kernel, and a new
session closing the old kernel.

`tests/test_shell_config.py` (+7) covers `codeMode`: default on, both spellings
refused together, a non-boolean refused rather than read as truthy, and
round-trip with a shell prefix present.

Dev work runs from this checkout: `uv run tau ...` inside `tau-dev`, or a global
`uv tool install --editable .`. `TAU_HOME` is the one knob for moving a home.

## Next

Phase 2c replaces raw payloads with addressed handles: session-log entries exposed
to the kernel as bounded, sliceable references, so "read this 200 KB branch" is a
handle the kernel opens rather than a string pasted into a prompt. Phase 2d ports
the `fabric-rlm` skill on top of that, with `rlm_query` carrying the depth guard.
