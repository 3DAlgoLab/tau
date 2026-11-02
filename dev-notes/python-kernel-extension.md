---
title: "Persistent Python kernel example extension"
---

Tau now ships `examples/extensions/python_kernel.py`, an extension that adds one
tool, `python`, backed by a single interpreter that stays alive for the whole
session. Names, imports, loaded data, open files, and live object handles carry
from one cell to the next, and only output returns to the conversation. It ports
the intent of Pi's [`pi-pytool`](https://github.com/3DAlgoLab/pi-pytool) onto
Tau's extension API.

## Problem

Every `bash python3 -c ...` call pays a cold interpreter and throws its
namespace away before the next call. A model doing five dependent computations
starts five interpreters and re-derives every value by hand; a model driving a
database session, a scratch dataset, or an RPC/COM proxy cannot hold the handle
at all. Tau has no code-mode tool, so before this extension the only Python
route was `bash`.

## Shape

Three processes' worth of design collapse to two here, because the bridge is
pure stdlib:

```text
Tau session ── exec tool ──> PythonKernel (host side, asyncio)
                                 │  JSON lines over stdin/stdout
                                 └──> bridge subprocess (persistent namespace)
```

- The bridge is embedded in the extension as a source string and started with
  `<interpreter> -u -c <source>`, so installing the extension stays one file.
- A cell is parsed with `ast`; a trailing expression is evaluated separately so
  its `repr` comes back as `value:` without the model having to write `print`.
- Captured stdout/stderr are bounded inside the bridge (2 MB, drop markers), so
  a runaway `print` loop cannot outgrow the pipe; the host then applies the
  model-facing caps (46,080 chars per cell, 4,096 per line, both ends kept on a
  middle cut).
- Each result carries the current public names, so the transcript line and
  `/python` can show what is live without an extra round trip.

## Interrupts, deadlines, and reaping

A cell can be handed a wall clock (`timeoutMs` per call,
`TAU_PY_CELL_TIMEOUT_MS` per session). The host escalates in three steps:

1. **Interrupt.** SIGINT to the bridge lands at the next Python bytecode
   boundary. The cell returns `status: timeout`, keeps its partial output, and
   the namespace and handles survive. The result says so in one line.
2. **Grace.** If no answer arrives within `TAU_PY_ABORT_GRACE_MS`, the cell is
   parked inside a blocking C call, where no signal can reach it.
3. **Kill.** The bridge is terminated, and the namespace dies with it. The
   result says `reaped`, and the next cell boots a fresh kernel and starts with
   a `<tau_python_kernel_reset>` notice.

There are deliberately no namespace snapshots: a live DB connection or RPC proxy
cannot be pickled, so a rebuilt kernel starts empty and says so out loud rather
than pretending to restore.

Pi interrupts through Jupyter's ZMQ control channel; this port sends SIGINT
directly. Same guarantee, same failure mode, one less dependency: Tau installs
need no `ipykernel`/`jupyter_client` venv build.

## Interpreter and environment ladder

```text
TAU_PYTHON / PYTOOL_PYTHON / PI_PYTHON pin
  →  the project's own environment  (.venv, or venv, at or above the cwd)
  →  the shared environment         (<TAU_HOME>/.venv, i.e. ~/.tau/.venv)
  →  PATH python3, python
  →  sys.executable
```

`/python` names the interpreter and the rule that picked it, because a failed
import is an install into *that* interpreter, not a reason to rewrite the cell.

The pin still wins over both environments: `TAU_PYTHON=/path/to/env` stays a
one-line escape hatch. The environment rungs exist because a kernel booted on
Tau's own interpreter installs packages into an environment the session can never
see again: in a repository it should be the environment the repository runs, and
outside one it should at least survive the session.

### Project scope or shared scope

`resolve_env_target` decides which environment a cwd belongs to before any
interpreter is chosen:

- **Project scope** — the cwd, or an ancestor inside the search bound, holds
  `.venv`, `venv`, `pyproject.toml`, `uv.lock`, or a `requirements*.txt`. That
  folder owns its Python, so an environment that does not exist yet is still
  reported against `<cwd>/.venv`: the fix is to create it, not to leave the
  project. `venv` is accepted alongside `.venv` because plenty of real repos name
  it that way, and `.venv` still wins when both exist.
- **Shared scope** — nothing in scope claims the folder. The session then uses one
  environment under `TAU_HOME` (`~/.tau/.venv` by default) instead of whatever
  `python3` the shell exposes. A scratch directory gets a stable place to install
  packages, and the location follows the same `TAU_HOME` rule as the rest of Tau.

The split matters for provisioning as much as for booting: only a project
environment is ever filled from manifests, so a stray `requirements.txt` in a
non-project folder cannot push its dependencies into the shared environment.

### Why the walk is bounded

`find_project_venv` walks up from the session cwd to the nearest `.venv` or
`venv`, but no further than the repo root — the nearest ancestor holding `.git` —
or, with no repo, no further than `$HOME`. Without that bound a stray `~/.venv`
becomes the kernel for every unrelated project that happens to have no venv of
its own, so `$HOME`'s own venv counts only when the session actually runs in
`$HOME`. That is also why the shared environment lives at `~/.tau/.venv` rather
than `~/.venv`: the shared rung is reached by an explicit rule, never by an
upward walk that a home-directory venv can hijack. A venv directory holding no
interpreter is reported as `broken` rather than quietly replaced by system Python:
a half-created venv is a repair job, not a fallback.

`TAU_PY_VENV` picks the mode:

| Mode | Behavior |
| --- | --- |
| `prefer` (default) | use an existing environment; with none, the first cell names the exact uv command that creates it |
| `require` | provision the resolved environment with uv before the first cell runs |
| `off` | skip both environment rungs entirely (the earlier PATH ladder) |

The outcome is one state word — `active`, `missing`, `broken`, `pinned`, `off` —
plus the scope (`project` or `shared`), shown by `/python` and carried in each
cell result's `details.venvState` so renderers and tests can see it without
parsing text.

### Provisioning stays uv's job

`provision_venv` (async, ahead of the first cell in `require` mode) and
`provision_venv_sync` (`/python venv`, sync because command handlers cannot
await) share one plan builder. `uv venv <dir>` comes first, and the dependency
step is read from the project's own manifests:

- `uv.lock` present → `uv sync`, because a lockfile means the project owns its
  resolution.
- only `requirements*.txt` present → `uv pip install --python <venv python> -r <file>`,
  which fills the venv without touching the project's own metadata.
- `pyproject.toml` without a lock → `uv sync`.
- nothing → create the environment and stop.

In shared scope only the first step runs: the shared environment belongs to no
repository, so no manifest of the cwd is ever installed into it.

An existing interpreter is reused unless `--force` is passed; uv is located via
`TAU_PY_UV` then PATH; a missing uv is reported as `uv missing` with the fallback
named rather than guessed around; each uv step gets its own 300 s budget.

### Resolution happens per cell, not at setup

The session cwd arrives after `setup` runs, and the agent may create `.venv`
mid-session — that is the whole point of the `prefer` hint. So every cell
re-resolves, and if the answer moved the kernel is shut down and rebuilt with a
`kernel moved … namespace reset.` line prepended to the result. Restarting is
deliberate: pretending a name still lives in the environment the kernel just left
is the one failure mode this extension cannot recover from afterwards.

## Tau APIs used

- `tau.register_tool(AgentTool(...))` with `execution_mode="sequential"` so two
  parallel tool calls can never interleave cells in one kernel.
- `render_call`/`render_result` keep the transcript to one line per cell;
  `prompt_guidelines` carries the usage rules (reuse names, many short calls for
  blocking APIs, treat a reset notice as a dead namespace, never compute through
  `bash`, repair a `venv: missing|broken` in `bash` with uv instead of deleting
  it or piping Python through the shell).
- `session_start` re-binds the kernel to the session cwd and drops any inherited
  one (`/new`, `/fork`, `/reload`); `session_shutdown` stops the process so no
  interpreter is left holding handles after Tau exits.
- `tau.register_command("python", ...)` gives read-only diagnostics plus
  `reset` and `venv [--force]`; command handlers are sync-only, so both the reset
  path and the provisioning path run without awaiting the loop
  (`provision_venv_sync` rather than the async twin the first cell uses).

## Configuration

| Variable | Effect |
| --- | --- |
| `TAU_PYTHON` (or `PYTOOL_PYTHON`, `PI_PYTHON`) | pin the interpreter, trusted without probing; wins over both environments |
| `TAU_PY_VENV` | `prefer` (default), `require`, or `off` for the project/shared environment rungs |
| `TAU_PY_UV` | pin the uv binary used to provision an environment (else PATH) |
| `TAU_HOME` | moves the shared environment (default `~/.tau`, so `~/.tau/.venv`) |
| `TAU_PY_CELL_TIMEOUT_MS` | default per-cell wall clock; `0` (off) by default |
| `TAU_PY_ABORT_GRACE_MS` | grace before an aborted cell's kernel is reaped (`5000`) |
| `TAU_PY_MAX_OUTPUT_CHARS` | per-cell output cap (`46080`) |
| `TAU_PY_LINE_CHARS` | per-line cap (`4096`) |

## Testing

```bash
uv run pytest tests/test_python_kernel_extension.py -q   # 16 tests
uv run pytest tests/test_example_extensions.py -q        # shipped-example load
```

The kernel tests drive a real interpreter: persistence across cells, a failing
import that keeps the namespace, a deadline that interrupts an interruptible
cell, and a `SIG_IGN` cell that gets reaped and reports the reset. One test pins
the UTF-8 transport regression Pi measured (a `print("재질 20° Ø8")` cell must not
arrive as replacement characters). The wiring tests drive a real `CodingSession`
with `FakeProvider` and assert that a name defined in one turn is readable in
the next turn's cell.

The venv rung is not covered by the shipped suite yet, and the existing
interpreter-ladder tests still call the pre-venv `resolve_interpreter(env)` /
`settings_from_env(env)` signatures, so they fail against this revision until
they are re-pointed at `choose_interpreter(env, cwd)` and
`settings_from_env(env, cwd)`. The cases worth pinning, in order of how badly
they would bite: the ancestor walk and its repo/`$HOME` bound, a `$HOME` venv
refusing to hijack a project, a broken venv reported rather than bypassed, the
uv plan per manifest kind, and reuse-vs-rebuild on a second run.

## Limits and likely follow-ups

- Output is buffered per cell, not streamed; a long-running cell is silent
  until it settles. `on_update` is the seam for progress later.
- SIGINT interruption is the POSIX path. On Windows a child has no SIGINT, so
  the ladder degenerates to interrupt-fail-then-kill; a `CTRL_C_EVENT` group
  send is the follow-up if Windows gating matters.
- The kernel boots on the first cell, not at `session_start`, so no interpreter
  is spawned for sessions that never use Python. A warm boot would move the
  ~30 ms start off the first cell.
- Ported from Pi but not yet implemented here: the import-hint (naming the pip
  command for a missing package) and the `bash`-pushback hook that warns when
  Python is routed through the shell instead of the kernel.
- Venv provisioning only knows uv. A project pinned to pip/pip-tools gets a
  working `uv venv` plus a `uv pip install -r` translation, which is usually
  right, but there is no path yet for projects whose setup depends on a specific
  installer (an editable install with build requirements, for instance).
- `/python venv` blocks the TUI while uv runs. The cell path is async; the
  command path is sync because the extension API's command handlers are, so a
  progress event would have to come from a different seam.
