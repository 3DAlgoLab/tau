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

## Interpreter ladder

```text
TAU_PYTHON / PYTOOL_PYTHON / PI_PYTHON pin  →  PATH python3, python  →  sys.executable
```

`/python` names the interpreter and the rule that picked it, because a failed
import is an install into *that* interpreter, not a reason to rewrite the cell.

## Tau APIs used

- `tau.register_tool(AgentTool(...))` with `execution_mode="sequential"` so two
  parallel tool calls can never interleave cells in one kernel.
- `render_call`/`render_result` keep the transcript to one line per cell;
  `prompt_guidelines` carries the usage rules (reuse names, many short calls for
  blocking APIs, treat a reset notice as a dead namespace, never compute through
  `bash`).
- `session_start` re-binds the kernel to the session cwd and drops any inherited
  one (`/new`, `/fork`, `/reload`); `session_shutdown` stops the process so no
  interpreter is left holding handles after Tau exits.
- `tau.register_command("python", ...)` gives read-only diagnostics plus
  `reset`; command handlers are sync-only, so the reset path terminates the
  process without awaiting the loop.

## Configuration

| Variable | Effect |
| --- | --- |
| `TAU_PYTHON` (or `PYTOOL_PYTHON`, `PI_PYTHON`) | pin the interpreter, trusted without probing |
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
