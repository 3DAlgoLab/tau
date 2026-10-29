# Code-Mode Kernel (Phase 2a)

Deterministic compaction decides what a summary contains. RLM decides what the
harness *is*: an inference environment where the model examines, decomposes, and
recursively calls itself over its own input. The upstream design is explicit that
this is a code environment, not a tool-call convention -- context lives as a
variable the model slices, and sub-LM calls are functions in that code.

Phase 2a builds the ground that sits on: one persistent Python kernel per
session, driven by the host over a socket. It is mechanism only. Nothing here is
wired into a session, a tool, or a prompt, exactly as the compaction core was
mechanism only until Phase 1d.

## What was added

```text
src/tau_coding/kernel/
  frames.py      length-prefixed JSON framing, both directions
  protocol.py    frame vocabulary, CellResult/KernelState, the Kernel protocol
  scaffold.py    kernel-side binding names, their help text, reserved names
  child.py       the child process: persistent namespace, exec-per-cell, limits
  process.py     the host side: spawn, pump, sub-call serving, kill and restart
```

## One channel, two directions

The host owns a localhost server and the child connects to it. Everything --
cells, output, state requests, and sub-call requests the model issues from
inside a cell -- is a framed message: 4-byte big-endian length, then that many
UTF-8 bytes of JSON.

A socket rather than the child's standard streams, deliberately:

- the model's own ``print`` output and our control frames must never share one
  byte stream, or framing becomes parsing prose;
- a sub-call issued inside a running cell has to reach the host *while that cell
  is still executing*, which a request/response stdio loop cannot express;
- the child's real stdout/stderr stay free for C-level writes and subprocesses,
  and the host drains them continuously into a bounded tail -- which is what
  makes a failed start explainable instead of silent.

The port is reachable by any local process, so the connection is adopted only
after a token handshake, and a second connection is refused rather than
multiplexed onto a namespace that belongs to the first.

## The boundary rules

**Values do not cross.** A variable created by model code stays in the kernel.
The host receives three things: byte-bounded output, a size-only inventory of
what is bound (``SHOW_VARS()``, and a ``state`` request that never touches
values), and whatever the code reports through ``ret()``. This is the whole
economic argument of the design: a 5 MB list can be sliced in the kernel and only
the slice the model decided to print enters anybody's context.

**Cell errors are data.** Anything a cell raises is formatted and reported; it
never takes the kernel down. A ``SystemExit`` from ``sys.exit()`` is reported the
same way. Reserved names are re-asserted after every cell so a cell cannot
clobber ``llm_query`` by assigning to it.

**Output is bounded and the loss is counted.** Per-cell streamed text is capped in
UTF-8 bytes and the omitted remainder is counted, in the child and again in the
host. A runaway loop reports how much it swallowed, which is the same
conservation rule the compaction renderer uses.

**Kill is the only interrupt.** Python offers no way to stop an ``exec`` from
outside, so a cell over its budget loses its process and its namespace. The
result carries ``timed_out`` and ``state_lost`` and says so in prose the model
reads: nothing that cell bound survives. A killed kernel restarts lazily on the
next cell; a *closed* kernel refuses, so a leaked handle cannot resurrect an
execution environment behind its owner's back.

**Credentials do not cross.** The kernel deliberately has the same file and
network reach as the ``bash`` tool. The boundary that matters is the harness's own
objects and secrets, so the child environment is filtered: anything that looks
like a key, token, password or credential -- plus provider and ``TAU_*``
prefixes -- is stripped before the process starts. The kernel's own handshake
values are injected after filtering.

**A reset peer is a dead kernel, not a raised exception.** The transport pushes
its `ConnectionResetError` into the `StreamReader`, so `read_frame` translates a
reset at either stage -- header or body -- into the same `FrameError` a failed write
already reports. Left as an `OSError` it would fly out of the server task instead of
ending the kernel with a reason, and under load that turns one dead child into
failures in whichever tests hold the loop.

**Sub-calls are bounded.** The host serves them through one semaphore shared by
single and batched requests, so kernel-side fan-out cannot exceed
``max_concurrent_subcalls`` regardless of how many ways the cell asks. A failure
inside a batch becomes a ``None`` hole in the response list rather than a lost
cell, because one unreachable prompt must not destroy the work of the others.

## Mapping to the upstream design

| upstream | here |
|---|---|
| ``LocalREPL`` namespace, ``exec`` per cell | ``child._Kernel.execute`` |
| globals ``llm_query`` / ``llm_query_batched`` | closures injected by the child, served by the host over the channel |
| ``_SubcallBroker`` + length-prefixed JSON | ``process._pump`` + ``frames`` |
| ``SHOW_VARS()`` names-and-types only | ``scaffold.SHOW_VARS`` and ``KernelState`` |
| ``answer`` side channel instead of probing the namespace | ``ret()`` |
| ``max_concurrent_subcalls`` | ``_bounded`` under one semaphore |
| ``custom_tools`` as code strings, callables cannot cross | only JSON crosses; bindings live in the child |
| ``context_N`` payload | ``bind(name, path)`` -- a handle, not a payload (Phase 2c extends it) |
| ``rlm_query`` among the injected globals | an opt-in ``rlm`` frame served by ``RlmQuery`` off a depth guard, absent unless the session asks (Phase 2c-iii) |

Upstream injects its bindings as *generated source* because its kernel is a
foreign process it does not own. We own the child, so the bindings are ordinary
closures that RPC back to the host -- same boundary, less machinery.

## The seam

``protocol.Kernel`` is the seam. ``ProcessKernel`` is option A: plain Python,
resource-capped by ``setrlimit`` (POSIX, best effort; reported through
``limits_notes`` when not applied), kill as the only interrupt. Option B is an
``ipykernel`` subprocess with hard per-cell interrupt and a surviving namespace,
and nothing above this protocol has to change to get it. Option A was chosen
first because it needs no new dependency, is testable in CI with the real child,
and the cells that dominate an RLM run are sub-call-dominated -- the expensive
work is in ``llm_query``, not in user code, so losing a namespace on overrun is
cheap most of the time.

## Non-goals for this phase

- No tool, no session wiring, no prompt: that is Phase 2b, as an extension.
- No ``tau_ai`` dependency: the host is handed a ``SubcallFn``. Phase 2b binds it
  to a provider call, and the same primitives then serve a non-recursive fan-out
  (code review) without ever naming RLM.
- No depth guard: recursion is not this phase's problem. It arrives as
  ``rlm_query`` in Phase 2c-iii.
- No context-as-variable handle beyond a bound path.
- No Windows claim: framing and the socket are portable, resource limits are not.

## Tests

``tests/test_kernel_frames.py`` (15) covers the codec: strict encoding, refused
``NaN``, torn frames reported as errors rather than clean EOF, oversized lengths.

``tests/test_kernel_process.py`` (25) spawns the real child, because the
guarantees under test only exist at the process boundary:

- namespace persistence, ``ret()`` data and the reason a non-data ``ret()`` was
  dropped, cell exceptions keeping the kernel alive;
- byte cap with an exact omission count;
- overrun → killed process, ``state_lost``, lazy restart, and the fresh namespace
  proven by the next cell;
- sub-call round trips with the exact prompt/model list the host received, a
  batch failure leaving ``None``, and a clear refusal when no handler is wired;
- a parametrized credential-filter table plus a live check that injected secrets
  read as ``None`` inside the kernel while ordinary variables travel;
- ``SHOW_VARS`` and ``state`` inventory without values;
- ``bind`` exposing a path (and refusing to shadow a reserved name), ``reset``
  dropping state but keeping bindings;
- a foreign connection with a guessed token refused without disturbing the real
  kernel;
- ``setrlimit`` values read back from inside the child;
- ``close`` being terminal: the pid is gone and ``state()`` says so.

## Next

Phase 2b wires this to a session: the ``python_repl`` tool, the host-side
``SubcallBridge`` that answers ``llm_query`` with the session's current model, and
the lifetime rules. See [Code Mode As A Session Tool](./code-mode-tool-and-subcalls.md).
