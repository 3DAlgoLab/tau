# File Handles In The Kernel (Phase 2c-i)

Phase 2a gave the kernel a `bind(name, path)` operation and Phase 2b forgot to
expose it: the model could ask for a file only by reading it into `code`, which
is exactly the cost code mode exists to avoid. Phase 2c-i closes that hole --
`python_repl` now takes `attach`, and the bytes stay on disk.

## What was added

```text
src/tau_coding/code_mode.py   attach argument, per-call handle binding, size/name ceilings, [attached as paths: ...] line
tests/test_code_mode.py       4 new tests over the real child process
website/content/reference/tools.md  the attach argument and its semantics
```

No kernel change: `Kernel.bind` and the child's `_bind` already existed and were
already tested (`test_bind_hands_the_kernel_a_path_not_a_payload`). The gap was
purely policy -- nothing above the protocol could reach it.

## What was added, in words

`attach` is a mapping of namespace name to file path:

```json
{ "code": "...", "description": "Grepping the log", "attach": {"log": "tests/fixtures/app.log"} }
```

The host resolves the path (absolute, or relative to the session cwd), refuses
anything that is not a regular file, and calls `kernel.bind(name, path)`. The
cell receives a **path string** and opens it itself. Nothing about the file's
contents crosses the channel, so a 900 KB log costs the prompt the length of the
path.

## Design decisions

| decision | rationale |
|---|---|
| Restate handles per call instead of remembering them | bound names live in the namespace, and a kernel killed by an overrun loses its namespace; the cell that needs a handle says so, and re-binding is a cheap control frame. Silent durability would also hide which cell introduced which name. |
| Refusing one handle fails the whole call, before the cell runs | a half-attached cell meets a `NameError` three frames deep instead of the reason the handle was rejected. `ToolInputError` is how this tool already reports a bad argument. |
| Names validated on the host before the child sees them | the child already rejects reserved and non-identifier names; a host-side check turns that into a tool-argument error instead of a kernel round-trip, and keeps the reserved-name list in one import (`scaffold.RESERVED`). |
| Directories and devices refused | the contract is "open this file"; a directory handle invites `os.walk` recursion the model did not intend, and device paths are not data. |
| Size ceiling (2 MiB) with a message that says how to fix it | an attached file never enters a prompt, so the ceiling is not a context guard -- it stops the tool from blessing a handle a cell will then read whole into memory. The error names the remedy (slice it with `bash`) rather than just the limit. |
| Count ceiling (16) | a call that attaches dozens of paths has stopped describing what this cell reads. |
| Report the bound names and sizes in the result | the model otherwise cannot tell an attached name from a name it bound itself in an earlier cell, and `SHOW_VARS()` shows the path but not what it cost. |

## Guarantees

- A file's bytes never cross the kernel boundary; only its path does.
- A refused handle never runs a cell: no process is started, no cell is counted.
- A handle can be rebound after a kill, and the test proves both halves -- the
  name is gone, and the restated call brings it back.
- Reserved kernel bindings (`llm_query`, `ret`, `SHOW_VARS`, dunder names) cannot
  be shadowed by an attachment.

## How later phases use this

Phase 2c's context-as-variable work sits on the same rail: session entries become
addressed handles, and the same `bind` call delivers them as paths the kernel
opens and slices. Whatever crosses into the kernel from here on is a handle, not
a payload -- which is the difference between an RLM traversal and a paste.

## Verification

```bash
uv run pytest tests/test_code_mode.py tests/test_kernel_process.py -q
uv run mypy && uv run ruff check src tests && uv run ruff format --check src tests
```
