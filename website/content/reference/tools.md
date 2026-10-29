---
title: Built-in tools
description: The read, write, edit, and bash tools the agent uses to work in your project.
---

Tools are the actions the agent can take in your working directory. The model
decides when to call them; Tau executes them and streams the results back. Tau
ships four built-in coding tools: `read`, `write`, `edit`, and `bash`.

All paths are resolved against the session's working directory (`--cwd`, or the
directory you launched Tau from).

> [!NOTE]
> This page documents tool *behavior* — what the model can do on your machine. To
> build a frontend or register your own tools, see
> [Building a custom frontend](/internals/custom-frontend/).

## `read`

Reads a file from disk.

```json
{ "path": "README.md", "offset": 1, "limit": 40 }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to read (relative to `cwd`). |
| `offset` | no | integer | 1-indexed start line (`0` = start of file). |
| `limit` | no | integer | Maximum number of lines to return. |

For text files, `read` returns UTF-8 content, applies `offset`/`limit`, and
truncates to at most 2,000 lines or 50 KB (whichever comes first), appending a
hint like `[42 more lines in file. Use offset=101 to continue.]`. Supported
images (JPEG, static PNG, GIF, WebP, and BMP) are detected from their file
content and sent to vision-capable models as image attachments. BMP files are
converted to PNG. Tau validates images and, when necessary, resizes them without
upscaling or changing their aspect ratio. The processed attachment is limited to
2,000 pixels on either side and 5 MB. Processing also rejects source files above
50 MB or 40 million pixels. Animated PNG and JPEG XL inputs receive explicit
unsupported-format notices. If decoding, conversion, or resizing cannot produce
a safe attachment, Tau returns a clear omission notice.

When the active model does not accept images, `read` returns an explicit
text-only notice that says the image contents are unavailable and recommends
switching to a vision-capable model. It does not attach or process the image.
Provider serialization applies the same defensive downgrade to image blocks from
older sessions or other tools. This avoids invalid requests and reduces pressure
on text-only models to invent a visual description.

The Textual TUI shows this notice, but does not render images inline; native
terminal-image rendering remains outside the `read` tool's provider-neutral
contract.

Fails when `path` is missing/invalid, the file doesn't exist, the path is a
directory, `offset` is past the end, or the file is neither UTF-8 text nor a
supported image.

## `write`

Creates or overwrites a complete UTF-8 text file.

```json
{ "path": "src/example.py", "content": "print('hello')\n" }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to write (relative to `cwd`). |
| `content` | yes | string | Complete file contents. |

Creates missing parent directories and overwrites any existing file. Writes to
the same path are serialized within a process, so concurrent `write`/`edit` calls
on one file don't interleave.

## `edit`

Applies exact text replacements to one file.

```json
{
  "path": "src/example.py",
  "edits": [
    { "oldText": "print('hello')", "newText": "print('hello, Tau')" }
  ]
}
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `path` | yes | string | File to edit (relative to `cwd`). |
| `edits` | yes | array | One or more `{oldText, newText}` replacements. |

Each `oldText` must be non-empty, match exactly (whitespace included), appear
**exactly once**, and not overlap another edit. All edits validate before
anything is written — if any fails, the file is left unchanged. Line endings are
normalized for matching and the original dominant ending is restored. Successful
results include a diff, a unified patch, and the first changed line number.

## `bash`

Runs a shell command in the working directory.

```json
{ "command": "pytest -q", "timeout": 30 }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `command` | yes | string | Shell command to run. |
| `timeout` | no | number | Max runtime in seconds (> 0). No default. |

Combines stdout and stderr, succeeds on exit code `0`, and returns the **tail**
of large output (truncated to 2,000 lines / 50 KB; the full output is written to
a temp `.log` file whose path is included in the result). On POSIX, a timeout
kills the whole process group.

## `python_repl`

Runs Python in a kernel that stays alive for the whole session.

```json
{ "code": "rows = parse(log)\nret({'count': len(rows)})", "description": "Counting failure rows", "timeout": 30 }
```

| Argument | Required | Type | Description |
| --- | --- | --- | --- |
| `code` | yes | string | Python to run. Statements print nothing implicitly: use `print()` to show something, `ret({...})` to return data. |
| `description` | yes | string | Short summary of what the cell does. |
| `timeout` | no | number | Per-cell budget in seconds (default 30). It measures the cell's own code: time spent inside `llm_query`/`rlm_query` belongs to the host and is not charged. Overrunning kills the cell **and its namespace**. |
| `attach` | no | object | `{"name": "path"}`: bind existing files in the namespace as **paths** to open. Names must be Python identifiers; paths may be relative to the session cwd. Contents stay on disk. |
| `install` | no | array | `["numpy", "scipy==1.15"]`: add packages to this kernel's Python environment **before** the cell runs, so the cell can import them. A list of package specs, never a shell command; a request that cannot be installed fails the call before your code runs, and the resolver's own words come back. |

Names bound in one call -- variables, imports, functions, objects -- are still
bound in the next, so a 5 MB list can stay in the kernel while only the slice the
model decided matters comes back. Inside a cell:

- `ret(value)` returns JSON-compatible data next to the captured output, so a
  result does not have to be rendered into prose and re-parsed.
- `SHOW_VARS()` lists bound names with types and sizes, never their values.
- `llm_query(prompt, model=None)` and `llm_query_batched(prompts, model=None)`
  call the session's current model from inside the cell; a failed batch entry
  comes back as `None`. Sub-calls are given no tools, so a cell cannot start an
  agent, and each cell has a sub-call budget that a runaway loop cannot exceed.
- `attach({"name": "path"})` on the tool call binds a file the cell can `open()`;
  the file's bytes never enter the prompt. Handles belong to the namespace, so a
  cell after a kernel kill must restate them.
- `rlm_query(task, model=None)` and `rlm_query_batched(tasks, model=None)` hand a
  **task** to a child that solves it by writing code: a fresh kernel, up to three
  cells in which it sees what its own code printed, then an answer through `ret(...)`
  or one plain-text reply when the turns run out. Every child turn is a call on the
  same per-cell sub-call budget, and the child cannot itself recurse for real -- past
  the depth ceiling a recursion is answered by one plain completion, which the result
  metadata says out loud. Off unless `"codeModeRecursion": true` is set in
  `~/.tau/settings.json`; when off, these names do not exist in the namespace.
- `context_index()` and `context_open(refs)` read the **session's own history** from
  inside the cell. The index describes the active context in bounded rows carrying
  the same `(#N)` addresses a compaction summary cites; `context_open("#8-#19")`, or
  `"last:10"`, `"first:3"`, `"all"`, an entry id, writes those entries to a JSONL
  file and returns its path to `open()`. Entry bodies never come back as payload,
  and when the index or a slice must shrink the reply says which rows and how many
  bytes. The session binds the context -- not the tool call -- so a cell can ask
  about the transcript it is standing in without paying for the transcript.

`install` writes into a directory keyed to the session's project and then to the
kernel's interpreter (`~/.tau/kernel-site/<project>/<implementation>-<version>/`),
so an environment built in one session is still there after a kernel kill, a
compaction, or the session ending -- durability a namespace cannot offer, since a
kernel restart discards every bound name. Both keys are deliberate: an environment
built for one ABI is not handed to another, and a project's packages stay that
project's disk usage instead of one shared pile every session inherits. The project
is the repository root when the session starts inside one, so launching Tau from a
subdirectory does not start a second environment. One project and interpreter share
a 2 GiB budget; past it `install` refuses and names the directory, which is a cache
you own and can delete. Tau prefers `uv` when it is on the machine and falls back to
that interpreter's own `pip`. It never installs without being asked, and it never
removes anything: pruning is yours.

The kernel is a separate Python process with no provider credentials in its
environment and byte-bounded output. That boundary keeps model-written code from
reaching harness objects or secrets; it is not a defence against hostile code.
Set `"codeMode": false` in `~/.tau/settings.json` to start every session without
this tool, and `"codeModeRecursion": true` to add the recursion names described
above (default off, because a recursion starts another process and spends a turn
budget on it). A host that supplies its own `tools` list never receives it
automatically, and `allowed_tool_names` can drop it like any other tool.

## Choosing the right tool

- **`read`** — inspect files (instead of `cat`/`sed`).
- **`write`** — new files or complete rewrites.
- **`edit`** — precise changes to an existing file.
- **`bash`** — tests, linters, searches, project inspection.
- **`python_repl`** — data-shaped work: parsing, slicing, counting, comparing, or
  one question fanned out over many chunks. Use `bash` to run project commands,
  not to compute.
