---
title: "Session history as handles inside python_repl"
date: "2026-10-29"
---

# Session history as handles inside `python_repl`

Phase 2c-i gave a cell files as paths. This gives it the session's own history the
same way: `context_index()` describes the active context in bounded rows, and
`context_open(refs)` writes the addressed entries to a JSONL file and returns its
path. Nothing about the transcript travels as a payload.

## Why not as tool arguments

Upstream CodeAct passes history inwards as `context_N` arguments, which means the
caller -- the model -- has to hold the history to pass it. That is the cost the
feature exists to avoid. Here the arguments stay typed (`refs` are `"#N"`,
`"A-B"`, `"last:N"`, `"first:N"`, `"all"`, or an entry id, checked by the host
before any source is consulted) and the *data* enters through a file. So the tool
schema gains nothing: `context_index`/`context_open` are kernel bindings, listed
by `HELP()` like `llm_query`.

## Addresses are the ones compaction already cites

The 1-based `(#N)` a compaction summary cites is the addressing scheme here too,
and the index rows carry `entry_id` for exactness. A summary that says "3 of 31
events [omitted]" now has a follow-up move: `context_open("#8-#19")`. One currency
for omission markers and for handles.

The rows come from a *live* view of the active branch (`_active_context_rows`), not
a snapshot taken at construction, because the context worth searching is the one the
next model call will see -- after a compaction or a branch switch. The test that
replaces the rows mid-session pins that.

## Budgets, and who enforces them

Two independent layers, both conservative:

- Policy (`tau_coding/context_entries.py`): index capped at `MAX_INDEX_ROWS` by
  `sample_addressed` (head + tail, middle counted and addressed); the slice file is
  bounded by `MAX_SLICE_BYTES`, solved jointly with `MAX_SLICE_ROWS` so that a
  slice of many long entries drops whole rows -- accounted, chronological -- and
  clips the survivors to one shared row budget. Each row carries its own
  `omitted_bytes`; the reply carries `per_row_bytes`, `omitted`, and a note naming
  the omitted entry range via `omission_line`. Nothing shrinks silently.
- Mechanism (`kernel/process.py`): whatever the source returns must still fit one
  frame, so an oversized index is trimmed from the tail with the drop folded into
  `omitted` and a note. Refs are refused before the source sees them (count,
  emptiness, type), and an unwired context answers with a sentence rather than a
  hang.

The samplers and the byte arithmetic are the Phase 1 compaction primitives
(`sample_addressed`, `bounded_excerpt`, `clip_utf8`, `omission_line`), not a second
implementation of the same idea -- which is also why the row budget is in UTF-8
bytes: the number has to mean the same thing at every layer that counts it.

## Lifecycle

`ContextEntries` materializes into a `tau-context-*` scratch directory created on
first use and removed by `CodeMode.dispose()`. A kernel that never asks for history
never creates a file; a session that closes never leaks one. Binding follows the
`bind_subcall` rule: before the first cell, because the kernel captures the source
when it starts.

## What is deliberately not here

No summarization, no retrieval ranking, no prose classifier. The cell decides what
matters after it can grep the file; the host's only job is to describe the context
honestly and stay inside its ceilings. Recursive `rlm_query` with a depth guard is
the next layer, and it should reuse these handles rather than inventing another way
to name history.
