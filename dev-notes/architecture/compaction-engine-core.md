---
title: "Deterministic Compaction Core (bounds, events, cut)"
---

Phase 1a of porting pi-fabric's deterministic compaction. It adds the three
primitives a provider-free compactor needs. Nothing calls them yet: the
`before_compact` seam from [Phase 0](./compaction-hook-seam.md) stays the only
way in, and Tau's default LLM compaction is unchanged.

```text
src/tau_coding/compaction/bounds.py   UTF-8 byte budgets and addressed sampling
src/tau_coding/compaction/events.py   typed event normalization of one turn
src/tau_coding/compaction/cut.py      closure-safe cut selection and token ceilings
```

## Why this exists

Tau's built-in compaction asks a model to write the summary. That costs a
provider round trip, is non-deterministic, and re-ingests the very content it is
meant to drop. pi-fabric's engine instead derives the summary from structured
facts already in the session: roles, block types, tool names, JSON arguments,
error flags, exit codes. Prose is carried, never classified -- there is no
keyword or language heuristic anywhere, which is what makes the port safe for
English, Korean, and mixed sessions.

## bounds.py

Every summary section is budgeted in **bytes**, because a byte ceiling is
computable before a request is built and is what `MAX_SUMMARY_BYTES` (32 KiB)
enforces downstream. `clip_utf8` backs off to the last whole codepoint so a
never-truncated field can never emit half a character; `bounded_excerpt` keeps a
head and tail and returns the exact omitted byte count, and its head/tail
allocations are scaled by each side's share so a one-sided field cannot waste
budget. `sample_addressed_from` samples a stream as head plus tail and reports
how many items and which entry range were omitted.

Sampling is conservative by construction: `kept_count == len(values)` and
`omitted_count == len(values) - len(sample.values)` are asserted in tests, so a
summary can never silently drop items while claiming conservation.

## events.py

`normalize_rows` turns one turn's context rows -- `(entry_id, AgentMessage)`
pairs, the same shape Tau's context builders produce -- into an ordered typed
event list: `UserEvent`, `AssistantTextEvent`, `PriorSummaryEvent`,
`CustomMessageEvent`, `ToolCallEvent`, `ToolResultEvent`, `BashEvent`. Each event
carries its 1-based ordinal and the session entry id it came from, and every
omission downstream cites those ids.

Three rules:

- **Thinking is erased, never summarized.** Thinking blocks are stripped and
  counted by `count_erased_thinking_blocks` so the summary can state that
  reasoning was dropped without reproducing it.
- **Bash pairs are re-joined.** A `bash` tool result is linked back to its call by
  `tool_call_id` so the command and its outcome are one fact; Tau's native
  `BashExecutionMessage` already carries both.
- **Extension payloads are bounded.** `CustomMessage.details` is copied under node,
  depth, item, and byte ceilings; anything over a ceiling is dropped whole rather
  than partially truncated, so no projection can render half a payload with no
  marker saying what is missing.

## cut.py

`compute_cut` picks the boundary that separates "summarize" from "keep", and
provider validity outranks token arithmetic:

- `call_result_spans` groups every `toolCall`/`toolResult` pair by call id,
  including a result whose call was already compacted away.
- `closure_safe` rejects any boundary that splits a pair, and `_is_cut_point`
  rejects landing on a `toolResult`. A cut that orphaned a result would make the
  next request invalid rather than merely large, which is exactly what Pi's
  `postprocess.ts` repairs after the fact; Tau refuses to create it.
- A whole turn is preferred when it fits, so the kept tail never starts
  mid-exchange. Otherwise the earliest boundary whose raw suffix fits the tail
  budget wins.
- If nothing fits -- one tool result larger than the entire tail budget -- the
  latest closure-safe boundary is used instead of returning no cut. Shrinking is
  still correct; overflowing the window is not.

Tau estimates tokens over message bodies, while providers bill the system prompt,
tool schemas, and serialization too. `_calibrate_tokens` therefore fits an upper
affine envelope `context ~= fixed + scale * raw` from assistant `Usage`
checkpoints, taking the maximum slope so the envelope stays above every
observation. That yields four ceilings -- continuity, occupancy, a 0.9x safety
margin under the hard ceiling, and a 0.95x reduction floor -- and the cut records
which one bound it. Without a provider-anchored `tokens_before`, the estimator's
own reading is used, so calibration degrades to `scale = 1` rather than guessing.

## What is not here yet

- Rendering: the bounded section renderer and projections (dialogue, requests,
  files, errors, earlier turns, current status, transcript pointers).
- QA: structural probes generated from ground-truth events, not from rendered
  text.
- Wiring: a built-in engine selector, and filling `CompactionEntry.details` with
  engine provenance for expand-on-demand.

## How to test

```bash
uv run pytest tests/test_compaction_bounds.py tests/test_compaction_events.py tests/test_compaction_cut.py
```
