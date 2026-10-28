---
title: "Deterministic Compaction Folds And Renderer"
---

Phase 1b of porting pi-fabric's deterministic compaction. It adds the part that
turns the normalized event stream from
[Phase 1a](./compaction-engine-core.md) into the text a model actually reads.
Still nothing calls it: the `before_compact` seam is the only way in, and Tau's
default LLM compaction is unchanged.

```text
src/tau_coding/compaction/dialogue.py      protected recent request/response pairs
src/tau_coding/compaction/projections.py   sampled folds over the event stream
src/tau_coding/compaction/render.py        byte allocation and final assembly
```

## Two classes of content

The renderer distinguishes content that must survive from content that may be
sampled, and that distinction drives the whole budget:

| Class | Sections | Drop rule |
| --- | --- | --- |
| protected | dialogue, explicit compaction request, carry-forward focus | never drops an item; only item text is clipped, and the clipped byte count is reported |
| historical | requests, files, activity, outstanding, earlier turns, status, transcript | drops whole lines with an addressed marker, then header only |

Protected blocks render first because their size is known exactly, then the footer
is allocated, then the historical sections share what is left. Order matters: the
expansion pointer is never the thing that gets dropped to fit. If allocation ever
fails, the final defensive clip cuts the tail, which is where the sampled
sections live, not the commitments.

## dialogue.py

Exchanges are **request-first**: a user instruction anchors the exchange, and the
assistant prose that follows it becomes the second excerpt. pi-fabric anchors on
the newest assistant prose and walks backwards for a user message; that can attach
an older reply to an unrelated later instruction. Tau pairs forward instead, so a
response is only ever shown as the response to the request that preceded it.

- Only the last `MAX_RECENT_EXCHANGES` (3) exchanges survive.
- One assistant entry holding several text blocks is one response.
- Assistant prose with no request to answer is dropped as unattributed.
- `MAX_EXCHANGE_BYTES` (3968) is the *text* budget; the two headings and the
  newline take the remaining 384 bytes of `MAX_DIALOGUE_BYTES` (12288), so the
  ceiling is never below `3 * (3968 + 44)`. A test pins that relationship so the
  two files cannot drift apart.
- Content already kept verbatim in the retained tail is **address-only**: the
  heading says `retained raw` and the body is empty. Spending summary bytes to
  duplicate text the model can still read is pure waste, and dropping the pair
  entirely would leave no record that the reply existed.

## projections.py

Seven sampled folds, each with its own cap and its own addressed omission marker:

- **requests** -- every user turn, minus the ones the dialogue fold already keeps.
- **files** -- grouped as `Written:` / `Modified:` / `Read:` under a collapsed
  common root, with the entry id of the call that touched each path. A read of a
  file that was also modified is not listed twice. There is no `Created:` label:
  proving creation would mean reading tool result prose, and prose classification
  is exactly what this engine refuses to do.
- **activity** -- non-file operations, mainly shell, with outcome.
- **outstanding** -- errors, marked `[RESOLVED]` only when a later success shares
  the same typed identity: `file + tool + path`, `bash + tool + command`, or a
  canonical JSON signature for everything else. Identity matching is why a retry
  of the same command resolves the earlier failure while an edit to a different
  file does not.
- **earlier_turns** -- one addressed line per turn with a tool histogram. The
  histogram counts one per execution: if the stream carries both a call and its
  result, only the call counts.
- **status** -- last request, last change, last command and outcome, last
  assistant note, to bridge the summarized window into the kept tail.
- **transcript** -- the last events with stable `(#N)` event-index pointers, so a
  reader can say "the third thing in the transcript" and expand it.

Prior compaction and branch summaries are carried as pointers, never re-expanded.
Re-inflating a previous summary's prose would launder old guesses into current
facts and re-ingest content the last compaction already paid to drop.

## render.py

Section ceilings live in one table; the global `MAX_SUMMARY_BYTES` (32 KiB) is the
hard ceiling. `_bounded_history` walks the number of kept lines down by the
shortest overage -- then head-and-tail sampling with an addressed marker -- then
header only, so a section shrinks proportionally instead of one section silently
eating another's room. `MAX_INPUT_LINES_PER_SECTION` (128) caps allocations before
the byte math, and `MAX_RENDERED_LINE_BYTES` (1024) clips a single runaway line
without ever deleting it.

The footer names the timestamp and the cumulative source range (`first -> last`
entry ids) plus the pointer line explaining that the pre-summary entries are still
in the append-only session log and `/tree` can branch from any of them. Tau has no
`memory.expand` tool, so the pointer points at the mechanism Tau actually has.

## What is not here yet

- QA probes generated from ground truth (Phase 1c).
- A `carry-forward` entry type: the renderer already has the block, the entry
  model does not exist yet.
- Wiring, provenance in `CompactionEntry.details`, and any UI affordance (Phase
  1d as an extension).

## How to test

```bash
uv run pytest tests/test_compaction_dialogue.py tests/test_compaction_projections.py tests/test_compaction_render.py
```
