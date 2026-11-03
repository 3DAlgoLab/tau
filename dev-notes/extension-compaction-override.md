# Extension Compaction Override (`session_before_compact`)

## What changed

Extensions can now replace Tau's model-generated compaction summary instead of
only watching it happen. Tau already emitted `compaction_start` /
`compaction_end`, but those are observation events: nothing could supply the
summary, and an extension that wanted a deterministic (zero-model) checkpoint
had no way to stop the summarization request.

Added:

- `session_before_compact` — a lifecycle hook in `LIFECYCLE_EVENT_TYPES`, with
  `SessionBeforeCompactEvent(reason, first_kept_entry_id, replaced_entry_ids,
  messages_to_summarize, tokens_before, context_window_tokens,
  custom_instructions)` and
  `SessionBeforeCompactResult(cancel, summary, details)`.
- `CompactionEntry.details` (`JSONValue`) — structured metadata from whichever
  component produced the summary. Tau's own summarizer leaves it empty.
- `ExtensionContext.entries` and `CodingSession.entries` — the active branch's
  append-only `SessionEntry` records, deep-copied.
- `ExtensionContext.context_window_tokens` — the effective window for the active
  model, already computed on the session.
- `CompactionPlan.replaced_entry_ids` — the ids behind the plan's existing
  `messages_to_summarize`, so the hook can name exactly which rows leave the
  projection.
- `ExtensionRuntime.has_session_before_compact_handlers` — sessions without a
  subscribing extension never build the hook payload and keep the prior path.

## One funnel for every trigger

Manual `/compact`, threshold compaction, overflow recovery, and truncation
recovery all plan through `_recent_preserving_compaction_plan()` and now
summarize through `CodingSession._compaction_summary_for_plan()`. Routing every
trigger through one helper is what makes the override unbypassable: the trigger
changes `reason` and `tokens_before`, never who owns the summary. A `None`
result means a handler cancelled, so the caller writes no entry at all.

## Ownership split

Tau keeps the parts that must stay authoritative:

- the boundary (`first_kept_entry_id`), the replaced set, token accounting, and
  the append-only write;
- replay: `SessionState` still rebuilds context from `summary` and the boundary,
  and never reads `details`.

The hook owns only the summary text and the canonical metadata that produced it.
This mirrors Pi's `session_before_compact`, where the harness prepares the
split-turn boundary and the extension returns `{summary, details}`.

`details` lives on the entry rather than in extension memory because Tau's state
must survive reload, resume, fork, and branch switches. A second compaction must
be able to re-render the first checkpoint's own records instead of re-summarizing
already-summarized text, and after a resume the only place those records exist is
the session file.

## Dispatch semantics

Handlers run in registration order and the first decisive one wins: `cancel=True`
or a non-empty `summary`. Remaining handlers are skipped — two checkpoint formats
in one compaction would be meaningless. A handler that returns nothing leaves
Tau's summarizer in charge.

A crashing handler is recorded as a runtime diagnostic and skipped, and Tau
continues with its own model summarization. This is the opposite of `tool_call`,
where a raising handler blocks the tool. The asymmetry is deliberate: blocking a
tool costs one call, while refusing to compact can strand a session that is
already over its window, including the overflow path that exists only to recover
from a provider rejection.

Manual compaction reports a cancellation as `ValueError("Compaction cancelled by
an extension")`; automatic paths return `False` and leave the context untouched.
An overflow compaction that a handler cancels will ask again on the next request.

## Lossless branch vs projected context

`context.transcript` is what the provider currently sees: compaction and branch
summaries folded in, older rows gone. That projection is the wrong source for an
extension that wants to recover raw history or build its own checkpoint, so
`context.entries` exposes the append-only branch instead — every message entry is
still there after compaction, along with `CompactionEntry` (with `details`) and
`CustomEntry` records. The hook event deliberately describes the projection
(`messages_to_summarize` carries an earlier checkpoint's rendered summary as the
carried base), while `context.entries` stays lossless; that pair is what lets an
extension tile spans without losing the original bytes.

## Validation

- `tests/test_session_before_compact.py` — manual override with zero provider
  summarization calls, persisted `details` plus a reopen/replay check,
  cancellation, raising-handler fallback to Tau's summarizer, and threshold
  compaction tiling where a later hook sees a shorter projection than the branch
  and the previous checkpoint as its base.
- `tests/test_extensions.py` — runtime dispatch (first decisive handler wins,
  cancel short-circuits, crash falls through, wrong-shaped result is a
  diagnostic) and the new `context.entries` / `context.context_window_tokens`
  copy semantics.

```bash
uv run pytest tests/test_session_before_compact.py tests/test_extensions.py tests/test_coding_session.py
uv run ruff check .
uv run mypy
```
