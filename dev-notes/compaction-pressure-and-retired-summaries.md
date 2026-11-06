# A Compaction That Priced and Stacked Its Own Replaced Prefix

## What was observed

A session on 2026-11-05/06 (`home-dev-pi-harness2-fce4b6`,
`4b94e0f8985b4e3da43261538143c101.jsonl`, 131,072-token window, auto threshold
114,688) compacted correctly and then paid twice more for summarization that bought
nothing. Read from the file:

| Row | Time | Event | Measured |
| --- | --- | --- | --- |
| 198 | 12:13:01 | last request before the first fold | provider usage `input 400 + cacheRead 114,809` = 115,209 — real pressure |
| 199 | 12:13:02 | mechanical checkpoint A | 54,445 chars, boundary `b41a2c22` (row 164) |
| 200 | 12:15:48 | second `session_before_compact` call | `ValueError: no uncompacted entries inside the replaced span`, `tokens_before = 115209` |
| 201 | 12:18:16 | compaction B, written by Tau's summarizer | 17,094 chars, 24,045 tokens, **same boundary** `b41a2c22` |
| 203 | 12:18:49 | first request after B | provider-measured prompt **51,476** |

So B was asked to summarize a context that the provider itself put at 51,476 tokens —
under half the threshold — and the number that justified it, 115,209, was the prompt of
row 198, a request made before A existed. The projection B produced was *larger* than
the one it replaced: A's row survived B (54,445 characters of checkpoint text still sit
in that session's live context today, ~18k dead tokens against a real prompt of 39,006).

Two defects, one each on the two sides of that exchange.

## Defect 1: the provider floor never reset at a compaction

`provider_prompt_floor_tokens` (`tau_coding/context_window.py`) scanned the message list
in reverse and **broke at the newest compaction summary row**. Replay *prepends* that row
(`_apply_compaction` returns `[summary_row, *retained]`), so every retained response sits
*after* it, and the first one found handed back the prompt it had been priced for before
the fold. A floor is a maximum, so once pinned it never came back down: from 12:13 on,
every pressure check in that session reported 115,209 against a threshold of 114,688.

That number did the damage twice over. It triggered the compaction, and it disarmed the
safety hatch on the other side: the `transparent-compact` extension had already reported
that the offered span held nothing uncompacted, and its own policy — refuse below the
window, stand down at it — chose to stand down because `_is_window_critical(115,209)`
said the window was critical. It was not.

The rule was already written down one function away. `_last_applicable_provider_usage`
rejects a usage block older than the newest prefix boundary by timestamp; the floor
compared positions instead. Bound it the same way — a response counts as a floor only if
it is dated after the newest summary row, which replay stamps from its own entry — and
the phantom trigger cannot form. Right after a fold there is no floor at all until the
first response measured since it, which is the same honest state a reload is in.

## Defect 2: the replaced rows were never recorded, so summaries stacked

`CompactionPlan.replaced_entry_ids` was added so a hook could see what a fold covers, and
the plan's *written* entry kept only `first_kept_entry_id`. A kept boundary cannot
describe this fold: the second fold started inside the window the first one kept, so the
row it supersedes — A — sits **after** `first_kept`. `_apply_compaction` replays from the
boundary and keeps A, then puts B in front of it. The context grew at every compaction,
and the next fold had a summary of a summary to work on.

`CompactionEntry.replaces_entry_ids` already existed and already drove exact replacement
in `_apply_compaction`; it was simply left empty by every new write, with a comment
calling it a legacy shape. It is now filled from the plan (filtered to rows still in the
active context), and replay drops the summary this fold replaces. Files that carry only a
boundary keep replaying from the entry tree, unchanged.

That also makes the hook's own replacement set meaningful after the fact: a later fold —
Tau's or an extension's — can now retire an extension checkpoint instead of stacking
underneath it.

## What is still open: one compaction written onto a dead branch

The same file records a third event that these fixes do not explain. At 13:21:00 a
compaction was requested with `tokens_before = 115,048`, which *was* real pressure (row
280's `input 528 + cacheRead 114,520`). The extension declined it, Tau's summarizer ran
for 1m57s and cost 56,240 tokens, and the entry landed at row 283 with:

- `parent_id = c7d7f124…` — row 153, an entry from 00:24;
- `first_kept_entry_id = c57d11ee…` — row 130, an entry the live context no longer held;
- the next live message (row 284) parented to row 282, not to the compaction above it.

So the object that wrote that compaction held a replayed state pointing at the 00:24
branch while the pressure number it acted on came from a message list that ran to 13:21,
and the session that carried on did not know entry 283 existed. The entry is off the live
branch: it changed nothing the model ever saw.

Not proven, and deliberately not "fixed" by a guess. What the code offers as candidates,
in the order they are worth reading: `_navigate` (`session.py`) refreshing state and
leaving `_harness.messages` alone when `_persist_active_tool_history_repairs()` returns a
repair; the session-replacement adoption path that copies `_state`, `_harness` and
`_last_parent_id` from another runtime; and a second runtime holding the same file open.
The last one fits the parent chain best — the writer's tip was 153 while the file's tip
was 281, and the next append came from a writer whose tip was 282 — and no entry shape in
the file distinguishes a stale writer from an intentional branch navigation. A writer
token on each entry would. That is the change to make if this recurs; it is not one to
make from one transcript.

## Validation

New coverage:

- `tests/test_context_window.py` — the floor ignores a replayed summary row at the front
  of the list, keeps only measurements taken since the newest of several summaries, and
  reports no floor rather than a stale one.
- `tests/test_coding_session.py::test_a_second_compaction_replaces_the_summary_it_folds` —
  two folds in a row, then a reload: one summary in the rebuilt context, and the second
  entry records the first as replaced.

Updated for the new storage shape:

- `test_session_compact_persists_summary_and_rebuilds_context` (the field is now written)
- `test_session_auto_compacts_after_response_when_threshold_is_exceeded`
- `test_persist_after_branch_keeps_state_on_active_branch` (the set names live-branch rows
  only, and never the abandoned ones)
- `tests/test_rpc.py::test_rpc_compaction_returns_canonical_summary_and_boundary`

Run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

Note: `tests/test_tui_app.py::test_tui_tree_labels_filter_timestamps_and_clear` renders a
local date and fails outside UTC on this machine; it passes with `TZ=UTC` and is unrelated
to this change.
