# Request-Boundary Compaction and Repeated Overflow Recovery

## What was observed

A long tool-calling run filled a 131,072-token window. The last turn ended with
`stop_reason="length"` and only a thinking block, the `stall-guard` extension retried it,
and the retry was rejected:

```text
request exceeds the context window: prompt (131064 tokens) leaves no room to
answer in the context (131072); requests are never truncated
```

Tau recovered from the *first* overflow in that session (compaction plus one retry), but
the second overflow happened inside the retry stream, which had no recovery of its own.
The run ended on the provider error and only compacted when the next prompt arrived.

Two gaps caused this:

1. Auto-compaction was checked only at prompt boundaries (`prompt()` and `continue_()`),
   so a run that chains tool calls or drains extension follow-ups could reach the hard
   limit without ever passing a check.
2. Overflow recovery retried exactly once. An overflow during that retry was terminal.

## What changed

### `tau_agent`: a portable request boundary

`AgentHarnessConfig.before_model_request` is an optional async callback, awaited by
`run_agent_loop` immediately before every provider request. The harness stays policy-free:
it exposes *when* a host may act, not *what* to do.

`AgentHarness.replace_messages` now replaces the transcript contents **in place**. A
running loop holds that list by reference, so rebinding it (the previous behavior) would
have left the loop appending to a stale list after compaction rewrote history.

### `tau_coding`: compaction policy at that boundary

`CodingSession.load` binds `_compact_context_before_request`, which runs the existing
threshold check (`_try_auto_compact(phase="auto_compact_before_request")`). The check is
cheap when the estimate is under the threshold; when it is over, the normal
recent-preserving compaction runs, and the next provider request sees the reduced context.

Overflow recovery is now a bounded loop (`MAX_OVERFLOW_RECOVERIES = 2` in `session.py`).
If a continued run overflows again, Tau compacts again and retries, up to that budget.
`AutoRetryStartEvent.attempt` / `AutoRetryEndEvent.attempt` report the current attempt,
`max_attempts` is the bound, and `SessionAgentEndEvent.will_retry` is true whenever a
further recovery is pending.

## Why the split belongs at these layers

- `tau_agent` owns the reusable loop and its event/request boundaries.
- `tau_coding` owns context policy, thresholds, summaries, and persistence.

No TUI, provider, or CLI assumption moved into the harness; the hook is optional, and
without it the loop behaves exactly as before.

## Behavior changes

- Automatic compaction can now happen **between turns of one run**, before a request,
  rather than only after the previous turn. Two existing tests that scripted
  "answer, summary" now script "summary, answer"; their assertions still verify the same
  threshold and the same active context.
- A retry that overflows again is no longer terminal.
- Mid-run compaction does not emit `CompactionStartEvent`/`CompactionEndEvent`: it happens
  inside the request boundary rather than a session turn, so it is silent in the transcript
  (the next context-usage refresh reflects it). Overflow recovery still emits its events.

## Limitations

- `CodingSession.continue_()` still has no overflow recovery of its own. It has no
  in-repo callers today; the request-boundary hook does run during it, so it compacts
  proactively but does not recover from a provider 400.
- The mid-run check uses Tau's estimate. Provider-anchored usage keeps it accurate, and the
  compaction reserve (16,384 tokens by default) absorbs tokenizer differences, but a
  provider that silently truncates instead of rejecting needed its own recovery, which is
  the next section.

## Second phase: the truncated turn that showed nothing

The same session failed again a few minutes later, one step earlier in the chain. The
transcript (`~/.tau/sessions/home-dev-pi-tau-89daca/d7a7cbc6459b4dffb7aaa911a7b98d7c.jsonl`)
records it exactly:

| entry | time | what happened |
| --- | --- | --- |
| 308 | 03:41:36 | assistant turn, `stop_reason="length"`, 131,064 of 131,072 tokens, 922 output tokens, content: one thinking block. No text, no tool call. |
| 309 | 03:43:40 | automatic compaction, `tokens_before=130,132`, own usage 76,644. |
| 310 | 03:50:42 | the user typed `continue …`. |

Nothing was broken, and that was the problem. The provider stopped the response at the
context limit instead of rejecting the request, so no overflow error reached the recovery
loop. The turn requested no tool call, so `run_agent_loop` ended the run. Auto-compaction
then ran from `prompt()`'s after-turn check and succeeded - but it ran *after* the run had
already settled, and no code path resumes a settled run. The session sat idle for nine
minutes.

The guard was not at fault; it had already been taught to leave `length` stops alone, and
retrying one at 131,064 of 131,072 tokens is exactly what produced the 400 in the first
section. The guard could not fix this even in principle: an extension cannot ask for
compaction.

### What changed

`CodingSession` now treats "this run settled on a truncated turn that produced nothing
visible" as recoverable:

- `_run_ended_on_empty_truncation()` returns true when the assistant message that ended a
  run has `stop_reason == "length"`, no tool call, and no text. Thinking is not output the
  user can read or act on, so a thinking-only truncation counts as silence.
- `_recover_from_truncated_turn()` compacts first when the estimate exceeds the
  auto-compaction threshold, then issues one `continue_()`. When the window is not the
  constraint the truncation came from the model's own output cap, so the continue runs by
  itself and no summarization request is spent.
- Both `prompt()` and `continue_()` remember the last `length` assistant of the run and run
  the recovery before the existing after-turn auto-compaction, so the room exists before
  the next turn rather than after the fact.
- `MAX_TRUNCATION_RECOVERIES = 1`. A second truncation reports
  `AutoRetryEndEvent(success=False, final_error="Response truncated at the model output
  limit again")` instead of looping.

The recovery emits `CompactionStartEvent`/`CompactionEndEvent` with
`reason="threshold"` plus the `AutoRetryStartEvent`/`AutoRetryEndEvent` pair, which is what
the overflow path already emits, so the TUI and extensions need no new handling. No new
`CompactionReason` literal was introduced: the trigger was a threshold that had been
exceeded, just observed from a different signal.

`stall_guard` keeps its half: on a `length` stop it notifies
`"stall-guard: response truncated at the token limit; not retrying (Tau will compact
context instead)"` and queues nothing. The division is deliberate - the extension can see
the stop reason and speak up, but only the session can make room.

### Why both halves have to be core-side

Retrying a truncated turn is only correct when there is room for the retry. Creating room
means compaction, compaction means the session's summary and persistence machinery, and
resuming a settled run means starting a new run from inside the one that just ended. That
is session policy, so it lives in `tau_coding`. The harness stays policy-free: the recovery
is built from `harness.continue_()`, the same primitive the TUI uses for a manual continue.

### Behavior changes

- A run that ends on a silent truncation now keeps going instead of settling silently:
  expect a compaction line and a retry line in the transcript, then the resumed answer.
- Recovery is skipped when the truncated turn produced readable text. A cut-off answer is
  the user's decision to continue; resuming it automatically would double the cost of every
  verbose reply.
- Recovery stands down when compaction cannot produce a plan (nearly everything sits in the
  retained tail). Continuing into a full window would only earn a rejection.

### Limitations

- One recovery attempt per run. A model that keeps truncating needs a smaller task, a
  larger window, or `/compact` with a focus - a decision for the user.
- The compaction check reuses the estimated context. Right after a compaction the estimate
  comes from the rewritten prefix rather than the pre-compaction provider number, so a
  second compaction is not triggered by a stale anchor.
- A `length` turn that carries a tool call is left alone: the loop already continues that
  run to execute the call.

## Third phase: the subagent whose recovery answered itself with a 400

The same chain failed a third time, in a `tau-subagents` child session. The child
is built with auto-compaction disabled, so `auto_compact_token_threshold` is
`None`, and the transcript grew to 130,581 of 131,072 tokens unimpeded. A turn
truncated at the limit with a thinking block only, recovery ran, and skipped the
compaction branch - `threshold is None` read as "the window is not the
constraint". The continue re-sent the same over-window prompt, now with the
truncated assistant message appended: 131,074 tokens, rejected with a 400, and
the single-attempt budget made that terminal. The `agent-calls.jsonl` diagnostic
records `phase: "truncation_recovery"` on the rejected request.

Two gaps, both inside `_recover_from_truncated_turn`:

1. "No proactive threshold" was conflated with "no window". Disabling
   auto-compaction removes the *request* to compact; it does not remove the
   context window. A full-window truncation is the constraint announcing itself
   regardless of what policy the session runs.
2. A retry rejected as over-window had no recovery, mirroring exactly the gap the
   main prompt path closed with `MAX_OVERFLOW_RECOVERIES`.

### What changed

- When the effective auto-compaction threshold is `None`, recovery falls back to
  the window-derived one (`auto_compaction_threshold_for_context_window`). A
  subagent with auto-compaction off now compacts on a full-window truncation
  before continuing, instead of blindly re-sending the oversized prompt.
- The single continue is now a bounded loop (`MAX_TRUNCATION_RECOVERIES = 2`).
  If the retry is rejected as over-window (`is_context_overflow_error`), one
  overflow compaction and one more continue repair it, emitting the same
  `CompactionStartEvent`/`CompactionEndEvent` with `reason="overflow"` the main
  prompt path uses. `AutoRetryEndEvent.attempt` can now be 2.
- A second *truncation* still ends the run: the extra budget is for rejections,
  not for a model that keeps overflowing its own output cap. That distinction is
  why the loop counts attempts but only the overflow branch spends a second.

### Behavior changes

- Subagent sessions (auto-compaction off) no longer end on a provider 400 after
  a full-window truncation; expect a compaction line and a retry line instead.
- A truncation retry that is rejected as over-window compacts and continues once
  more instead of reporting the provider error as the final failure.

### Validation

New coverage:

- `tests/test_coding_session.py::test_truncation_recovery_falls_back_to_the_window_when_auto_compaction_is_off`
- `tests/test_coding_session.py::test_truncation_recovery_compacts_when_the_retry_is_rejected_as_overflow`

## Fourth phase: the session that died with its recovery budget spent

The third phase fixed a child session. The parent died anyway.
`~/.tau/sessions/home-my-vault-20-ai_and_agent-bab571/96beebd123074b188f01a2e02d22bc43.jsonl`
(897 lines) ends mid-sentence with a provider rejection and nothing after it:

| line | time | what happened |
| --- | --- | --- |
| 625 | 03:11:43 | 400 at 131,239 tokens; line 626 compacts (`reason="overflow"`, `tokens_before=131,199`) |
| 785 | 05:42:23 | 400 at 131,131; line 786 compacts (`reason="overflow"`, `tokens_before=131,165`) — the second and last `MAX_OVERFLOW_RECOVERIES` |
| 869 | 06:51:55 | the provider reports 115,511 input tokens; the auto-compaction threshold for a 131,072 window is 114,688, so this is already over |
| 891, 893, 895 | 07:15-07:16 | three turns end `stop_reason="length"` at 131,064 of 131,072, each carrying readable text (451, 267, 86 output tokens) |
| 897 | 07:17:57 | 400 at 131,161. No compaction entry follows. `index.jsonl`'s `updated_at` is this timestamp |

Both recoveries had been spent hours earlier, so the third rejection had nothing
left to spend. Two things had to be true for the session to reach it, and both turned
out to be defects.

### The proactive guard went blind at exactly the moment it mattered

Between line 869 (over threshold) and line 897 there are 28 provider calls, every one
of which passes `run_agent_loop`'s request-boundary check, and none of which compacts.
The check compares `context_token_estimate` with the threshold, and the estimate had
fallen to 98,767 while the provider reported 115,511.

The cause is in `tau_agent/session/memory.py`. On re-read, the compaction summary row is
*synthesized* — and it was synthesized as a `UserMessage(content=...)` with no
timestamp, so it took the wall clock of the replay. A reloaded session therefore holds
its summary newer than every message that summary covers.
`_last_applicable_provider_usage` requires the anchor to sit at or after the newest
boundary, so on replay it rejects every recorded usage, the provider anchor dies, and
the estimate falls back to chars/4:

| compaction | `tokens_before` recorded live | the same state after a reload |
| --- | --- | --- |
| line 313 | 115,711 | 83,991 |
| line 449 | 116,041 | 83,372 |
| line 626 | 131,199 | 97,349 |
| line 786 | 131,165 | 98,813 |

At 03:06 the session was resumed and its first request-boundary check compared 97,349
against 114,688. The provider was at 119,194. Custom messages already take their
timestamp from the entry (`round(entry.timestamp * 1000)`); the compaction and
branch-summary rows were the two rows that did not.

### A full window was not allowed to be the constraint

The three truncations at 131,064 carried text, and `_run_ended_on_empty_truncation`
recovered only silent ones — the phase-two rule, reasoning first. So the model answered
451 tokens, got cut off, the run settled, the same context was sent again, and it
happened three times before the provider rejected the request. That rule is correct when
the model ran out of *output* room. At 131,064 of 131,072 the window ran out, and
re-sending the same prompt only refills the same buffer.

### What changed

- `tau_agent/session/memory.py`: synthesized compaction-summary and branch-summary rows
carry the entry's timestamp. Replay of one branch is now one timeline, and the provider
anchor survives a reload.
- `tau_coding/context_window.py`: `provider_prompt_floor_tokens(messages)` returns the
largest input side any provider response measured since the newest summarization
boundary. It is a lower bound, never a guess, and it survives an unusable anchor —
including a zero-usage summarization response, which is what a stubbed or offline
summarizer records. `estimate_context_usage` returns the larger of its two answers, so an
anchor that cannot be used still contributes its measurement; when it can be used, that
number is already inside it.
- `tau_coding/session.py`: `_context_pressure_tokens()` is the one number the threshold
check, the truncation-recovery check, a manual compaction's `tokens_before`, and the
`session_before_compact` event all use. A handler deciding whether this compaction is the
last one its session will ever get needs the count the provider itself reported.
- `_run_ended_on_empty_truncation` became `_run_ended_on_recoverable_truncation`, which
takes the window: text-carrying truncations are recoverable when the measured context is
over the threshold, and still left to the user when it is not.
- `transparent-compact` (separate repository) can no longer refuse a window-critical
compaction, and records every decision it makes on the hook, so a declined compaction is
a session entry rather than a toast.

### Behavior changes

- A reload no longer resets the estimate. A session resumed after a compaction now
  auto-compacts at the threshold the provider has already crossed instead of running ~20k
  tokens past it.
- A truncated turn that carried text compacts and continues when the window is the
  constraint; a truncated turn under no window pressure still ends the run for the user
  to continue.
- `estimate_context_usage` can return a provider-reported number where it previously
  returned a character estimate, so a session whose summarizer records no usage can still
  auto-compact on the next turn.

### Limitations

- A floor is a floor: it says nothing about what was appended after the response that
  measured it, so it can under-count. It cannot over-count, which is why it triggers
  compaction rather than deciding how much to keep.
- The third rejection in the observed session was still fatal. The fixes above would have
  compacted at line 870; they do not add a third overflow recovery. A session that spends
  its whole budget before anything is wrong has a different problem than this one had.

### Validation

New coverage:

- `tests/test_compaction_replay_accounting.py` — the summary row is stamped from its
  entry, replaying a branch twice is identical, the anchor survives a reload, and a
  zero-usage boundary still leaves a floor.
- `tests/test_context_window.py` — floor tests, including that a boundary with no usage
  does not discard earlier measurements.
- `tests/test_coding_session.py::test_session_recovers_a_truncated_text_turn_at_a_full_window`

Run:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Validation

New coverage:

- `tests/test_agent_harness.py::test_before_model_request_runs_before_every_provider_call`
- `tests/test_coding_session.py::test_session_compacts_between_requests_of_one_run`
- `tests/test_coding_session.py::test_session_recovers_when_the_overflow_retry_also_overflows`
- `tests/test_coding_session.py::test_session_compacts_mid_run_on_provider_reported_usage`
- `tests/test_coding_session.py::test_session_compacts_and_continues_after_an_empty_truncated_turn`
- `tests/test_coding_session.py::test_session_continues_an_empty_truncated_turn_without_compacting`
- `tests/test_coding_session.py::test_session_does_not_recover_a_truncated_turn_that_produced_text`
- `tests/test_coding_session.py::test_truncation_recovery_stops_after_one_attempt`
- `tests/test_coding_session.py::test_truncation_recovery_stands_down_when_compaction_cannot_help`
- `tests/test_stall_guard_extension.py::test_length_stop_is_never_rescued`
- `tests/test_stall_guard_extension.py::test_truncated_turn_warns_and_leaves_the_continue_to_core`

Updated for the new timing:

- `test_session_auto_compacts_after_response_when_threshold_is_exceeded`
- `test_session_auto_compacts_with_pi_style_default_threshold`

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
