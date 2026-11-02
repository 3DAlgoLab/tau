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
