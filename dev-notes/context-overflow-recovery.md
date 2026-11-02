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
  provider that silently truncates instead of rejecting could still behave differently.

## Validation

New coverage:

- `tests/test_agent_harness.py::test_before_model_request_runs_before_every_provider_call`
- `tests/test_coding_session.py::test_session_compacts_between_requests_of_one_run`
- `tests/test_coding_session.py::test_session_recovers_when_the_overflow_retry_also_overflows`

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
