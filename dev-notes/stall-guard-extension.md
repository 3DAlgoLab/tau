---
title: "Stall-guard example extension"
---

Tau now ships `examples/extensions/stall_guard.py`, an extension that recovers a
model turn that ended with no tool call. It ports the intent of the Pi extension
`stall-guard.ts` onto Tau's extension API, and in doing so documents an important
property of Tau's event dispatch: extension handlers run inline, so a handler can
still change what the current run does next.

## Problem

An OpenAI-compatible endpoint can answer with `finish_reason: "stop"` while the
tool-call markup sits inside `content` text instead of the structured
`tool_calls` array. Tau's provider layer has nothing to parse, the loop sees an
assistant message with no calls, and the run settles silently: no tool ran, no
error surfaced, and a long unattended job simply stops. Two sibling shapes settle
the same way:

- `stop_reason: "toolUse"` with no `toolCall` block (`src/tau_ai/stream.py` maps
  a provider `tool_calls` finish reason to `toolUse` even when no call parsed),
  and
- an assistant message carrying only `thinking` content.

Tau's automatic retry covers provider errors (`stop_reason` `error`), so none of
these are retried.

## Where the guard hooks

Pi exposes an `agent_before_settle` decision hook that can inject an entry and
request one more model call. Tau has no settlement hook; `agent_settled` is
observation-only and fires after the run generator's teardown.

The Tau boundary is the `turn_end` of a turn whose assistant message requested no
tool call:

```text
run_agent_loop                    emits TurnEndEvent for the tool-call-free turn
        |
AgentHarness._run                 awaits its listeners, then yields the event
        |
ExtensionRuntime._on_agent_event  dispatches to the extension turn_end handler
        |                         (session.is_running is still True, so a queued
        |                          follow-up lands in the harness queue)
run_agent_loop                    drains the follow-up queue and loops again
```

The harness awaits `_notify(event)` before yielding, so the queued message is
already in `_follow_up_queue` when `run_agent_loop` reaches its
`get_follow_up_messages()` check. The run makes exactly one more model request
instead of settling. This path is queue-based rather than UI-based, so it behaves
the same in the TUI, in print mode, and in tests; it needs no host
`turn_requested` callback.

The guard inspects `turn_end` rather than `message_end` because only `turn_end`
carries whether the turn produced tool results.

## Detection

A turn is rescued only when its assistant message requested no tool call, the
turn was not `error` or `aborted` (Tau retries those itself), no queued input was
already pending, and one of these shapes matches:

| Shape | Signal |
|---|---|
| leaked markup | tool-call grammars (`invoke`, `parameter`, `think`, `im_start`, `<\|...|>`) in prose, after fenced blocks and inline spans are stripped |
| phantom call | `stop_reason == "toolUse"` with no `toolCall` block |
| thinking only | no text, at least one `thinking` block |

Two false-positive guards carried over from Pi:

- markup quoted inside a fenced block or inline span is documentation, not a
  leaked call, so it is stripped before matching;
- a message that leaked markup *and* carried more than `PROSE_OVERRIDE_CHARS`
  (600) characters of real prose is treated as a deliberate answer. The prose
  override cancels only the leaked-markup signal; a phantom or thinking-only
  message is still rescued.

## Budget and queue awareness

`MAX_PER_RUN = 2` and `MAX_PER_SESSION = 8` bound the rescue; the per-run counter
resets on `agent_start`, and both counters reset on `/stall-guard reset`. The
budget lives in the extension's load generation, so `/reload`, `/new`, and
`/resume` into a fresh generation start it over.

Pi skips its rescue when `event.context.pendingMessages` is non-empty, because a
queued message already resumes the run. Tau exposes no queue read to extensions,
and emits `queue_update` only when a message is queued, never when the loop
drains one. The extension reproduces the check with two documented events: it
counts pending steering plus follow-up entries from `queue_update`, and clears
that count when a `user` or `custom` `message_start` shows the queued message
being drained into the run.

## Presentation

The injection uses `send_custom_message` with `custom_type="stall-guard:retry"`,
so the model-facing correction text enters context while the transcript renders
one compact line through a registered message renderer (full text only when the
row is expanded). `tau.register_command("stall-guard", ...)` reports or resets the
budget. `session.extension_runtime.ui.notify(...)` warnings name the detected
shape and the retry count.

## API correction

While writing this, the published guide's `turn_start` example used
`context.api.notify(...)`. `ExtensionContext` has no `api` member (only
`ExtensionCommandContext` does), so the example was silently wrong; handlers
receive read-only context plus `context.ui` and must call actions through the
`tau` object captured in `setup`. The guide now shows that and states the inline
dispatch property above.

## Tests

`tests/test_stall_guard_extension.py` loads the real file through the real
runtime. The end-to-end cases drive `CodingSession.prompt` with `FakeProvider` and
assert a second provider request whose context contains the correction, which is
what proves the queue-drain mechanism; they also pin the non-rescued cases
(plain prose, documented markup, long answers, normal tool rounds) and the
per-run cap. Runtime-level cases pin the decisions that are hard to reach through
a provider: pending-queue suppression, `error`/`aborted` turns, `agent_start`
budget reset, and the cross-run session cap.

Run them with:

```bash
uv run pytest tests/test_stall_guard_extension.py tests/test_example_extensions.py
```

## Known limits

- The guard can only act while a run exists. If a frontend has already let the
  run settle, nothing is rescued; Tau's idle-time `turn_requested` path would
  start a new run instead, which the guard does not use.
- Messages queued by other extensions through the session view emit no
  `queue_update`, so they are not counted as pending; the budget still bounds
  the result.
- Budgets are per generation and not persisted to the session; a resumed session
  begins with a full budget.
