# Compaction Hook Seam (`before_compact`)

Phase 0 of porting pi-fabric's deterministic compaction engine. It gives an
extension the interception point the engine needs, without changing any default
behavior: Tau still compacts with a model-generated summary unless an extension
says otherwise.

## What was added

```text
src/tau_agent/session/entries.py        CompactionEntry.details
src/tau_coding/extensions/api.py        BeforeCompactEvent, CompactionContextEntry, CompactHookResult, "before_compact"
src/tau_coding/extensions/runtime.py    run_before_compact_hooks(), CompactHookOutcome
src/tau_coding/session.py               CompactionPlan.context_rows, _run_before_compact_hooks(), _resolve_compaction_summary()
```

`before_compact` joins the lifecycle hook set alongside `input`, `tool_call`,
`tool_result`, and `project_trust`. It fires at all three compaction triggers —
manual (`/compact`), threshold (`_maybe_auto_compact`), and provider overflow
(`_try_overflow_compact`) — and a handler may:

- return `CompactHookResult()` (or `None`) to defer to Tau's summarizer,
- return `action="replace"` with a `summary`, skipping the summarizer request
  entirely, optionally moving the cut with `first_kept_entry_id`,
- return `action="cancel"` to abort this compaction.

`CompactionEntry.details: dict[str, JSONValue] | None` carries engine-owned
provenance — engine id, schema version, source ranges, omission counts — so a
later compaction can re-derive state from typed facts instead of re-reading
rendered summary prose.

## Why it exists

pi-fabric's compaction feature is built entirely on Pi's
`session_before_compact` hook. Tau previously exposed only the observational
`compaction_start`/`compaction_end` agent events, which a frontend renders but no
extension can alter. Without an interception seam, any alternative engine would
have to monkeypatch `CodingSession` itself.

Skipping the summarizer call is the point. Tau's current `/compact` spends a full
provider round trip, which makes compaction slow and makes it expensive for
child agents to compact. A deterministic engine needs neither.

## Why the payload carries entry ids

The first version of the seam handed handlers `messages: tuple[AgentMessage, ...]`
and nothing else, and `ExtensionContext.transcript()` strips ids too. Checking
that against what a deterministic engine actually needs showed three things it
could not do with bare messages:

- choose its own cut, since a boundary is an **entry id**;
- keep a `toolCall` on the same side of the cut as its `toolResult`, since
  closure safety is a statement about positions in the entry stream;
- cite what it left out, so an omitted span stays expandable from the session.

So `BeforeCompactEvent.entries` carries the whole active context in order as
`CompactionContextEntry(entry_id, message)` rows — including the rows the cut
would keep, because the retained half is what tells an engine whether its
boundary orphans a tool pair. `messages` stays as the exact prefix Tau's
`first_kept_entry_id` would drop, which is all a simple handler wants.

Plumbing cost is one field: `_recent_preserving_compaction_plan()` already builds
from `self._active_context_rows()`, which is `tuple[tuple[str, AgentMessage], ...]`
— the same shape `tau_coding.compaction` normalizes — so `CompactionPlan` now
keeps those rows instead of discarding them.

This was safe to change because `before_compact` exists in no released Tau: the
installed `tau-ai` 0.4.8 has no trace of it, so there is no older handler to keep
source-compatible.

## Mapping to Pi's design

| Pi | Tau |
| --- | --- |
| `session_before_compact` hook | `before_compact` lifecycle hook |
| hook payload `{ messages, firstKeptEntryId, ... }` | `BeforeCompactEvent(entries, messages, first_kept_entry_id, ...)` |
| hook result `{ summary, firstKeptEntryId }` | `CompactHookResult(summary, first_kept_entry_id, details)` |
| handlers in load order, latest non-cancelling wins | same |
| a cancellation terminates dispatch | same |
| `compaction.details` provenance envelope | `CompactionEntry.details` |

Tau keeps its short lifecycle-hook naming (`input`, `tool_call`,
`before_compact`) rather than Pi's `session_before_*` prefix, matching the
existing `compaction_start`/`compaction_end` event names.

## Precedence and failure containment

Handlers run in extension load order. `action="continue"`, `None`, or a wrong
result type leave the decision to later handlers and finally to Tau. A
`"replace"` with a blank summary is rejected as a bad result rather than
persisted — an empty summary would silently erase model context on replay. A
handler that raises is recorded as an extension diagnostic (visible in
`/session`) and cannot abort the turn; only an explicit `"cancel"` does that.

The session validates `first_kept_entry_id` against the active context through
the existing `_append_compaction` guard. An extension that returns a boundary
outside the active branch therefore produces no compaction: manual compaction
raises `ValueError`, and the automatic path is contained by
`_try_auto_compact`'s existing error handling and leaves the conversation
uncompacted.

## Compatibility rules

`details` is read-side only in this phase: no writer in Tau populates it, and
`entry_to_json_line` omits it when `None`, so every existing session loads and
replays byte-identically. Released Tau validates session entries with
`extra="forbid"`, so a field written by newer code breaks older readers. Any
follow-up that writes `details` must land the schema first (done here) and must
keep the payload inside what released readers can tolerate, or run against a
separate `TAU_HOME` while developing.

## How to use it

```python
from tau_coding.extensions import BeforeCompactEvent, CompactHookResult


def setup(tau):
    def engine(event: BeforeCompactEvent, context) -> CompactHookResult:
        rows = [(item.entry_id, item.message) for item in event.entries]
        summary = deterministic_projection(rows, event)  # no provider call
        return CompactHookResult(
            action="replace",
            summary=summary,
            details={"engine": "my-engine", "version": 2},
        )

    tau.on("before_compact", engine)
```

## Tests

```text
tests/test_extensions.py       six dispatch tests: latest-wins precedence, cancel
                               short-circuit, defer-by-default, blank-summary
                               rejection, crash containment, payload integrity
                               including entry ids and the messages/cut prefix
tests/test_coding_session.py   seven wiring tests: manual replacement with no
                               provider call, boundary override, entries covering
                               both halves of the cut, engine-chosen boundary,
                               manual cancel, automatic cancel, invalid-boundary
                               containment
tests/test_session.py          `details` JSONL round trip; sessions without the
                               field load as `None` and serialize without it
```

```bash
uv run pytest tests/test_extensions.py tests/test_coding_session.py tests/test_session.py
```
