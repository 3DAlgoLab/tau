# Compaction Hook Seam (`before_compact`)

Phase 0 of porting pi-fabric's deterministic compaction engine. It gives an
extension the interception point the engine needs, without changing any default
behavior: Tau still compacts with a model-generated summary unless an extension
says otherwise.

## What was added

```text
src/tau_agent/session/entries.py        CompactionEntry.details
src/tau_coding/extensions/api.py        BeforeCompactEvent, CompactHookResult, "before_compact"
src/tau_coding/extensions/runtime.py    run_before_compact_hooks(), CompactHookOutcome
src/tau_coding/session.py               _run_before_compact_hooks(), _resolve_compaction_summary()
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

## Mapping to Pi's design

| Pi | Tau |
| --- | --- |
| `session_before_compact` hook | `before_compact` lifecycle hook |
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
        summary = deterministic_projection(event.messages, event)  # no provider call
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
tests/test_coding_session.py   five wiring tests: manual replacement with no
                               provider call, boundary override, manual cancel,
                               automatic cancel, invalid-boundary containment
tests/test_session.py          `details` JSONL round trip; sessions without the
                               field load as `None` and serialize without it
```

```bash
uv run pytest tests/test_extensions.py tests/test_coding_session.py tests/test_session.py
```
