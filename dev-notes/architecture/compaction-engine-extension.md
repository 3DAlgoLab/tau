---
title: "Deterministic Compaction As An Extension"
---

Phase 1d of porting pi-fabric's deterministic compaction. Phases 1a–1c built the
primitives; this phase makes them change what a real session does, by riding the
`before_compact` seam from Phase 0b.

```text
examples/extensions/deterministic_compaction.py   policy, registration, provenance
src/tau_coding/compaction/                        primitives, still behavior-free
```

## Why the policy lives in an extension

The engine package stays a library: no registration, no session knowledge, no
UI. The extension owns the decisions a product would argue about -- when to
decline, what provenance to record, what a command reports -- and it is a file
you can delete. A/B is therefore trivial: same session log, same trigger,
summarizer on one day and engine the next, and the compaction entry says which
one wrote the summary.

The cost of extension-first is honest and worth stating: an extension controls
**what** a compaction writes, not **when** one fires. Trigger threshold, reserve,
and keep-recent stay core constants. If that ceiling ever becomes the bottleneck,
the fallback is a built-in `compaction.engine = "llm" | "deterministic"` selector
reusing the same primitives; nothing in `tau_coding/compaction/` would change.

## Three ways to decline

`run_engine()` returns `action="continue"` -- Tau's summarizer request runs
unchanged -- in exactly three cases:

| decision | condition |
| --- | --- |
| `no_cut` | `compute_cut()` found no closure-safe boundary within budget |
| `empty_window` | the boundary would summarize nothing, or keep nothing |
| `qa_failed` | a ground-truth probe lost its fact, or its pointer |

`qa_failed` is the load-bearing one. A summary is only written once
`qa_report(...).ok` says nothing went silently missing, so a bug in a fold
degrades to the old summarizer path rather than to a shorter summary that
quietly dropped an unresolved error. The failed probe reasons go into the
provenance blob, so the fallback is diagnosable from the session log.

## Provenance

Extension summaries carry no provider usage: the runtime attributes no cost to a
summary no model wrote. Everything else the engine knows goes on the compaction
entry as `details`:

```text
engine, engine_version, decision, trigger
summarized_entries, retained_entries, events, thinking_blocks_erased
summary_bytes, clipped_bytes, section_headers, omitted{...}
qa{probes, score, content_score, address_score}
retained_raw_tokens, budget{...}
```

`budget` is the calibrated cut envelope -- token scale, fixed overhead, the
ceilings, and the binding constraint -- which is what makes a boundary
reproducible after the fact rather than merely plausible.

## What the integration tests pin

`tests/test_compaction_engine_extension.py` loads the example into a real
`CodingSession` with `FakeProvider`, so the chain under test is loader → seam →
engine → append → replay.

- provider call count is unchanged across compaction: the summarizer round trip
  is gone, not merely shorter;
- thinking text never reaches the summary, and `thinking_blocks_erased` counts it;
- `qa` scores are 1.0 for a real render, which is what let it be written;
- the active context after compaction starts at the engine's boundary;
- two independent sessions with the same turns produce the same bytes.

The determinism test masks exactly two things: the UTC timestamp in the footer
and the 32-hex entry ids. Both are session-scoped, and everything the engine
itself decides is byte-identical -- which is the property that makes a compaction
diffable, and the one an LLM summarizer cannot offer.

## Known edges

`/compact-engine` reads module state from the last decision in this process, not
from the session log. A QA report is only computable while the summarized events
are still in memory, so scores are recorded at write time and the command can
only echo them. It does not survive `/reload`, and it reports the most recent
decision rather than the one attached to a specific entry.

Still open in this track: an inline TUI affordance to expand omitted content from
its `[entry ...]` pointer, a carry-forward entry type (the renderer already
accepts `carry_lines`), and the built-in engine selector if extension scope turns
out to be too tight.

## Next

The compaction folds address omitted content by entry id. Making that address
*dereferenceable by model code* -- slice the log, grep it, expand a fold in a
kernel rather than re-asking a provider -- is the code-mode track: see
[Code-Mode Kernel (Phase 2a)](./code-mode-kernel.md).
