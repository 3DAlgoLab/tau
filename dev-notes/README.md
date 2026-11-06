# Tau dev notes (contributor build-log)

These are the internal, phase-by-phase build journals and design records for Tau.
They are **not** published on the docs site — they live here for contributors who
want to trace how the system was assembled.

User-facing documentation lives in `website/content/` and is published at
<https://twotimespi.dev/>.

## Contents

- `design/` — high-level design docs written alongside the build:
  - `00-roadmap.md` — phased roadmap
  - `01-architecture.md` — the three-layer split
  - `02-agent-loop.md` — agent loop responsibilities
  - `03-tools.md` — built-in tool design
  - `04-sessions.md` — session tree / persistence design
  - `05-core-types-and-events.md` — provider-neutral types and events
  - `project-trust.md` — researched, implementation-ready project-trust design
    (design only; enforcement is not implemented)
  - `agent-loop.md`, `harness.md` — harness/loop reference notes
- `architecture/` — per-phase implementation notes (`phase-1` … `phase-24`, plus
  hardening and feature notes). Each answers: what was added, why it exists, how
  later phases use it.
- `adr/` — architecture decision records.
- `catalog-model-safety.md` — checklist for adding providers and models to the built-in catalog safely.
- `google-stream-completion.md` — why native Google streams require an explicit
  `finishReason` before Tau reports successful completion.
- `startup-thinking-level-fallback.md` — why startup resolves a valid thinking
  level per model instead of assuming the global `medium` default.
- `last-model-starting-model.md` — why a fresh launch and `/new` start on the
  model actually in use, and why that memory is a `providers.json` preference
  rather than a `catalog.toml` default.
- `models-dev-catalog.md` — Pi-compatible build-time models.dev catalog
  generation, offline fallback, and snapshot refresh workflow.
- `llama-cpp-phase-5.md` — built-in llama.cpp connection, safe state, `/local`,
  failure handling, and Phase 5 validation.
- `architecture/phase-6-local-inference-hardening.md` — second-backend contract
  validation, lifecycle hardening, migration, and security decisions.
- `stall-guard-extension.md` — how the shipped `stall_guard` example rescues a
  turn that ended with no tool call, and why inline agent-event dispatch makes a
  queued follow-up continue the same run.
- `python-kernel-extension.md` — how the shipped `python_kernel` example keeps one
  live interpreter per session, why its interrupt ladder is SIGINT-then-kill
  instead of a Jupyter control channel, and why that interpreter is the project's
  own `.venv` rather than whatever Python Tau happens to run on.
- `extension-compaction-override.md` — the `session_before_compact` hook that lets
  an extension replace Tau's model-generated summary, why every compaction trigger
  funnels through one helper, and why `CompactionEntry.details` and the raw
  `context.entries` branch exist for lossless recovery.
- `compaction-pressure-and-retired-summaries.md` — how a session came to summarize a
  context that was not near its window: a provider-measured floor that never reset at a
  compaction, and a fold that kept the summary it replaced because its boundary could
  not name it. Includes a third failure left unfixed, with the evidence for it.
- `web-search-extension.md` — how the shipped `web_search` example reaches five
  search backends and reads a page in reader mode, and why titles-only results,
  a rejected key, and class-named page chrome each needed their own rule instead
  of one generic "try the next provider" loop.
- `copy-last-response-command.md` — how `/copy` splits "what to copy" from "how to
  reach this frontend's clipboard", and why the core helper never emits terminal
  escape codes.
- `tui-latency-fixes.md` — profiled causes of slow prompt echo and startup, and
  the stamp-validated caches and scheduling changes that fixed them.
- `tui-transcript-spacing-and-tool-runs.md` — the uniform one-blank-line block
  rhythm, invisible whitespace-only blocks, and same-tool run continuation.

The roadmap is tracked in [GitHub issue #1](https://github.com/huggingface/tau/issues/1).
