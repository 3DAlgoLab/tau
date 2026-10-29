# Remembered Model Selection at Startup

## What was added

A bare `tau` (TUI), `tau --print`, or `tau --rpc` starts from the provider/model
pair the **last session actually settled on**, instead of the provider
catalog's `default_model`.

Three layers, because the value has to be remembered, stored, and applied:

- `shell_config.LastModelPreference` -- the durable shape in
  `~/.tau/settings.json` (`{"lastModel": {"provider": …, "model": …}}`), with
  `remember_model_choice()` as the only writer (atomic temp+replace, preserving
  keys this version does not understand).
- `last_model.py` -- `remember_last_model()` (one settings.json write per settled
  selection) and `last_model_selection()` / `apply_last_model()`, which read the
  pair back and decide whether it may be used.
- The two places a selection is *settled*: `CodingSession.set_provider` and
  `CodingSession.create_settled_runtime` (the runtime seam every entry point --
  TUI, print, RPC, `tau-dev`-style hosts -- funnels through), plus
  `run_tui_app` / `run_openai_print_mode` / `run_openai_rpc_mode` at startup.

## Why a separate key from `default_model`

Provider preferences already hold `default_model`. Overwriting that from a
session would be surprising in two ways: it silently rewrites what
`tau providers list` calls the default, and model **ids are not unique across
providers** (`gpt-5.4` exists on several), so the remembered value has to carry
the provider it belongs to. `lastModel` is a pair; preferences stay the user's.

## Why the fallback has to stay

A remembered pair is a *default*, never a requirement. If it no longer resolves
-- the provider entry was deleted, the model left a catalog that lists one, the
account model has no credentials -- startup falls back to the ordinary chain.
A remembered pair that cannot be used would otherwise make **every** new
session on the machine fail, which is a much worse failure than starting on the
catalog default. The credential gate is passed by the interactive and RPC paths
(they have an OAuth/key fallback chain to fall back *into*); print mode skips it,
because a print run is one-shot and its provider error is the useful answer.

Catalog validation is deliberately durable-only: providers whose inventory lives
at the account (`models == ()`) accept any remembered model id, otherwise an
account-only model could never survive a restart.

## Precedence

```text
--provider / --model        explicit wins, and refreshes the remembered pair
saved transcript (--resume / --session-id)   carries its own model; memory is ignored
lastModel                   the remembered pair
provider default_model      unchanged behaviour
```

## Contract notes

- A partial flag (`--model` without `--provider`, no `--provider` in the
  remembered pair) applies **neither** and writes nothing: pairing an explicit
  model with an unrelated remembered provider would invent a selection nobody
  asked for.
- Persisting on every settled selection means an *auto-selected* fallback model
  also gets remembered. Accepted: the next start then runs on a model the
  previous start genuinely used, instead of jumping back to a default that just
  failed.
- `remember_model_choice()` preserves unknown top-level keys. The reader has
  always ignored them, so a settings.json written by a newer Tau used to lose
  its own keys the first time an older Tau wrote a flag into the same file.

## Verified on a real machine

`tau -p --mode json --provider dpi2 -m coder2` wrote
`{"codeMode": true, "lastModel": {"provider": "dpi2", "model": "coder2"}}`; a
following bare `tau -p --mode json` streamed 48 events carrying
`"provider": "dpi2"`, `"model": "coder2"`, and the session index recorded the
same pair.
