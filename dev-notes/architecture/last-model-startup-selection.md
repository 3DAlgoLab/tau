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

## Two memories, two switches

`persist_default` gates the catalog, not the memory. `select_provider_model` and
`_set_provider_model` used to write `default_model` **and** `lastModel` inside
the same `if persist_default:` block, so the two callers that pass
`persist_default=False` — `_commit_preview()` (a `/model` picker choice committed
on the next accepted turn) and the `/login` provider switch — moved the session
onto a new pair and recorded nothing. The next bare `tau` then started on the
pair the user had just rejected, which is the one behaviour `lastModel` exists to
prevent.

Now `_persist_last_model_choice()` runs on every settled selection. The catalog
default stays under `persist_default`, because an account-scoped or temporary
choice must not silently become what `tau providers list` calls the default; the
startup memory follows whatever the session actually settled on. The memory write
no longer depends on `selected_config` either — a dynamic provider's account-only
model has no durable config to write, and it is still what the user chose.

## `/new` keeps the pair

`CodingSession.new_session()` used to call `resolve_provider_selection(settings)`
with no provider and no model, i.e. the provider catalog's default. On a machine
whose default provider holds no credentials — the case above, where `openai` is
the default and the working pair is `dpi2/coder2` — every `/new` (TUI) and
`new_session` (RPC) failed with "Missing provider API key. Set `OPENAI_API_KEY`
or run `/login openai.`": the user cleared the transcript and was thrown onto a
provider they never chose.

Now the pair in force carries over. The settings default is used only when the
pair can no longer be named at all: the provider entry was deleted, or a provider
that lists models no longer lists this one. An extension or account provider
carries its own inventory, so it is carriable by definition and the dynamic branch
in `new_session` re-picks the model from it; a provider with an empty `models`
tuple is carriable for the same reason.

`/new` clears the transcript, not the user's choice of model. To change the
model, use `/model`; to re-read provider settings, restart or `/reload`.

## Verified on a real machine

`tau -p --mode json --provider dpi2 -m coder2` wrote
`{"codeMode": true, "lastModel": {"provider": "dpi2", "model": "coder2"}}`; a
following bare `tau -p --mode json` streamed 48 events carrying
`"provider": "dpi2"`, `"model": "coder2"`, and the session index recorded the
same pair.

`/new` before and after, same real home, through `tau --mode rpc` with
`get_state` → `new_session` → `get_state`:

```text
HEAD~ (the jump):      {"id":"2","command":"new_session","success":false,
                        "error":"Missing provider API key. Set OPENAI_API_KEY or
                        run /login openai."}
with the fix:         {"id":"2","command":"new_session","success":true}
                      get_state before/after: model.id "coder2", provider "dpi2"
```
