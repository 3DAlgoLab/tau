# Starting a session on the model already in use

## What was added

- `ProviderSettings.last_model: ScopedModelConfig | None`, persisted as a
  `last_model` object in `~/.tau/providers.json`.
- `set_last_provider_model()` and `save_model_choice()` in
  `src/tau_coding/provider_config.py`.
- `resolve_provider_selection()` now consults that memory before a provider's
  configured default.
- `CodingSession._carried_session_selection()` so `/new` starts on the provider
  and model the current session is actually running.

## Why it exists

Pi remembers the model you were on; Tau used to resolve a new session from the
provider's configured default. Those are different values in practice. A default
lives in `catalog.toml` and moves only when a switch is a catalog-level choice,
while the model in use may be account-scoped (a Codex or Claude subscription
model that is not in the catalog at all). So the pair in use could be right and
the default could still send the next session somewhere else.

That is why `last_model` is a separate field instead of an extra write to
`catalog.toml`:

- `default_model` is a catalog-backed definition. Writing an account-scoped model
  id there would put a machine-specific value into a shared definition file.
- `last_model` is a runtime preference. It may name a model the catalog does not
  list, and it is allowed to go stale.

Precedence, deliberately:

1. An explicit `-m <model>` wins outright.
2. `--provider <name>` narrows the answer to that provider — its own remembered
   model if it has one, otherwise its configured default.
3. With nothing asked for, the remembered pair wins.
4. A remembered pair this settings snapshot cannot name (removed provider, model
   no longer declared) is skipped rather than raised. Startup on a default is a
   working session; startup on a validation error is a dead terminal.
5. If the resolved provider has no usable credential, the fallback walks the same
   preference ladder instead of provider order: remembered pair, then saved
   scoped models in save order, then provider defaults in provider order. Taking
   provider order alone let an unrelated provider win the startup slot because it
   sorted first and had an environment key the user never chose. With nothing
   usable at all the plain resolution is returned, so the failure names the
   provider the user configured rather than a substitute.

The ladder lives in `resolve_startup_provider_selection()` in `provider_config.py`,
not in the TUI, so `--print`, `--mode`, and RPC mode resolve identically. Headless
runs used to fail outright on a credential-less `default_provider` while an
authenticated model sat unused.

`/new` uses the session's live choice first, because the running model may come
from a dynamic provider registry that durable settings have never seen. If the
settings snapshot cannot name it, `/new` falls back to normal resolution instead
of stranding the user on a model that no longer resolves.

One consequence worth knowing: the mutation helpers in `provider_config.py` now
use `dataclasses.replace()` instead of rebuilding `ProviderSettings(...)`. The old
constructors silently dropped any field they did not mention, which is exactly how
a new preference field gets lost on its first write.

## How it maps to Pi

Pi keeps a "last used model" and starts there. This keeps the same user-visible
behavior while respecting Tau's split between catalog definitions
(`catalog.toml`) and runtime preferences (`providers.json`).

## How to test

```bash
uv run pytest tests/test_provider_config.py tests/test_coding_session.py -q
```

Look for:

- `test_resolve_provider_selection_prefers_the_last_used_model`
- `test_resolve_provider_selection_keeps_a_provider_request_inside_that_provider`
- `test_resolve_provider_selection_ignores_memory_it_cannot_name`
- `test_save_model_choice_remembers_models_the_catalog_cannot_name`
- `test_session_new_session_starts_on_the_model_in_use`
- `test_session_new_session_falls_back_when_durable_settings_cannot_name_the_provider`
- `test_model_switch_is_remembered_for_the_next_session`
- `test_tui_startup_fallback_starts_on_the_last_used_model`
- `test_tui_startup_fallback_prefers_a_saved_model_to_provider_order`
- `test_tui_startup_fallback_skips_a_remembered_provider_without_keys`
- `test_tui_startup_fallback_uses_provider_order_without_history`
- `test_resolve_startup_provider_selection_prefers_a_saved_model_to_provider_order`
- `test_resolve_startup_provider_selection_keeps_the_default_without_keys`
- `test_print_session_record_starts_headless_on_the_saved_model`
