"""Remembering the provider/model pair, and starting a new session from it."""

from pathlib import Path

import pytest

from tau_coding.last_model import (
    apply_last_model,
    last_model_selection,
    remember_last_model,
)
from tau_coding.paths import TauPaths
from tau_coding.provider_config import OpenAICompatibleProviderConfig, ProviderSettings
from tau_coding.shell_config import (
    LastModelPreference,
    ShellSettings,
    load_shell_settings,
    remember_model_choice,
    shell_settings_path,
)

API_KEY_ENV = "TAU_TEST_LAST_MODEL_KEY"


class NoCredentials:
    """A credential reader that has nothing for anyone."""

    def get(self, name: str) -> str | None:
        return None


def paths_in(tmp_path: Path) -> TauPaths:
    return TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")


def settings_with(
    *,
    name: str = "dpi2",
    models: tuple[str, ...] = ("coder2", "coder3"),
    default_model: str = "coder3",
) -> ProviderSettings:
    return ProviderSettings(
        default_provider=name,
        providers=(
            OpenAICompatibleProviderConfig(
                name=name,
                base_url="http://127.0.0.1:9/v1",
                api_key_env=API_KEY_ENV,
                models=models,
                default_model=default_model,
            ),
        ),
    )


def settings_holding(preference: LastModelPreference, tmp_path: Path) -> ShellSettings:
    paths = paths_in(tmp_path)
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"codeMode": false}', encoding="utf-8")
    remember_model_choice(preference, paths=paths)
    return load_shell_settings(paths)


def test_the_pair_a_session_settled_on_becomes_the_next_starting_pair(
    tmp_path: Path,
) -> None:
    paths = paths_in(tmp_path)

    remember_last_model("dpi2", "coder2", paths=paths)

    loaded = load_shell_settings(paths)
    assert loaded.last_model == LastModelPreference(provider_name="dpi2", model="coder2")
    assert last_model_selection(settings_with(), loaded) == ("dpi2", "coder2")


def test_a_pair_for_a_removed_provider_is_ignored_not_fatal(tmp_path: Path) -> None:
    holding = settings_holding(LastModelPreference(provider_name="gone", model="m"), tmp_path)

    assert last_model_selection(settings_with(), holding) is None


def test_a_model_the_durable_catalog_does_not_list_is_ignored(tmp_path: Path) -> None:
    # One typo on the command line must not make every later bare `tau` fail.
    holding = settings_holding(LastModelPreference(provider_name="dpi2", model="coder9"), tmp_path)

    assert last_model_selection(settings_with(models=("coder2",)), holding) is None


def test_an_account_only_model_survives_a_restart(tmp_path: Path) -> None:
    # Providers whose inventory lives at the account, not in the catalog, have
    # nothing to validate against; their remembered model is taken as-is.
    holding = settings_holding(
        LastModelPreference(provider_name="dpi2", model="account-only-model"), tmp_path
    )

    assert last_model_selection(settings_with(models=()), holding) == (
        "dpi2",
        "account-only-model",
    )


def test_a_pair_without_credentials_is_dropped_when_the_interpreter_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    holding = settings_holding(LastModelPreference(provider_name="dpi2", model="coder2"), tmp_path)

    assert last_model_selection(settings_with(), holding, credential_reader=NoCredentials()) is None
    # Print mode has no fallback chain to protect, so it keeps the remembered
    # pair and lets the provider error surface instead.
    assert last_model_selection(settings_with(), holding) == ("dpi2", "coder2")


def test_explicit_flags_become_the_next_starting_pair(tmp_path: Path) -> None:
    settings = settings_with()

    assert apply_last_model(settings, ShellSettings(), provider_name="dpi2", model="coder2") == (
        "dpi2",
        "coder2",
    )
    remembered = load_shell_settings()
    assert remembered.last_model == LastModelPreference(provider_name="dpi2", model="coder2")
    assert apply_last_model(settings, remembered, provider_name=None, model=None) == (
        "dpi2",
        "coder2",
    )


def test_a_partial_selection_leaves_memory_untouched(tmp_path: Path) -> None:
    paths = paths_in(tmp_path)

    assert apply_last_model(settings_with(), ShellSettings(), provider_name="dpi2", model=None) == (
        "dpi2",
        None,
    )
    assert not shell_settings_path(paths).exists()


def test_a_resumed_transcript_outranks_memory(tmp_path: Path) -> None:
    holding = settings_holding(LastModelPreference(provider_name="dpi2", model="coder2"), tmp_path)

    assert apply_last_model(
        settings_with(), holding, provider_name=None, model=None, explicit_resume=True
    ) == (None, None)


def test_a_bare_start_falls_back_to_catalog_defaults_when_nothing_is_remembered() -> None:
    assert apply_last_model(settings_with(), ShellSettings(), provider_name=None, model=None) == (
        None,
        None,
    )
