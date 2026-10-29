from json import loads
from pathlib import Path

import pytest

from tau_coding import (
    LastModelPreference,
    ShellConfigError,
    ShellSettings,
    TauPaths,
    load_shell_settings,
    remember_model_choice,
    shell_settings_from_json,
    shell_settings_path,
)


def test_load_shell_settings_missing_file_uses_defaults(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    assert load_shell_settings(paths) == ShellSettings()


def test_load_shell_settings_accepts_pi_style_shell_command_prefix(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"shellCommandPrefix": "shopt -s expand_aliases\\nalias gs=\\"git status\\""}',
        encoding="utf-8",
    )

    settings = load_shell_settings(paths)

    assert settings.shell_command_prefix == 'shopt -s expand_aliases\nalias gs="git status"'
    assert settings.to_json() == {
        "shellCommandPrefix": 'shopt -s expand_aliases\nalias gs="git status"'
    }


def test_shell_settings_accepts_tau_style_shell_command_prefix() -> None:
    settings = shell_settings_from_json({"shell_command_prefix": " alias ll='ls -la' "})

    assert settings.shell_command_prefix == "alias ll='ls -la'"


def test_shell_settings_ignore_unknown_fields() -> None:
    settings = shell_settings_from_json(
        {
            "shellCommandPrefix": "source ~/.tau/aliases.bash",
            "future_shell_option": True,
        }
    )

    assert settings.shell_command_prefix == "source ~/.tau/aliases.bash"


def test_code_mode_is_on_when_nothing_says_otherwise() -> None:
    assert ShellSettings().code_mode is True
    assert shell_settings_from_json({}).code_mode is True
    assert ShellSettings().to_json() == {}


def test_code_mode_can_be_turned_off_for_every_session_on_this_machine() -> None:
    settings = shell_settings_from_json({"codeMode": False})
    assert settings.code_mode is False
    assert settings.to_json() == {"codeMode": False}


def test_code_mode_round_trips_with_a_shell_prefix() -> None:
    settings = shell_settings_from_json({"shellCommandPrefix": "bash -c", "codeMode": False})
    assert settings.to_json() == {"shellCommandPrefix": "bash -c", "codeMode": False}


def test_code_mode_accepts_the_snake_case_alias() -> None:
    assert shell_settings_from_json({"code_mode": False}).code_mode is False


def test_code_mode_refuses_both_spellings_at_once() -> None:
    with pytest.raises(ShellConfigError, match="codeMode"):
        shell_settings_from_json({"codeMode": False, "code_mode": True})


@pytest.mark.parametrize("value", ["false", "off", 0, None])
def test_code_mode_refuses_a_non_boolean(value: object) -> None:
    # "false" as a string would mean the opposite of what its author wrote.
    with pytest.raises(ShellConfigError, match="boolean"):
        shell_settings_from_json({"codeMode": value})


def test_recursion_is_off_when_nothing_says_otherwise() -> None:
    assert ShellSettings().code_mode_recursion is False
    assert shell_settings_from_json({}).code_mode_recursion is False
    assert "codeModeRecursion" not in ShellSettings().to_json()


def test_recursion_is_opt_in_and_round_trips() -> None:
    settings = shell_settings_from_json({"codeModeRecursion": True})
    assert settings.code_mode_recursion is True
    assert settings.to_json() == {"codeModeRecursion": True}


def test_recursion_accepts_the_snake_case_alias() -> None:
    assert shell_settings_from_json({"code_mode_recursion": True}).code_mode_recursion is True


def test_recursion_refuses_both_spellings_at_once() -> None:
    with pytest.raises(ShellConfigError, match="codeModeRecursion"):
        shell_settings_from_json({"codeModeRecursion": True, "code_mode_recursion": False})


@pytest.mark.parametrize("value", ["true", "on", 1])
def test_recursion_refuses_a_non_boolean(value: object) -> None:
    # A truthy string would silently switch on a setting that starts processes.
    with pytest.raises(ShellConfigError, match="boolean"):
        shell_settings_from_json({"codeModeRecursion": value})


def test_recursion_survives_a_session_without_code_mode() -> None:
    # Enabling recursions while code mode is off is nonsense but not fatal: the
    # tool is absent, so the recursion flag has nothing to bind into.
    settings = shell_settings_from_json({"codeMode": False, "codeModeRecursion": True})
    assert settings.code_mode is False
    assert settings.code_mode_recursion is True


def test_the_kernel_interpreter_is_inherited_by_default() -> None:
    # Tau's own interpreter is the default because it already has tau importable
    # and needs no setup; naming another one is a deliberate act.
    assert ShellSettings().code_mode_python is None
    assert "codeModePython" not in ShellSettings().to_json()


def test_a_configured_kernel_interpreter_round_trips() -> None:
    settings = shell_settings_from_json({"codeModePython": "/usr/bin/python3.12"})
    assert settings.code_mode_python == "/usr/bin/python3.12"
    assert settings.to_json() == {"codeModePython": "/usr/bin/python3.12"}


def test_the_kernel_interpreter_accepts_the_snake_case_alias() -> None:
    assert shell_settings_from_json({"code_mode_python": "pypy3"}).code_mode_python == "pypy3"


def test_the_kernel_interpreter_refuses_both_spellings_at_once() -> None:
    with pytest.raises(ShellConfigError, match="codeModePython"):
        shell_settings_from_json({"codeModePython": "python3", "code_mode_python": "python3"})


def test_the_kernel_interpreter_must_be_a_name_or_null() -> None:
    with pytest.raises(ShellConfigError, match="codeModePython"):
        shell_settings_from_json({"codeModePython": 3})
    # null is how a user says "back to Tau's own interpreter"; a blank name is
    # not a name, so both mean the same thing and neither blocks the tool.
    assert shell_settings_from_json({"codeModePython": None}).code_mode_python is None
    assert shell_settings_from_json({"codeModePython": "   "}).code_mode_python is None


def test_the_kernel_interpreter_is_read_from_the_durable_settings_file(
    tmp_path: Path,
) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text('{"codeModePython": "python3.12"}', encoding="utf-8")

    assert load_shell_settings(paths).code_mode_python == "python3.12"


def test_recursion_is_read_from_the_durable_settings_file(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text('{"codeModeRecursion": true}', encoding="utf-8")

    assert load_shell_settings(paths).code_mode_recursion is True


def test_code_mode_is_read_from_the_durable_settings_file(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text('{"codeMode": false}', encoding="utf-8")

    assert load_shell_settings(paths).code_mode is False


def test_the_remembered_model_pair_round_trips_through_the_settings_file() -> None:
    settings = shell_settings_from_json({"lastModel": {"provider": "dpi2", "model": "coder2"}})

    assert settings.last_model == LastModelPreference(provider_name="dpi2", model="coder2")
    assert settings.to_json() == {"lastModel": {"provider": "dpi2", "model": "coder2"}}


def test_the_remembered_pair_accepts_the_snake_case_alias() -> None:
    settings = shell_settings_from_json({"last_model": {"provider": "openai", "model": "gpt-5.4"}})

    assert settings.last_model is not None
    assert settings.last_model.provider_name == "openai"
    assert settings.last_model.model == "gpt-5.4"


def test_the_remembered_pair_refuses_both_spellings() -> None:
    pair = {"provider": "dpi2", "model": "coder2"}

    with pytest.raises(ShellConfigError, match="lastModel or last_model"):
        shell_settings_from_json({"lastModel": pair, "last_model": pair})


@pytest.mark.parametrize(
    "payload",
    [
        '"dpi2/coder2"',
        '{"provider": "dpi2"}',
        '{"provider": "", "model": "coder2"}',
        '{"provider": "dpi2", "model": 7}',
    ],
)
def test_a_half_shaped_remembered_pair_is_refused(payload: object) -> None:
    # A pair that parses to "provider dpi2, model ?" would start every later
    # session on a half-known selection; refusing it keeps the failure at the file.
    with pytest.raises(ShellConfigError):
        shell_settings_from_json({"lastModel": payload})


def test_remembering_a_pair_keeps_keys_this_version_does_not_understand(
    tmp_path: Path,
) -> None:
    # The reader ignores unknown keys so a newer Tau can write them; a writer
    # must not quietly delete them on the way back out.
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True)
    path.write_text('{"codeMode": false, "someFutureSetting": {"kept": true}}', encoding="utf-8")

    remember_model_choice(LastModelPreference(provider_name="dpi2", model="coder2"), paths=paths)

    reloaded = load_shell_settings(paths)
    assert reloaded.code_mode is False
    assert reloaded.last_model == LastModelPreference(provider_name="dpi2", model="coder2")
    assert loads(path.read_text(encoding="utf-8"))["someFutureSetting"] == {"kept": True}


def test_remembering_a_pair_replaces_the_previous_one(tmp_path: Path) -> None:
    paths = TauPaths(home=tmp_path / ".tau", agents_home=tmp_path / ".agents")

    remember_model_choice(LastModelPreference(provider_name="dpi2", model="coder2"), paths=paths)
    remember_model_choice(LastModelPreference(provider_name="zai", model="glm-5.1"), paths=paths)

    raw = loads(shell_settings_path(paths).read_text(encoding="utf-8"))
    assert raw["lastModel"] == {"provider": "zai", "model": "glm-5.1"}
    assert "last_model" not in raw
