"""The provisional ``tau-rlm`` launcher must isolate state, never share it."""

from __future__ import annotations

from pathlib import Path
from tomllib import loads

import pytest

from tau_coding.dev_cli import (
    DEV_HOME_DEFAULT,
    DEV_HOME_ENV,
    HOME_ENV,
    prepare_dev_environment,
    provider_import_hint,
)


def test_the_launcher_repoints_home_at_the_dev_directory() -> None:
    environ: dict[str, str] = {}
    home = prepare_dev_environment(environ)
    assert home is not None
    assert home == Path(DEV_HOME_DEFAULT).expanduser()
    assert environ[HOME_ENV] == str(home)
    assert home.is_absolute()


def test_an_explicit_home_always_wins() -> None:
    environ = {HOME_ENV: "/srv/tau"}
    assert prepare_dev_environment(environ) is None
    assert environ == {HOME_ENV: "/srv/tau"}


def test_the_track_specific_home_overrides_the_default() -> None:
    environ = {DEV_HOME_ENV: "/tmp/tau-track"}
    assert prepare_dev_environment(environ) == Path("/tmp/tau-track")
    assert environ[HOME_ENV] == "/tmp/tau-track"


def test_a_relative_home_is_refused_before_paths_are_built() -> None:
    try:
        prepare_dev_environment({DEV_HOME_ENV: "relative/home"})
    except ValueError as exc:
        assert DEV_HOME_ENV in str(exc)
    else:  # pragma: no cover - the guard is the point
        raise AssertionError("a relative home must raise")


def test_the_report_line_only_arrives_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A hermetic HOME keeps this about the report contract, not about this machine.
    monkeypatch.delenv(HOME_ENV, raising=False)
    monkeypatch.delenv(DEV_HOME_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    seen: list[str] = []
    prepare_dev_environment({}, report=seen.append)
    assert len(seen) == 1
    assert "experimental build" in seen[0]
    assert prepare_dev_environment({HOME_ENV: "/x"}) is None
    assert len(seen) == 1


def test_a_fresh_dev_home_is_told_how_to_reach_a_model(tmp_path: Path) -> None:
    previous = tmp_path / ".tau"
    previous.mkdir()
    (previous / "providers.json").write_text('{"providers": []}', encoding="utf-8")
    dev = tmp_path / ".tau-dev"
    dev.mkdir()

    hint = provider_import_hint(dev, previous)

    assert hint is not None
    assert "tau-rlm setup" in hint
    assert str(previous / "providers.json") in hint


def test_the_import_hint_stays_quiet_when_there_is_nothing_to_import(tmp_path: Path) -> None:
    dev = tmp_path / ".tau-dev"
    dev.mkdir()
    previous = tmp_path / ".tau"
    previous.mkdir()

    assert provider_import_hint(dev, None) is None
    assert provider_import_hint(dev, previous) is None
    assert provider_import_hint(dev, dev) is None

    (previous / "providers.json").write_text("{}", encoding="utf-8")
    assert provider_import_hint(dev, previous) is not None
    (dev / "providers.json").write_text("{}", encoding="utf-8")
    assert provider_import_hint(dev, previous) is None


def test_the_console_script_is_registered_and_named_apart_from_tau() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["tau"] == "tau_coding.cli:app"
    assert scripts["tau-rlm"] == "tau_coding.dev_cli:main"


def test_the_report_names_the_import_path_on_a_first_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(HOME_ENV, raising=False)
    monkeypatch.delenv(DEV_HOME_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    previous = tmp_path / ".tau"
    previous.mkdir()
    (previous / "providers.json").write_text('{"providers": []}', encoding="utf-8")

    environ = {DEV_HOME_ENV: str(tmp_path / ".tau-dev"), "HOME": str(tmp_path)}
    seen: list[str] = []
    home = prepare_dev_environment(environ, report=seen.append)

    assert home == tmp_path / ".tau-dev"
    assert len(seen) == 2
    assert "no provider config" in seen[1]
    assert environ[HOME_ENV] == str(tmp_path / ".tau-dev")
