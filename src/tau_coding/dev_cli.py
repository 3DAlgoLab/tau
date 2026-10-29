"""Provisional entry point for the code-mode and RLM track: ``tau-rlm``.

``tau`` belongs to whatever Tau is installed -- right now the official PyPI
build. The features on this track are still moving, and a session store is durable
data, so this command exists to be run, broken, and reset without ever touching
that install or the sessions it writes:

* it runs from this checkout, so the code being exercised is the code under
  development;
* it points ``TAU_HOME`` at a separate home (``~/.tau-dev`` by default), so
  sessions, settings, logs, and credentials for experimental work never land in
  ``~/.tau``.

It is one module plus one ``[project.scripts]`` line. Deleting both costs nothing,
which is the point: nothing on this track may become load-bearing by accident.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path

from tau_coding.cli import app
from tau_coding.paths import TauPaths
from tau_coding.provider_config import provider_settings_path

__all__ = ["DEV_HOME_DEFAULT", "DEV_HOME_ENV", "HOME_ENV", "main", "prepare_dev_environment"]

HOME_ENV = "TAU_HOME"
DEV_HOME_ENV = "TAU_RLM_HOME"
DEV_HOME_DEFAULT = "~/.tau-dev"
"""Home used when neither ``TAU_HOME`` nor ``TAU_RLM_HOME`` is set.

``TAU_HOME`` wins when the caller sets it, because that is the one knob Tau itself
documents; ``TAU_RLM_HOME`` exists to move this track's state without repointing
the whole environment for a child process that inherits it.
"""


def _home_of(environ: Mapping[str, str]) -> Path:
    """Resolve the home Tau would use for this environment, mirroring ``TauPaths``."""
    raw = environ.get(HOME_ENV)
    return Path(raw).expanduser() if raw else Path.home() / ".tau"


def provider_import_hint(dev_home: Path, previous_home: Path | None) -> str | None:
    """Tell a first-run dev home how to acquire provider config, without doing it.

    A separate home is the point of this launcher, but it also means a fresh home
    has no model to talk to. The hint names the two ways in; it never copies a
    credential file on its own, because silently duplicating secrets into another
    directory is not this module's call to make.
    """
    if previous_home is None or previous_home == dev_home:
        return None
    dev_paths = TauPaths(home=dev_home, agents_home=dev_home / ".agents")
    if provider_settings_path(dev_paths).exists():
        return None
    previous_paths = TauPaths(home=previous_home, agents_home=previous_home / ".agents")
    if not provider_settings_path(previous_paths).exists():
        return None
    return (
        f"no provider config in {dev_home}: run `tau-rlm setup`, or import the "
        f"existing one with cp {previous_home / 'providers.json'} "
        f"{previous_home / 'credentials.json'} {dev_home}/"
    )


def prepare_dev_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    report: Callable[[str], None] | None = None,
) -> Path | None:
    """Point ``TAU_HOME`` at this track's home unless the caller chose one.

    Returns the home this entry point selected, or ``None`` when the caller
    already set ``TAU_HOME``. An unusable override fails here rather than deep
    inside path resolution, where the message stops mentioning the launcher.
    """
    env = os.environ if environ is None else environ
    explicit = env.get(HOME_ENV)
    if explicit:
        return None
    raw = env.get(DEV_HOME_ENV) or DEV_HOME_DEFAULT
    home = Path(raw).expanduser()
    if not home.is_absolute():
        raise ValueError(f"{DEV_HOME_ENV} must be an absolute path or start with '~': {raw!r}")
    previous_home = _home_of(env)
    env[HOME_ENV] = str(home)
    if report is not None:
        report(f"tau-rlm: experimental build; TAU_HOME={home}")
        hint = provider_import_hint(home, previous_home)
        if hint is not None:
            report(hint)
    return home


def main() -> None:
    """Launch Tau against this checkout's development home."""
    prepare_dev_environment(report=_report_if_interactive())
    app()


def _report_if_interactive() -> Callable[[str], None] | None:
    """Say which home is active, but never into a pipe or a log file."""
    if not sys.stderr.isatty():
        return None
    return lambda message: print(message, file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover - console-script convenience
    main()
