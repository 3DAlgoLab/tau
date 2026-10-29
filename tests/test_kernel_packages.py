"""Package environment tests: interpreter identity, layout, and the resolvers.

No test here touches the network. What this module owns is the part a resolver
cannot be trusted with: which interpreter the kernel will run, which site
directory that interpreter searches, and which resolver gets asked to fill it.
Fake executables that write a marker where they were told to install pin all of
that without depending on a package index. Script bodies below avoid escape
sequences on purpose: they are written verbatim into files and executed.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from tau_coding.kernel.packages import (
    MAX_SPECS,
    PackageInstaller,
    PackageInstallError,
    probe_interpreter,
    validate_specs,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


FAKE_UV = """#!/bin/sh
if [ "$FAKE_UV_FAIL" = "1" ]; then echo "uv: no version matches" >&2; exit 1; fi
if [ "$FAKE_UV_SLEEP" = "1" ]; then sleep 5; exit 0; fi
target=""
previous=""
for arg in "$@"; do
    if [ "$previous" = "--target" ]; then target="$arg"; fi
    previous="$arg"
done
mkdir -p "$target"
echo "MARK = 'from uv'" > "$target/installed_probe.py"
echo "Installed 1 package"
"""

FAKE_PYTHON = """#!/usr/bin/env python3
import os
import pathlib
import sys


def emit(path, text):
    path.write_text(text + chr(10), encoding="utf-8")


if "-m" in sys.argv and "pip" in sys.argv:
    emit(pathlib.Path(os.environ["PIP_LOG"]), "pip ran")
    if os.environ.get("FAKE_PIP_FAIL") == "1":
        print("no matching distribution for installed-probe")
        raise SystemExit(1)
    target = pathlib.Path(sys.argv[sys.argv.index("--target") + 1])
    target.mkdir(parents=True, exist_ok=True)
    emit(target / "installed_probe.py", "MARK = 'from the interpreter pip'")
    print("Successfully installed installed-probe-1.0")
    raise SystemExit(0)
raise SystemExit(2)
"""


def _script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _installer(tmp_path: Path, uv: Path | None = None) -> PackageInstaller:
    """An installer pointed at fakes only, so no test can reach a package index.

    Naming a uv that does not exist is how a test says "no uv here": the dev
    machine usually has one, and letting the chain find it would mean installing
    from PyPI inside a test run.
    """
    python = _script(tmp_path / "fake-python", FAKE_PYTHON)
    return PackageInstaller(
        executable=str(python),
        target=tmp_path / "site",
        uv_executable=str(uv) if uv is not None else str(tmp_path / "no-such-uv"),
    )


# -- interpreter identity ----------------------------------------------------
#
# The site directory is derived from the interpreter, so an install and an
# import cannot disagree about which ABI they mean. Both halves are cheap to get
# wrong: a kernel configured onto a different interpreter would otherwise be
# handed the host's package directory and fail to import what it installed.


async def test_an_interpreter_names_its_implementation_and_version(tmp_path: Path) -> None:
    info = await probe_interpreter(sys.executable)
    assert info.implementation == sys.implementation.name
    assert info.version == ".".join(str(part) for part in sys.version_info[:2])
    site = info.site_dir(tmp_path / "kernel-site")
    assert site.parent == tmp_path / "kernel-site"
    assert info.implementation in site.name and info.version in site.name


async def test_the_same_interpreter_always_maps_to_one_directory(tmp_path: Path) -> None:
    first = await probe_interpreter(sys.executable)
    second = await probe_interpreter(sys.executable)
    assert first == second
    assert first.site_dir(tmp_path) == second.site_dir(tmp_path)


async def test_an_interpreter_that_cannot_be_run_is_reported(tmp_path: Path) -> None:
    with pytest.raises(PackageInstallError, match="cannot run"):
        await probe_interpreter(str(tmp_path / "not-an-interpreter"))


async def test_an_executable_that_is_not_python_is_refused() -> None:
    # /bin/true runs, exits 0, and says nothing about itself. An executable that
    # cannot answer this question cannot run a cell either, so the kernel stops
    # here instead of failing inside a cell later.
    with pytest.raises(PackageInstallError, match="reported a malformed interpreter identity"):
        await probe_interpreter("/bin/true")


# -- requirement specs -------------------------------------------------------


def test_specs_survive_the_shapes_a_model_actually_sends() -> None:
    assert validate_specs(["numpy", "scipy==1.15.0", "rich>=13"]) == (
        "numpy",
        "scipy==1.15.0",
        "rich>=13",
    )


# Specs travel as separate arguments and never through a shell, so the only real
# risk is a spec that reads as an option of the resolver itself.
@pytest.mark.parametrize(
    "specs",
    [
        [],
        [""],
        ["   "],
        ["--target=/tmp/elsewhere"],
        ["-c"],
        ["numpy" + chr(0) + "tail"],
        ["x" * 201],
        [str(index) for index in range(MAX_SPECS + 1)],
        [None],
    ],
    ids=[
        "empty list",
        "empty spec",
        "blank spec",
        "option that redirects the target",
        "leading dash",
        "control character",
        "over-long",
        "too many",
        "not a string",
    ],
)
def test_a_spec_that_could_read_as_more_than_a_name_is_refused(specs: list[object]) -> None:
    with pytest.raises(PackageInstallError):
        validate_specs(specs)


# -- the resolver chain ------------------------------------------------------


async def test_uv_is_preferred_and_pip_is_never_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = _script(tmp_path / "fake-uv", FAKE_UV)
    pip_log = tmp_path / "pip-ran"
    monkeypatch.setenv("PIP_LOG", str(pip_log))

    report = await _installer(tmp_path, uv).install(["installed-probe"])

    assert report.ok, report.output
    assert report.resolver == str(uv)
    site = tmp_path / "site"
    assert (site / "installed_probe.py").read_text(encoding="utf-8").strip() == "MARK = 'from uv'"
    assert not pip_log.exists()


async def test_the_interpreters_own_pip_catches_a_failing_uv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = _script(tmp_path / "fake-uv", FAKE_UV)
    pip_log = tmp_path / "pip-ran"
    monkeypatch.setenv("PIP_LOG", str(pip_log))
    monkeypatch.setenv("FAKE_UV_FAIL", "1")

    report = await _installer(tmp_path, uv).install(["installed-probe"])

    assert report.ok, report.output
    assert report.resolver != str(uv)
    assert pip_log.read_text(encoding="utf-8").strip() == "pip ran"
    site = tmp_path / "site"
    marker = (site / "installed_probe.py").read_text(encoding="utf-8")
    assert "from the interpreter pip" in marker


async def test_a_resolver_that_is_not_there_is_skipped_before_it_runs(tmp_path: Path) -> None:
    installer = PackageInstaller(
        executable=sys.executable,
        target=tmp_path / "site",
        uv_executable=str(tmp_path / "no-such-uv"),
    )
    commands = installer.resolver_commands(["numpy"])
    assert [command[0] for command in commands] == [sys.executable]
    assert commands[0][:4] == (sys.executable, "-m", "pip", "install")


async def test_every_resolver_failing_is_a_report_not_an_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = _script(tmp_path / "fake-uv", FAKE_UV)
    monkeypatch.setenv("PIP_LOG", str(tmp_path / "pip-ran"))
    monkeypatch.setenv("FAKE_UV_FAIL", "1")
    monkeypatch.setenv("FAKE_PIP_FAIL", "1")

    report = await _installer(tmp_path, uv).install(["installed-probe"])

    assert not report.ok
    assert report.error is not None and "no resolver" in report.error
    # Both attempts are quoted back: the model needs to know which one to blame,
    # and the resolver's own lines are the only part that explains a bad spec.
    assert str(uv) in report.output and "no version matches" in report.output
    assert "no matching distribution" in report.output
    assert not (tmp_path / "site" / "installed_probe.py").exists()


async def test_a_resolver_that_outstays_its_timeout_is_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uv = _script(tmp_path / "fake-uv", FAKE_UV)
    monkeypatch.setenv("PIP_LOG", str(tmp_path / "pip-ran"))
    monkeypatch.setenv("FAKE_UV_SLEEP", "1")

    installer = PackageInstaller(
        executable=str(_script(tmp_path / "fake-python", FAKE_PYTHON)),
        target=tmp_path / "site",
        timeout_seconds=0.5,
        uv_executable=str(uv),
    )
    report = await installer.install(["installed-probe"])

    assert not report.ok
    assert "exceeded" in report.output and "0.5s" in report.output
    # The budget is the whole install, not each attempt, so a resolver that
    # burned it is not followed by a second one that would only be killed too.
    assert "budget was spent" in report.output
    assert not (tmp_path / "pip-ran").exists()


async def test_the_site_directory_survives_a_failed_install(tmp_path: Path) -> None:
    # The kernel's PYTHONPATH names this directory, so it has to exist before the
    # child starts; a failed install must not take it away again.
    report = await _installer(tmp_path).install(["installed-probe"])

    assert not report.ok
    assert (tmp_path / "site").is_dir()
