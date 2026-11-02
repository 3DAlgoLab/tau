"""Behavioral tests for the persistent Python kernel example extension.

The load-bearing behavior is the live namespace: a cell must be able to use a
name an earlier cell defined, across a turn boundary, and a cell that will not
yield must be cut without taking the harness down with it. Those are driven with a
real interpreter subprocess. Configuration, output caps, and the transcript
renderers are checked without a kernel, and the wiring is checked through a real
`CodingSession` with `FakeProvider`.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pi_event_helpers import assistant_done, assistant_start, tool_call_end
from tau_agent.messages import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
)
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import AgentToolResult
from tau_ai import FakeProvider
from tau_coding import CodingSession, CodingSessionConfig, TauResourcePaths

pytestmark = pytest.mark.anyio

EXTENSION_PATH = (
    Path(__file__).resolve().parents[1] / "examples" / "extensions" / "python_kernel.py"
)


def _load_extension() -> ModuleType:
    """Import the shipped example module without registering it as an extension."""
    spec = importlib.util.spec_from_file_location("tau_python_kernel_example", EXTENSION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _settings(module: ModuleType, cwd: Path, **overrides: str) -> Any:
    return module.settings_from_env({"TAU_PYTHON": sys.executable, **overrides}, cwd)


@asynccontextmanager
async def _kernel(module: ModuleType, tmp_path: Any, **overrides: str) -> AsyncIterator[Any]:
    kernel = module.PythonKernel(_settings(module, tmp_path, **overrides), tmp_path)
    try:
        yield kernel
    finally:
        await kernel.aclose()


def _resource_paths(tmp_path: Any) -> TauResourcePaths:
    paths = TauResourcePaths(
        root=tmp_path / "home-tau",
        cwd=tmp_path / "project",
        agents_root=tmp_path / "home-agents",
    )
    (tmp_path / "project").mkdir(parents=True, exist_ok=True)
    return paths


async def _load_session(tmp_path: Any, provider: FakeProvider) -> CodingSession:
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path / "project",
            resource_paths=_resource_paths(tmp_path),
            extension_paths=(EXTENSION_PATH,),
        )
    )
    runtime = session.extension_runtime
    assert not [item for item in runtime.diagnostics if item.severity == "error"]
    return session


# -- configuration -------------------------------------------------------------


def test_interpreter_ladder_prefers_an_explicit_pin(tmp_path: Any) -> None:
    module = _load_extension()
    resolved, rule, report = module.choose_interpreter(
        {"TAU_PYTHON": "/opt/side/py/bin/python3", "PATH": "/usr/bin"}, tmp_path
    )

    assert (resolved, rule) == ("/opt/side/py/bin/python3", "TAU_PYTHON pin")
    assert report.state == "pinned"


def test_interpreter_ladder_uses_a_path_interpreter_before_taus_own(tmp_path: Any) -> None:
    module = _load_extension()
    resolved, rule, _report = module.choose_interpreter(
        {"PATH": str(Path(sys.executable).parent)}, tmp_path
    )

    assert rule.startswith("PATH")
    assert Path(resolved).parent == Path(sys.executable).parent


def test_interpreter_ladder_falls_back_to_the_tau_interpreter(tmp_path: Any) -> None:
    module = _load_extension()
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    resolved, rule, _report = module.choose_interpreter({"PATH": str(empty_bin)}, tmp_path)

    assert (resolved, rule) == (sys.executable, "tau interpreter")


def test_settings_read_environment_overrides_and_tolerate_junk(tmp_path: Any) -> None:
    module = _load_extension()
    settings = module.settings_from_env(
        {
            "TAU_PYTHON": sys.executable,
            "TAU_PY_MAX_OUTPUT_CHARS": "1200",
            "TAU_PY_LINE_CHARS": "40",
            "TAU_PY_CELL_TIMEOUT_MS": "250",
            "TAU_PY_ABORT_GRACE_MS": "soon",
        },
        tmp_path,
    )

    assert settings.max_output_chars == 1200
    assert settings.line_chars == 40
    assert settings.cell_timeout_ms == 250
    assert settings.abort_grace_ms == module.DEFAULT_ABORT_GRACE_MS


# -- project venv --------------------------------------------------------------


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_venv(root: Path) -> Path:
    """Create a real venv the tests can execute, using the running interpreter."""
    subprocess.run(
        [sys.executable, "-m", "venv", str(root / ".venv")],
        check=True,
        capture_output=True,
        text=True,
    )
    return root / ".venv"


def _home_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {"HOME": str(home), "PATH": "/usr/bin:/bin"}


def test_project_venv_is_found_in_an_ancestor(tmp_path: Path) -> None:
    module = _load_extension()
    home = tmp_path / "home"
    project = home / "dev" / "repo"
    nested = project / "src" / "pkg"
    nested.mkdir(parents=True)
    _touch(project / ".git", "gitdir: x\n")
    _touch(project / ".venv" / "bin" / "python", "#!/bin/sh\n")

    venv_dir, interpreter = module.find_project_venv(nested, {"HOME": str(home), "PATH": "/bin"})

    assert venv_dir == project / ".venv"
    assert interpreter == project / ".venv" / "bin" / "python"


def test_a_home_venv_cannot_hijack_a_project(tmp_path: Path) -> None:
    """The search stops at the project root, and below `$HOME` when there is none."""
    module = _load_extension()
    home = tmp_path / "home"
    project = home / "dev" / "repo"
    project.mkdir(parents=True)
    _touch(home / ".venv" / "bin" / "python", "#!/bin/sh\n")

    assert module.find_project_venv(project, {"HOME": str(home), "PATH": "/bin"}) == (None, None)


def test_a_home_venv_counts_when_the_session_runs_in_home(tmp_path: Path) -> None:
    module = _load_extension()
    home = tmp_path / "home"
    home.mkdir(parents=True)
    _touch(home / ".venv" / "bin" / "python", "#!/bin/sh\n")

    venv_dir, _interpreter = module.find_project_venv(home, {"HOME": str(home), "PATH": "/bin"})

    assert venv_dir == home / ".venv"


def test_a_venv_at_the_git_root_is_in_scope(tmp_path: Path) -> None:
    module = _load_extension()
    home = tmp_path / "home"
    repo = home / "repo"
    nested = repo / "deep" / "nested"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    _touch(repo / ".venv" / "bin" / "python", "#!/bin/sh\n")

    venv_dir, _interpreter = module.find_project_venv(nested, {"HOME": str(home), "PATH": "/bin"})

    assert venv_dir == repo / ".venv"


def test_a_half_created_venv_reports_no_interpreter(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    _touch(project / ".venv" / "README", "half-created\n")

    venv_dir, interpreter = module.find_project_venv(project, _home_env(tmp_path))

    assert venv_dir == project / ".venv"
    assert interpreter is None


def test_an_active_venv_wins_over_path(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    python = _make_venv(project) / "bin" / "python"

    interpreter, source, report = module.choose_interpreter(_home_env(tmp_path), project)

    assert interpreter == str(python)
    assert "project venv" in source
    assert (report.state, report.venv_dir) == ("active", project / ".venv")


def test_a_pin_beats_the_project_venv(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    _make_venv(project)
    env = {**_home_env(tmp_path), "TAU_PYTHON": "/opt/special/bin/python"}

    interpreter, source, report = module.choose_interpreter(env, project)

    assert interpreter == "/opt/special/bin/python"
    assert source == "TAU_PYTHON pin"
    assert report.state == "pinned"
    assert "ignored" in report.note


def test_venv_mode_off_ignores_the_venv(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    _make_venv(project)
    env = {**_home_env(tmp_path), "TAU_PY_VENV": "off"}

    interpreter, source, report = module.choose_interpreter(env, project)

    assert report.state == "off"
    assert source.startswith("PATH") or source == "tau interpreter"
    assert interpreter != str(project / ".venv" / "bin" / "python")


def test_a_missing_venv_in_require_mode_says_uv_will_create_it(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    project.mkdir()

    _interpreter, _source, report = module.choose_interpreter(
        {**_home_env(tmp_path), "TAU_PY_VENV": "require"}, project
    )

    assert (report.state, report.venv_dir) == ("missing", None)
    assert "uv" in report.note


def test_a_broken_venv_is_not_silently_replaced(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    _touch(project / ".venv" / "pyvenv.cfg", "broken\n")

    _interpreter, _source, report = module.choose_interpreter(_home_env(tmp_path), project)

    assert report.state == "broken"
    assert "uv venv" in report.note


def test_a_lockfile_goes_through_uv_sync(tmp_path: Path) -> None:
    module = _load_extension()
    _touch(tmp_path / "uv.lock", "")
    _touch(tmp_path / "requirements.txt", "numpy\n")

    assert module.provision_steps(tmp_path / ".venv", tmp_path) == [["uv", "sync"]]


def test_requirements_files_are_installed_into_the_venv(tmp_path: Path) -> None:
    module = _load_extension()
    _touch(tmp_path / "requirements.txt", "numpy\n")

    steps = module.provision_steps(tmp_path / ".venv", tmp_path)

    assert len(steps) == 1
    assert steps[0][:4] == ["uv", "pip", "install", "--python"]
    assert steps[0][-2:] == ["-r", "requirements.txt"]


def test_a_pyproject_without_a_lock_still_uses_uv_sync(tmp_path: Path) -> None:
    module = _load_extension()
    _touch(tmp_path / "pyproject.toml", "[project]\nname='x'\n")

    assert module.provision_steps(tmp_path / ".venv", tmp_path) == [["uv", "sync"]]


def test_no_manifest_means_no_dependency_step(tmp_path: Path) -> None:
    module = _load_extension()

    assert module.provision_steps(tmp_path / ".venv", tmp_path) == []


async def test_uv_provisions_a_working_project_venv(tmp_path: Path) -> None:
    """The venv uv creates is the one the kernel then resolves to."""
    module = _load_extension()
    env = {"HOME": str(tmp_path / "home"), "PATH": os.environ.get("PATH", "")}
    if module.uv_binary(env) is None:
        pytest.skip("uv is not installed")
    project = tmp_path / "proj"
    _touch(project / "requirements.txt", "")

    report = await module.provision_venv(project, env)

    assert report.ok, report.render()
    assert report.state == "provisioned"
    interpreter, _source, resolved = module.choose_interpreter(env, project)
    assert resolved.state == "active"
    check = subprocess.run(
        [interpreter, "-c", "import sys; print(sys.prefix)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr
    assert Path(check.stdout.strip()).resolve() == (project / ".venv").resolve()

    again = await module.provision_venv(project, env)

    assert (again.ok, again.state) == (True, "reused")


def test_provisioning_reports_a_missing_uv(tmp_path: Path) -> None:
    module = _load_extension()
    project = tmp_path / "proj"
    project.mkdir()

    report = module.provision_venv_sync(project, {"PATH": str(tmp_path)})

    assert report.ok is False
    assert report.state == "uv missing"


def test_venv_report_renders_one_status_line() -> None:
    module = _load_extension()

    assert (
        module.VenvReport("prefer", "missing", None, None).render()
        == "missing: none at or above cwd (mode=prefer)"
    )
    venv_dir = Path("/repo/.venv")
    found = module.VenvReport("prefer", "active", venv_dir, venv_dir / "bin" / "python")
    assert found.render() == "active: /repo/.venv (mode=prefer)"


# -- output shape --------------------------------------------------------------


def test_output_caps_name_every_cut() -> None:
    module = _load_extension()

    line = module.clip_line("x" * 50, 20)
    assert line.startswith("x" * 20)
    assert "30 chars cut" in line

    text = "\n".join(["head"] + ["y" * 300] * 4 + ["tail"])
    capped = module.clip_output(text, 400, 300)
    assert capped.startswith("head")
    assert capped.endswith("tail")
    assert "chars cut" in capped
    assert len(capped) < len(text)


def test_compose_reports_kernel_notices() -> None:
    module = _load_extension()

    cut = module.compose_cell_text(
        module.CellOutcome(status="aborted", names=["df"], timed_out=True), 2000, 200
    )
    assert "<tau_python_cell_timeout>" in cut
    assert "status: timeout" in cut
    assert "names survived: 1 names: df" in cut

    reaped = module.compose_cell_text(
        module.CellOutcome(status="aborted", reaped=True, noticed_reset=True), 2000, 200
    )
    assert "<tau_python_kernel_reaped>" in reaped
    assert "<tau_python_kernel_reset>" in reaped
    assert "namespace: empty" in reaped

    failed = module.compose_cell_text(
        module.CellOutcome(
            status="error",
            stdout="partial progress\n",
            error={
                "name": "ValueError",
                "message": "bad row",
                "stack": ["Traceback", "ValueError"],
            },
            names=["df"],
        ),
        2000,
        200,
    )
    assert "status: error" in failed
    assert "partial progress" in failed
    assert "error: ValueError" in failed


def test_transcript_renderers_stay_compact() -> None:
    module = _load_extension()

    assert module._render_call({"code": "\n\ndef f():\n    return 1\n"}) == "def f():"
    assert module._render_call({"code": "z" * 120}) == f"{'z' * 80} ..."

    result = AgentToolResult(
        content=[],
        details={"status": "ok", "names": ["df", "x"]},
    )
    compact = module._render_result(result, expanded=False)
    assert compact is not None
    assert "2 live names" in compact
    assert "\n" not in compact
    assert module._render_result(result, expanded=True) is None


# -- a real interpreter --------------------------------------------------------


async def test_namespace_survives_from_cell_to_cell(tmp_path: Any) -> None:
    module = _load_extension()
    async with _kernel(module, tmp_path) as kernel:
        first = await kernel.exec_cell("rows = [3, 4]\nprint('loaded')\nlen(rows)")
        assert first.status == "ok"
        assert first.stdout == "loaded\n"
        assert first.value == "2"
        assert first.names == ["rows"]

        second = await kernel.exec_cell("sum(rows) * 2")
        assert second.value == "14"
        assert kernel.state == "ready"
        assert kernel.pid is not None


async def test_a_failing_cell_keeps_the_namespace(tmp_path: Any) -> None:
    module = _load_extension()
    async with _kernel(module, tmp_path) as kernel:
        await kernel.exec_cell("keep = 'held'\n")

        failed = await kernel.exec_cell("import definitely_not_installed")
        assert failed.status == "error"
        assert failed.error is not None
        assert failed.error["name"] == "ModuleNotFoundError"
        assert failed.names == ["keep"]

        after = await kernel.exec_cell("keep")
        assert after.status == "ok"
        assert after.value == "'held'"


async def test_a_deadline_interrupts_a_cell_and_keeps_its_state(tmp_path: Any) -> None:
    module = _load_extension()
    async with _kernel(module, tmp_path) as kernel:
        await kernel.exec_cell("keep = 'held'\n")

        cut = await kernel.exec_cell(
            "print('progress')\nimport time\nwhile True:\n    time.sleep(0.05)",
            timeout_ms=600,
            grace_ms=2_000,
        )
        assert cut.status == "aborted"
        assert cut.timed_out is True
        assert cut.reaped is False
        assert "progress" in cut.stdout

        text = module.compose_cell_text(cut, 2000, 200)
        assert "<tau_python_cell_timeout>" in text
        assert "keep" in text
        assert (await kernel.exec_cell("keep")).value == "'held'"


async def test_an_uninterruptible_cell_is_reaped_and_the_next_reports_reset(tmp_path: Any) -> None:
    module = _load_extension()
    async with _kernel(module, tmp_path) as kernel:
        await kernel.exec_cell("keep = 'held'\n")

        wedged = await kernel.exec_cell(
            "import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(5)",
            timeout_ms=400,
            grace_ms=300,
        )
        assert wedged.reaped is True
        assert wedged.timed_out is True
        assert kernel.state == "stopped"

        rebuilt = await kernel.exec_cell("1 + 1")
        assert rebuilt.status == "ok"
        assert rebuilt.value == "2"
        assert rebuilt.noticed_reset is True
        assert rebuilt.names == []
        assert "<tau_python_kernel_reset>" in module.compose_cell_text(rebuilt, 2000, 200)


async def test_cell_source_survives_the_protocol_pipe(tmp_path: Any) -> None:
    module = _load_extension()
    async with _kernel(module, tmp_path) as kernel:
        cell = await kernel.exec_cell("print('재질 20° Ø8 \\u00e9\\u4e2d')")

        assert cell.status == "ok"
        assert cell.stdout == "재질 20° Ø8 é中\n"


async def test_closing_the_kernel_stops_the_bridge_process(tmp_path: Any) -> None:
    module = _load_extension()
    kernel = module.PythonKernel(_settings(module, tmp_path), tmp_path)
    await kernel.exec_cell("1")
    pid = kernel.pid
    assert pid is not None

    await kernel.aclose()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert kernel.state == "stopped"


async def test_a_missing_interpreter_is_reported_as_a_tool_error(tmp_path: Any) -> None:
    module = _load_extension()
    settings = module.KernelSettings(
        interpreter=str(tmp_path / "nope" / "python3"),
        interpreter_source="TAU_PYTHON pin",
        max_output_chars=1000,
        line_chars=100,
        cell_timeout_ms=0,
        abort_grace_ms=100,
        venv=module.VenvReport("prefer", "pinned", None, None),
    )
    kernel = module.PythonKernel(settings, tmp_path)

    with pytest.raises(OSError):
        await kernel.exec_cell("1")

    await kernel.aclose()


# -- wiring through a coding session -------------------------------------------


async def test_session_registers_the_tool_and_its_command(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAU_PYTHON", sys.executable)
    session = await _load_session(tmp_path, FakeProvider([]))

    assert "python" in [tool.name for tool in session.tools]

    status = session.handle_command("/python")
    assert status.handled is True
    assert status.message is not None
    assert "python:" in status.message
    assert "output cap:" in status.message
    assert "no snapshots" in status.message

    reset = session.handle_command("/python reset")
    assert reset.handled is True
    assert reset.message is not None
    assert "fresh, empty" in reset.message

    await session.aclose()


async def test_cells_across_turns_share_one_namespace(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAU_PYTHON", sys.executable)

    first = ToolCall(id="call-1", name="python", arguments={"code": "rows = [1, 2, 3]\nlen(rows)"})
    first_message = AssistantMessage(content=[first], model="fake")
    second = ToolCall(id="call-2", name="python", arguments={"code": "sum(rows)"})
    second_message = AssistantMessage(content=[second], model="fake")
    final = AssistantMessage(content="three items, summing to six", model="fake")
    provider = FakeProvider(
        [
            [assistant_start(), tool_call_end(first), assistant_done(first_message, "toolUse")],
            [assistant_start(), tool_call_end(second), assistant_done(second_message, "toolUse")],
            [assistant_start(), assistant_done(final)],
        ]
    )
    session = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("count these rows")]

    results = [message for message in session.messages if isinstance(message, ToolResultMessage)]
    assert len(results) == 2
    assert "value: 3" in results[0].text
    assert "value: 6" in results[1].text
    assert results[1].details is not None
    assert "rows" in str(results[1].details)

    await session.aclose()
