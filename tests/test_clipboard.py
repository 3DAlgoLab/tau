"""Tests for the frontend-independent clipboard path.

The load-bearing guarantees are that a clipboard write never becomes a shell
command, that a missing or broken tool falls through to the next candidate rather
than raising, and that a machine with no clipboard at all produces a message a
user can act on instead of a stack trace.
"""

from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tau_coding import clipboard
from tau_coding.clipboard import copy_to_clipboard, format_clipboard_result


class _RunRecorder:
    """Stand-in for ``subprocess.run`` that records argv and can inject failures."""

    def __init__(self, failing: tuple[tuple[str, ...], ...] = ()) -> None:
        self.calls: list[tuple[tuple[str, ...], str, dict[str, Any]]] = []
        self.failing = set(failing)

    def __call__(
        self,
        argv: list[str],
        *,
        input: bytes,  # noqa: A002 - mirrors subprocess.run's parameter name
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[bytes]:
        command = tuple(argv)
        self.calls.append((command, input.decode("utf-8"), kwargs))
        if command in self.failing:
            raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    @property
    def backends(self) -> tuple[str, ...]:
        return tuple(command[0] for command, _text, _kwargs in self.calls)


def _patch_tools(
    monkeypatch: pytest.MonkeyPatch, available: tuple[str, ...], recorder: _RunRecorder
) -> None:
    """Make only ``available`` clipboard binaries resolvable, with a fake runner."""
    monkeypatch.setattr(
        clipboard,
        "shutil",
        SimpleNamespace(
            which=lambda name: path if (path := f"/usr/bin/{name}") in available else None
        ),
    )
    monkeypatch.setattr(
        clipboard,
        "subprocess",
        SimpleNamespace(
            run=recorder,
            DEVNULL=subprocess.DEVNULL,
            SubprocessError=subprocess.SubprocessError,
        ),
    )


@pytest.fixture(autouse=True)
def _no_pyperclip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the optional dependency out of every test unless it adds one back."""
    monkeypatch.setitem(sys.modules, "pyperclip", None)


def test_empty_text_is_reported_without_touching_the_clipboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, ("/usr/bin/wl-copy",), recorder)

    outcome = copy_to_clipboard("")

    assert outcome.ok is False
    assert "no text" in (outcome.detail or "")
    assert recorder.calls == []


def test_pyperclip_is_preferred_when_it_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied: list[str] = []
    fake = ModuleType("pyperclip")
    fake.copy = lambda text: copied.append(text)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake)
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, ("/usr/bin/wl-copy",), recorder)

    outcome = copy_to_clipboard("hello")

    assert (outcome.ok, outcome.backend) == (True, "pyperclip")
    assert copied == ["hello"]
    assert recorder.calls == []


def test_a_broken_pyperclip_falls_through_to_a_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("pyperclip")

    def _boom(_text: str) -> None:
        raise RuntimeError("no clipboard found")

    fake.copy = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyperclip", fake)
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, ("/usr/bin/wl-copy",), recorder)

    outcome = copy_to_clipboard("hello")

    assert (outcome.ok, outcome.backend) == (True, "wl-copy")
    assert recorder.calls[0][1] == "hello"


def test_the_first_available_tool_wins_and_receives_the_text_on_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, ("/usr/bin/xclip", "/usr/bin/xsel"), recorder)

    outcome = copy_to_clipboard("first one wins")

    assert (outcome.ok, outcome.backend) == (True, "xclip")
    argv, text, kwargs = recorder.calls[0]
    assert argv == ("xclip", "-selection", "clipboard")
    assert text == "first one wins"
    # A clipboard tool is fed through a pipe, never a shell, and never gets to
    # hold the terminal's stdout.
    assert "shell" not in kwargs
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["timeout"] > 0


def test_a_refusing_tool_moves_to_the_next_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RunRecorder(failing=(("wl-copy",),))
    _patch_tools(monkeypatch, ("/usr/bin/wl-copy", "/usr/bin/pbcopy"), recorder)

    outcome = copy_to_clipboard("try again")

    assert (outcome.ok, outcome.backend) == (True, "pbcopy")
    assert recorder.backends == ("wl-copy", "pbcopy")


def test_a_machine_with_no_tool_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, (), recorder)

    outcome = copy_to_clipboard("nowhere to go")

    assert outcome.ok is False
    assert outcome.backend is None
    assert recorder.calls == []
    detail = outcome.detail or ""
    assert "wl-copy" in detail and "xclip" in detail


def test_clipboard_text_is_data_and_never_a_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _RunRecorder()
    _patch_tools(monkeypatch, ("/usr/bin/wl-copy",), recorder)
    hostile = "'; rm -rf / #$(whoami)"

    outcome = copy_to_clipboard(hostile)

    assert outcome.ok is True
    argv, text, _kwargs = recorder.calls[0]
    assert argv == ("wl-copy",)
    assert text == hostile
    assert not any(hostile in " ".join(command) for command, _t, _k in recorder.calls)


def test_format_clipboard_result_names_the_count_and_backend() -> None:
    ok = format_clipboard_result(
        clipboard.ClipboardOutcome(ok=True, backend="wl-copy"), characters=1234
    )
    failed = format_clipboard_result(
        clipboard.ClipboardOutcome(ok=False, detail="Could not reach a clipboard."), characters=5
    )

    assert ok == "Copied 1,234 characters to the clipboard (wl-copy)."
    assert failed == "Could not reach a clipboard."
