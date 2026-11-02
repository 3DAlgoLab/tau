"""Clipboard writes for frontends that do not own a terminal clipboard.

The TUI has a richer path — Textual knows about OSC 52 and the terminal it is
sitting in — so it keeps its own copy chain. Everything else (print mode, and any
future frontend) uses this module. It shells out to the clipboard tool the machine
actually has instead of assuming one, and it never writes terminal escape codes: a
frontend may own the screen, and an escape sequence from underneath it corrupts
that render.

Security note: the subprocess calls below pass a fixed argv with no shell, so
clipboard *content* is data, never a command line. The executables themselves come
from ``PATH`` and are resolved with :func:`shutil.which` before use.
"""

from __future__ import annotations

import shutil
import subprocess  # trusted stdlib use: fixed argv, stdin pipe, no shell=True
from contextlib import suppress
from dataclasses import dataclass

__all__ = ["ClipboardOutcome", "copy_to_clipboard", "format_clipboard_result"]

# A clipboard tool that does not answer promptly is a broken tool, not a slow one.
_TIMEOUT_SECONDS = 5.0

# Tried in order: Wayland, macOS, WSL, X11, then Windows' own clip.
_COMMAND_CANDIDATES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("wl-copy", ("wl-copy",)),
    ("pbcopy", ("pbcopy",)),
    ("clip.exe", ("clip.exe",)),
    ("xclip", ("xclip", "-selection", "clipboard")),
    ("xsel", ("xsel", "--clipboard", "--input")),
    ("clip", ("clip",)),
)


@dataclass(frozen=True, slots=True)
class ClipboardOutcome:
    """Whether a clipboard write landed, and how or why not."""

    ok: bool
    backend: str | None = None
    detail: str | None = None


def copy_to_clipboard(text: str) -> ClipboardOutcome:
    """Copy ``text`` using the first clipboard backend this machine offers.

    Every backend is best-effort: a missing tool, a tool that refuses, or a tool
    that times out moves on to the next candidate rather than raising. Callers
    report the outcome; they do not have to handle clipboard exceptions.
    """
    if not text:
        return ClipboardOutcome(ok=False, detail="There is no text to copy.")

    with suppress(Exception):
        # Optional dependency; when present it already knows how to reach the
        # platform clipboard, including falling back to a tool below.
        import pyperclip  # type: ignore[import-untyped]

        pyperclip.copy(text)
        return ClipboardOutcome(ok=True, backend="pyperclip")

    for backend, argv in _COMMAND_CANDIDATES:
        if shutil.which(argv[0]) is None:
            continue
        try:
            subprocess.run(
                argv,
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=_TIMEOUT_SECONDS,
                check=True,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        return ClipboardOutcome(ok=True, backend=backend)

    return ClipboardOutcome(
        ok=False,
        detail=(
            "Could not reach a clipboard. Install a clipboard tool "
            "(wl-copy, xclip, or pbcopy) or copy the text from the transcript."
        ),
    )


def format_clipboard_result(outcome: ClipboardOutcome, *, characters: int) -> str:
    """Return the one-line status a frontend should show for a clipboard write."""
    if not outcome.ok:
        return outcome.detail or "Could not copy to the clipboard."
    return f"Copied {characters:,} characters to the clipboard ({outcome.backend})."
