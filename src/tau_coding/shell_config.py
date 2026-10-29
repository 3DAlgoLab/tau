"""Durable shell execution settings for Tau terminal commands."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from tau_coding.paths import TauPaths
from tau_coding.project_trust import TrustDefault


class ShellConfigError(ValueError):
    """Raised when Tau shell settings are invalid."""


@dataclass(frozen=True, slots=True)
class LastModelPreference:
    """The provider/model pair Tau should start a new session with.

    Remembered when a session settles on a pair the user asked for -- a
    ``--provider``/``--model`` run, or an in-session switch -- and read back when
    a new session starts with no selection flags. It is a starting point, not a
    lock: ``-m``/``--provider`` and a resumed transcript still win, and a pair
    that no longer resolves (provider removed, credentials gone) is ignored
    rather than fatal.
    """

    provider_name: str
    model: str

    def to_json(self) -> dict[str, str]:
        """Serialize the pair in the shape ``settings.json`` stores."""
        return {"provider": self.provider_name, "model": self.model}


@dataclass(frozen=True, slots=True)
class ShellSettings:
    """Shell execution settings loaded from Tau home."""

    shell_command_prefix: str | None = None
    default_project_trust: TrustDefault = "ask"
    code_mode: bool = True
    """Whether sessions start with the persistent ``python_repl`` kernel tool.

    On by default, because the tool's value is in data-shaped work the shell does
    poorly. Setting it to ``false`` removes the tool from every session on this
    machine, which is the switch for a host that wants no model-written Python at
    all; a per-session host can instead pass its own ``tools`` list, which never
    gains ``python_repl`` automatically.
    """

    code_mode_recursion: bool = False
    """Whether ``python_repl`` also binds ``rlm_query`` / ``rlm_query_batched``.

    Off by default, because a recursion is not a cheaper model call: it starts a
    kernel of its own and spends a turn budget on it. Whether that is worth
    allowing on this machine is the user's cost decision; the ceilings themselves
    belong to :class:`~tau_coding.rlm.RlmLimits` and are not set here.
    """

    last_model: LastModelPreference | None = None
    """Provider/model remembered from the last session that settled on one."""

    def to_json(self) -> dict[str, object]:
        """Serialize these settings to JSON-compatible data."""
        result: dict[str, object] = {}
        if self.default_project_trust != "ask":
            result["defaultProjectTrust"] = self.default_project_trust
        if self.shell_command_prefix is not None:
            result["shellCommandPrefix"] = self.shell_command_prefix
        if not self.code_mode:
            result["codeMode"] = False
        if self.code_mode_recursion:
            result["codeModeRecursion"] = True
        if self.last_model is not None:
            result["lastModel"] = self.last_model.to_json()
        return result


def shell_settings_path(paths: TauPaths | None = None) -> Path:
    """Return the durable shell settings path."""
    return (paths or TauPaths()).home / "settings.json"


def load_shell_settings(paths: TauPaths | None = None) -> ShellSettings:
    """Load durable shell settings, falling back to built-in defaults."""
    path = shell_settings_path(paths)
    if not path.exists():
        return ShellSettings()
    try:
        raw = loads(path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ShellConfigError(f"Shell settings are not valid JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise ShellConfigError("Shell settings must be a JSON object")
    return shell_settings_from_json(raw)


def shell_settings_from_json(data: dict[str, Any]) -> ShellSettings:
    """Parse shell settings from JSON-compatible data."""
    # Read only settings this version understands so fields written by a newer
    # Tau installation cannot prevent an older installation from starting.
    if "shellCommandPrefix" in data and "shell_command_prefix" in data:
        raise ShellConfigError("Use only one of shellCommandPrefix or shell_command_prefix")
    if "codeMode" in data and "code_mode" in data:
        raise ShellConfigError("Use only one of codeMode or code_mode")
    if "codeModeRecursion" in data and "code_mode_recursion" in data:
        raise ShellConfigError("Use only one of codeModeRecursion or code_mode_recursion")
    if "lastModel" in data and "last_model" in data:
        raise ShellConfigError("Use only one of lastModel or last_model")

    raw_default = data.get("defaultProjectTrust", "ask")
    if raw_default not in {"ask", "always", "never"}:
        raise ShellConfigError("defaultProjectTrust must be ask, always, or never")

    # Code mode is on unless a settings file says otherwise; a non-boolean value is
    # refused rather than read as truthy, because "false" and "off" would silently
    # mean the opposite of what the author wrote.
    raw_code_mode = data.get("codeMode", data.get("code_mode", True))
    if not isinstance(raw_code_mode, bool):
        raise ShellConfigError("codeMode must be a boolean")

    # Recursion is opt-in in both directions: a non-boolean is refused rather
    # than read as truthy, since "true" as a string would silently enable a
    # feature that starts extra processes.
    raw_recursion = data.get("codeModeRecursion", data.get("code_mode_recursion", False))
    if not isinstance(raw_recursion, bool):
        raise ShellConfigError("codeModeRecursion must be a boolean")

    last_model = _last_model_from_json(data.get("lastModel", data.get("last_model")))

    raw_prefix = data.get("shellCommandPrefix", data.get("shell_command_prefix"))
    if raw_prefix is None:
        return ShellSettings(
            default_project_trust=raw_default,
            code_mode=raw_code_mode,
            code_mode_recursion=raw_recursion,
            last_model=last_model,
        )
    if not isinstance(raw_prefix, str):
        raise ShellConfigError("shellCommandPrefix must be a string")
    prefix = raw_prefix.strip()
    return ShellSettings(
        shell_command_prefix=prefix or None,
        default_project_trust=raw_default,
        code_mode=raw_code_mode,
        code_mode_recursion=raw_recursion,
        last_model=last_model,
    )


def _last_model_from_json(raw: object) -> LastModelPreference | None:
    """Parse the remembered startup pair, refusing anything half-shaped."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ShellConfigError("lastModel must be an object naming a provider and a model")
    provider = raw.get("provider")
    model = raw.get("model")
    if not isinstance(provider, str) or not provider.strip():
        raise ShellConfigError("lastModel.provider must be a non-empty string")
    if not isinstance(model, str) or not model.strip():
        raise ShellConfigError("lastModel.model must be a non-empty string")
    return LastModelPreference(provider_name=provider.strip(), model=model.strip())


def remember_model_choice(
    preference: LastModelPreference,
    paths: TauPaths | None = None,
) -> ShellSettings:
    """Persist the startup provider/model pair, leaving unrelated keys intact.

    Written through the raw JSON object rather than :meth:`ShellSettings.to_json`
    so that keys this version of Tau does not know about survive the write --
    the same forward-compatibility rule that makes the reader ignore them.
    """
    path = shell_settings_path(paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = loads(path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise ShellConfigError(f"Shell settings are not valid JSON: {path}") from exc
        if not isinstance(loaded, dict):
            raise ShellConfigError("Shell settings must be a JSON object")
        raw = loaded
    for key in ("lastModel", "last_model"):
        raw.pop(key, None)
    raw["lastModel"] = preference.to_json()
    _atomic_write_text(path, dumps(raw, indent=2, sort_keys=True) + "\n")
    return shell_settings_from_json(raw)


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text through a sibling temp file and atomically replace the target."""
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
            temp_file.flush()
        temp_path.replace(path)
    except Exception:
        if temp_path is not None:
            with suppress(OSError):
                temp_path.unlink()
        raise
