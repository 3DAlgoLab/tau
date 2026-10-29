"""The package environment a code-mode kernel installs third-party code into.

The kernel runs on the interpreter that started Tau. After a ``uv tool install``
that interpreter is a sealed tool environment: no pip, and none of the packages
a data-shaped task reaches for. That is the difference between a kernel the model
uses and a kernel it routes around -- the session this exists because of checked
``numpy`` through ``bash``, found it, then found the kernel did not have it, and
quietly gave up on the persistent namespace.

Two pieces fix that. A kernel with a package environment starts with a
version-keyed site directory already on its ``PYTHONPATH``, and this module
installs into that directory using whatever resolver the machine has: ``uv`` when
it is on PATH, otherwise the interpreter's own pip. Installing into a directory
rather than into the interpreter keeps the interpreter pristine -- an environment
is a directory the user can delete, and a different interpreter gets its own --
while still surviving a kernel restart, because the directory outlives the child
process.

The directory is keyed by implementation and minor version because wheels are:
a cp313 build imported by a 3.12 kernel fails in a way that reads like a bug in
the package rather than a mismatch the host created.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "DEFAULT_INSTALL_TIMEOUT_SECONDS",
    "DEFAULT_INSTALL_TIMEOUT_SECONDS",
    "MAX_INSTALL_OUTPUT_BYTES",
    "MAX_SPEC_CHARS",
    "MAX_SPECS",
    "InstallReport",
    "InterpreterInfo",
    "PackageInstallError",
    "PackageInstaller",
    "probe_interpreter",
    "validate_specs",
]

MAX_SPECS = 20
MAX_SPEC_CHARS = 200
DEFAULT_INSTALL_TIMEOUT_SECONDS = 180.0
# Below this, a further resolver would only be started to be killed: the note
# says the budget ran out instead of pretending another attempt was possible.
MIN_RESOLVER_ATTEMPT_SECONDS = 1.0
MAX_INSTALL_OUTPUT_BYTES = 8 * 1024

_PROBE_SNIPPET = (
    "import sys; "
    "print(sys.implementation.name); "
    "print('.'.join(str(part) for part in sys.version_info[:2]))"
)


class PackageInstallError(RuntimeError):
    """Raised when a package request is unusable or no resolver could run."""


@dataclass(frozen=True, slots=True)
class InterpreterInfo:
    """What an interpreter reported about itself.

    Only the identity matters here: it decides which site directory the kernel
    sees, so an install and an import can never disagree about the ABI.
    """

    executable: str
    implementation: str
    version: str

    def site_dir(self, base: Path) -> Path:
        """Return the version-keyed package directory under ``base``."""
        return base / f"{self.implementation}-{self.version}"


@dataclass(frozen=True, slots=True)
class InstallReport:
    """What one install attempt did, including the commands it tried.

    ``output`` is the resolver's own tail, bounded, because a failed resolve is
    usually explained by lines only the resolver wrote.
    """

    ok: bool
    target: Path
    specs: tuple[str, ...]
    commands: tuple[tuple[str, ...], ...]
    output: str
    error: str | None = None
    seconds: float = 0.0
    resolver: str | None = None


_probe_cache: dict[str, InterpreterInfo] = {}


async def probe_interpreter(executable: str) -> InterpreterInfo:
    """Ask ``executable`` what it is, so the package directory matches its ABI.

    The host needs the answer *before* the kernel starts, because the site
    directory is part of the child's ``PYTHONPATH``. One subprocess per distinct
    interpreter is cheap and cached; a kernel is started a handful of times per
    session, and an interpreter that cannot answer this question cannot run the
    kernel either.
    """
    cached = _probe_cache.get(executable)
    if cached is not None:
        return cached
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "-B",
            "-c",
            _PROBE_SNIPPET,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise PackageInstallError(f"cannot run {executable!r}: {exc}") from exc
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace").strip() or "no output"
        raise PackageInstallError(
            f"{executable!r} reported no interpreter identity (exit {process.returncode}): {detail}"
        )
    lines = [line.strip() for line in stdout.decode("utf-8", errors="replace").splitlines()]
    if len(lines) < 2 or not lines[0] or not lines[1]:
        raise PackageInstallError(f"{executable!r} reported a malformed interpreter identity")
    info = InterpreterInfo(executable=executable, implementation=lines[0], version=lines[1])
    _probe_cache[executable] = info
    return info


def validate_specs(specs: Sequence[object]) -> tuple[str, ...]:
    """Return usable requirement specs, or explain the first refusal.

    Specs are passed to the resolver as separate arguments, so the only real
    injection risk is a spec that starts out looking like an option. Control
    characters are refused as well: they mark a spec assembled by accident.
    """
    if not isinstance(specs, Sequence) or isinstance(specs, str | bytes):
        raise PackageInstallError('install expects a list of package specs, such as ["numpy"]')
    if not specs:
        raise PackageInstallError("install was given an empty list of packages")
    if len(specs) > MAX_SPECS:
        raise PackageInstallError(f"install accepts at most {MAX_SPECS} packages per call")
    checked: list[str] = []
    for raw in specs:
        if not isinstance(raw, str):
            raise PackageInstallError("every package spec must be a string")
        spec = raw.strip()
        if not spec:
            raise PackageInstallError("a package spec cannot be empty")
        if len(spec) > MAX_SPEC_CHARS:
            raise PackageInstallError(
                f"package spec {spec[:40]!r} is longer than {MAX_SPEC_CHARS} characters"
            )
        if spec.startswith("-"):
            raise PackageInstallError(f"package spec {spec!r} cannot start with a dash")
        if any(ord(char) < 32 for char in spec):
            raise PackageInstallError(f"package spec {spec[:40]!r} contains a control character")
        checked.append(spec)
    return tuple(checked)


class PackageInstaller:
    """Install requirement specs into one persistent site directory.

    The resolver choice is the machine's, not the model's: ``uv`` is preferred
    because it resolves against the kernel's interpreter without needing pip in
    that interpreter, and a bare ``uv tool install`` environment has no pip at
    all. The pip fallback keeps a plain-CPython install working.
    """

    def __init__(
        self,
        *,
        executable: str,
        target: Path,
        timeout_seconds: float = DEFAULT_INSTALL_TIMEOUT_SECONDS,
        uv_executable: str | None = None,
    ) -> None:
        self._executable = executable
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._uv_executable = uv_executable

    @property
    def target(self) -> Path:
        """The site directory packages are installed into."""
        return self._target

    def resolver_commands(self, specs: Sequence[str]) -> tuple[tuple[str, ...], ...]:
        """Return the resolvers to try, in order, for ``specs``."""
        target = str(self._target)
        commands: list[tuple[str, ...]] = []
        # which() normalises an explicit path too: a configured uv that is not
        # on this machine should fall through to pip, not burn a failed spawn.
        wanted = self._uv_executable if self._uv_executable is not None else "uv"
        uv = shutil.which(wanted)
        if uv:
            commands.append(
                (uv, "pip", "install", "--python", self._executable, "--target", target, *specs)
            )
        commands.append((self._executable, "-m", "pip", "install", "--target", target, *specs))
        return tuple(commands)

    async def install(self, specs: Sequence[object]) -> InstallReport:
        """Install ``specs``, trying each resolver in turn."""
        checked = validate_specs(specs)
        try:
            self._target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise PackageInstallError(f"cannot create {self._target}: {exc}") from exc
        commands = self.resolver_commands(checked)
        started = time.perf_counter()
        # One budget for the whole install, shared across resolvers: the caller
        # named a wait, and trying a second resolver after spending it would
        # silently double the time a tool call can block.
        deadline = started + self._timeout_seconds
        notes: list[str] = []
        for index, command in enumerate(commands):
            remaining = deadline - time.perf_counter()
            if index and remaining < MIN_RESOLVER_ATTEMPT_SECONDS:
                notes.append(
                    f"no time left for {command[0]}: the "
                    f"{self._timeout_seconds:g}s install budget was spent earlier"
                )
                break
            ok, output = await self._run(command, remaining)
            if ok:
                return InstallReport(
                    ok=True,
                    target=self._target,
                    specs=checked,
                    commands=commands,
                    output=output,
                    seconds=round(time.perf_counter() - started, 1),
                    resolver=command[0],
                )
            notes.append(f"{command[0]}: {output.strip() or 'the resolver exited unsuccessfully'}")
        return InstallReport(
            ok=False,
            target=self._target,
            specs=checked,
            commands=commands,
            output="\n".join(notes),
            error="no resolver on this machine could install the request",
            seconds=round(time.perf_counter() - started, 1),
        )

    async def _run(self, argv: Sequence[str], timeout_seconds: float) -> tuple[bool, str]:
        """Run one resolver command within ``timeout_seconds`` and report its tail."""
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            return False, f"cannot start {argv[0]!r}: {exc}"
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            return (
                False,
                f"the install exceeded {timeout_seconds:g}s and was killed; "
                "retry with fewer packages",
            )
        text = stdout.decode("utf-8", errors="replace")
        encoded = text.encode("utf-8")
        if len(encoded) > MAX_INSTALL_OUTPUT_BYTES:
            text = encoded[-MAX_INSTALL_OUTPUT_BYTES:].decode("utf-8", errors="ignore")
        return process.returncode == 0, text
