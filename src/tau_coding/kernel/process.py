"""The option-A kernel: a persistent Python process driven over a socket.

The host owns a localhost server and the kernel child connects to it, which keeps
one mechanism in both directions: framed JSON, with the child sending output and
asking for completions, and the host sending cells and control requests. A socket
rather than the child's standard streams matters more than it looks -- the model's
own ``print`` output and our control frames must never share a byte stream, and
a sub-call the model issues from inside a cell has to reach the host *while* that
cell is still running.

Two failure properties are designed in rather than papered over:

* A cell that overruns its budget is killed with its process, because Python
  offers no way to interrupt ``exec`` from outside. The result says
  ``state_lost`` so the model learns the namespace is gone.
* The port is reachable by any local process, so the connection is only adopted
  after a token handshake. A foreign connection is closed without being served.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import secrets
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tau_coding.kernel.frames import FrameError, read_frame, write_frame
from tau_coding.kernel.protocol import (
    ENV_HOST,
    ENV_PORT,
    ENV_TOKEN,
    HOST_DEFAULT,
    OP_BIND,
    OP_DONE,
    OP_EXEC,
    OP_HELLO,
    OP_OUT,
    OP_READY,
    OP_RESET,
    OP_SHUTDOWN,
    OP_STARTUP,
    OP_STATE,
    OP_SUBCALL,
    OP_SUBCALL_BATCHED,
    OP_SUBCALL_REPLY,
    CellResult,
    KernelError,
    KernelLimits,
    KernelRejectedError,
    KernelState,
    KernelUnavailableError,
    KernelVar,
    SubcallFn,
    SubcallOutcome,
    bounded_text,
)
from tau_coding.kernel.scaffold import BINDINGS

DEFAULT_CELL_TIMEOUT_SECONDS = 30.0
DEFAULT_STARTUP_TIMEOUT_SECONDS = 20.0
CONTROL_TIMEOUT_SECONDS = 5.0
DEFAULT_CELL_OUTPUT_BYTES = 64 * 1024
DEFAULT_RAW_TAIL_BYTES = 4096

_SECRET_MARKERS = (
    "API_KEY",
    "APIKEY",
    "ACCESS_TOKEN",
    "REFRESH_TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE_KEY",
    "SESSION_TOKEN",
)
_SECRET_PREFIXES = (
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "ANTHROPIC_",
    "OPENAI_",
    "OPENROUTER_",
    "XAI_",
    "GROQ_",
    "HF_",
    "HUGGING_FACE",
    "GITHUB_",
    "GITLAB_",
    "SLACK_",
    "STRIPE_",
    "TAU_",
)


def looks_secret(name: str) -> bool:
    """Whether an environment variable name smells like a credential.

    The kernel intentionally has the same file and network reach as the ``bash``
    tool -- the boundary that matters is the harness's own objects and secrets,
    so a model-written cell must not be able to read the keys the session is
    paying with.
    """
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS) or upper.startswith(_SECRET_PREFIXES)


class _CellBuffer:
    """Accumulates one cell's observations until the done frame closes it."""

    __slots__ = (
        "cell_id",
        "done",
        "frame",
        "max_bytes",
        "state_lost",
        "cancelled",
        "texts",
        "sizes",
        "omitted",
    )

    def __init__(self, cell_id: str, max_bytes: int) -> None:
        self.cell_id = cell_id
        self.done = asyncio.Event()
        self.frame: dict[str, Any] | None = None
        self.max_bytes = max_bytes
        self.state_lost = False
        self.cancelled = False
        self.texts: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self.sizes: dict[str, int] = {"stdout": 0, "stderr": 0}
        self.omitted: dict[str, int] = {"stdout": 0, "stderr": 0}

    def add_output(self, stream: str, text: str) -> None:
        """Append streamed output, capping what is kept and counting the rest."""
        name = "stderr" if stream == "stderr" else "stdout"
        room = self.max_bytes - self.sizes[name]
        clipped, omitted = bounded_text(text, max(0, room))
        if clipped:
            self.texts[name].append(clipped)
            self.sizes[name] += len(clipped.encode("utf-8"))
        self.omitted[name] += omitted

    def finish(self, frame: Mapping[str, Any]) -> None:
        """Record the done frame and wake the waiting caller."""
        if self.frame is not None:
            return
        self.frame = dict(frame)
        if frame.get("state_lost"):
            self.state_lost = True
        if frame.get("cancelled"):
            self.cancelled = True

    def result(
        self,
        *,
        timed_out: bool = False,
        state_lost: bool = False,
        cancelled: bool = False,
        error: str | None = None,
    ) -> CellResult:
        """Build the :class:`CellResult` for this cell."""
        frame = self.frame or {}
        raw_ret = frame.get("ret")
        return CellResult(
            cell_id=self.cell_id,
            ok=bool(frame.get("ok", False)),
            duration_ms=int(frame.get("ms", 0)),
            stdout="".join(self.texts["stdout"]),
            stderr="".join(self.texts["stderr"]),
            stdout_omitted_bytes=int(frame.get("stdout_omitted", 0)) + self.omitted["stdout"],
            stderr_omitted_bytes=int(frame.get("stderr_omitted", 0)) + self.omitted["stderr"],
            error=error if error is not None else _optional_str(frame.get("error")),
            ret=raw_ret,
            returned=bool(frame.get("returned", False)),
            ret_error=_optional_str(frame.get("ret_error")),
            timed_out=timed_out,
            cancelled=cancelled or self.cancelled,
            state_lost=state_lost or self.state_lost,
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


class ProcessKernel:
    """A persistent Python namespace the host drives one cell at a time.

    This is the option-A implementation: the kernel is a plain ``python``
    process, resource-capped by ``setrlimit``, and the only way to stop a runaway
    cell is to kill it. :class:`~tau_coding.kernel.protocol.Kernel` is the seam
    an ``ipykernel``-backed implementation can replace it through.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        subcall: SubcallFn | None = None,
        max_concurrent_subcalls: int = 4,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        limits: KernelLimits | None = None,
        max_cell_output_bytes: int = DEFAULT_CELL_OUTPUT_BYTES,
        python_executable: str | None = None,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        self._cwd = cwd
        self._subcall = subcall
        self._max_concurrent_subcalls = max(1, max_concurrent_subcalls)
        self._cell_timeout = cell_timeout_seconds
        self._startup_timeout = startup_timeout_seconds
        self._limits = limits or KernelLimits()
        self._max_cell_output_bytes = max_cell_output_bytes
        self._executable = python_executable or sys.executable
        self._extra_env = dict(extra_env or {})

        self._host = HOST_DEFAULT
        self._token = secrets.token_hex(16)
        self._port = 0
        self._server: asyncio.Server | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._drains: list[asyncio.Task[None]] = []
        self._raw: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        self._ready = asyncio.Event()
        self._ready_info: Mapping[str, Any] = {}
        self._pending: dict[str, _CellBuffer] = {}
        self._cell_lock = asyncio.Lock()
        self._subcall_semaphore = asyncio.Semaphore(self._max_concurrent_subcalls)
        self._alive = False
        self._closing = False
        self._closed = False
        self._cells_run = 0
        self._subcalls_used = 0
        self._restarts = 0
        self._ever_started = False
        self._dead_reason: str | None = None

    # -- introspection ------------------------------------------------------

    @property
    def alive(self) -> bool:
        """Whether the kernel process is currently usable."""
        return self._alive

    @property
    def subcalls_used(self) -> int:
        """Sub-completions served for this kernel, counting failures."""
        return self._subcalls_used

    @property
    def cells_run(self) -> int:
        """Cells executed since construction, including killed ones."""
        return self._cells_run

    @property
    def restarts(self) -> int:
        """Times the kernel was replaced after being killed."""
        return self._restarts

    @property
    def port(self) -> int:
        """The localhost port the kernel connects back to."""
        return self._port

    @property
    def pid(self) -> int | None:
        """The kernel process id, when one is running."""
        return None if self._proc is None else self._proc.pid

    @property
    def limits_notes(self) -> tuple[str, ...]:
        """Resource limits the child reported as not applied."""
        notes = self._ready_info.get("limits")
        if not isinstance(notes, list):
            return ()
        return tuple(str(note) for note in notes)

    @property
    def python_version(self) -> str | None:
        """The interpreter version the kernel reported."""
        return _optional_str(self._ready_info.get("python"))

    @property
    def dead_reason(self) -> str | None:
        """Why the kernel last stopped, if it has."""
        return self._dead_reason

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the kernel, restarting it if it was killed."""
        if self._closed:
            raise KernelUnavailableError("this kernel was closed; construct and start a new one")
        if self._alive:
            return
        if self._server is None:
            self._server = await asyncio.start_server(self._adopt, self._host, 0)
            sockets = list(self._server.sockets or [])
            self._port = int(sockets[0].getsockname()[1]) if sockets else 0
        self._ready = asyncio.Event()
        self._ready_info = {}
        self._dead_reason = None
        self._closing = False
        env = self._child_env()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._executable,
                "-u",
                "-B",
                "-m",
                "tau_coding.kernel.child",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._cwd),
                env=env,
            )
        except OSError as exc:
            raise KernelUnavailableError(f"cannot start the kernel: {exc}") from exc
        self._drains = [
            asyncio.create_task(self._drain(self._proc.stdout, "stdout")),
            asyncio.create_task(self._drain(self._proc.stderr, "stderr")),
        ]
        try:
            await asyncio.wait_for(self._ready.wait(), self._startup_timeout)
        except TimeoutError as exc:
            reason = self._crash_note("the kernel did not report ready")
            await self._stop(f"{reason} (startup exceeded {self._startup_timeout:.1f}s)")
            raise KernelUnavailableError(reason) from exc
        if self._writer is None or not self._alive:
            # A cell cancelled while the kernel was booting wakes this waiter without
            # handing it a channel; failing here keeps the caller from writing to None.
            reason = self._dead_reason or "the kernel stopped before it finished starting"
            await self._stop(reason)
            raise KernelUnavailableError(reason)
        if self._ever_started:
            self._restarts += 1
        self._ever_started = True

    async def _ensure_running(self) -> None:
        if self._closed:
            raise KernelUnavailableError("this kernel was closed; construct and start a new one")
        if not self._alive:
            await self.start()
        if self._writer is None:  # pragma: no cover - start() would have raised
            raise KernelUnavailableError(self._dead_reason or "the kernel channel is not open")

    async def close(self) -> None:
        """Shut the kernel down and release its channel and server.

        Closing is terminal: a killed kernel restarts lazily on the next cell,
        but a closed one refuses, so a leaked handle cannot resurrect a session's
        execution environment behind its owner's back.
        """
        self._closing = True
        self._closed = True
        if self._writer is not None and self._alive:
            with contextlib.suppress(FrameError):
                await write_frame(self._writer, {"op": OP_SHUTDOWN})
        await self._stop("the kernel was closed")
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(OSError, TimeoutError):  # pragma: no cover - defensive
                await self._server.wait_closed()
            self._server = None
            self._port = 0

    # -- execution ----------------------------------------------------------

    async def execute(self, code: str, *, timeout: float | None = None) -> CellResult:
        """Run one cell in the persistent namespace."""
        async with self._cell_lock:
            cell_id = uuid.uuid4().hex[:12]
            buffer = _CellBuffer(cell_id, self._max_cell_output_bytes)
            limit = self._cell_timeout if timeout is None else float(timeout)
            # Registered before the process is checked: a stop that lands while this
            # cell is still starting has to answer *this* cell, not a future one.
            self._pending[cell_id] = buffer
            timed_out = False
            try:
                try:
                    await self._ensure_running()
                    writer = self._writer
                    if writer is None:  # pragma: no cover - _ensure_running guards this
                        raise KernelUnavailableError(
                            self._dead_reason or "the kernel channel is not open"
                        )
                    await write_frame(writer, {"op": OP_EXEC, "cell_id": cell_id, "code": code})
                    try:
                        await asyncio.wait_for(buffer.done.wait(), limit)
                    except TimeoutError:
                        timed_out = True
                        await self._stop(f"the cell ran past its {limit:.1f}s budget")
                except (KernelError, FrameError, OSError):
                    if buffer.frame is None:
                        raise
            finally:
                self._pending.pop(cell_id, None)
                self._cells_run += 1
            if timed_out:
                return buffer.result(
                    timed_out=True,
                    state_lost=True,
                    error=(
                        f"the cell ran past its {limit:.1f}s budget and the kernel process was "
                        "killed; a Python cell cannot be interrupted any other way, so nothing "
                        "that cell bound survives"
                    ),
                )
            return buffer.result(state_lost=buffer.state_lost, cancelled=buffer.cancelled)

    async def state(self) -> KernelState:
        """Inventory bound names and sizes, never their values."""
        if not self._alive:
            return self._snapshot(note=self._dead_reason or "the kernel is not running")
        if self._cell_lock.locked():
            return self._snapshot(note="a cell is running; the kernel answers only between cells")
        try:
            frame = await self._control({"op": OP_STATE})
        except (KernelError, FrameError) as exc:
            return self._snapshot(note=str(exc))
        rows = frame.get("vars")
        variables: tuple[KernelVar, ...] = ()
        if isinstance(rows, list):
            variables = tuple(KernelVar.from_frame(row) for row in rows if isinstance(row, Mapping))
        return self._snapshot(variables=variables)

    async def reset(self) -> None:
        """Drop the namespace and re-create the bindings."""
        await self._control({"op": OP_RESET})

    async def bind(self, name: str, path: Path) -> None:
        """Expose ``path`` inside the namespace as a handle the cell can read."""
        await self._control({"op": OP_BIND, "name": name, "path": str(path)})

    async def _control(
        self, payload: Mapping[str, Any], *, timeout: float = CONTROL_TIMEOUT_SECONDS
    ) -> dict[str, Any]:
        async with self._cell_lock:
            await self._ensure_running()
            assert self._writer is not None
            cell_id = uuid.uuid4().hex[:12]
            buffer = _CellBuffer(cell_id, DEFAULT_CELL_OUTPUT_BYTES)
            self._pending[cell_id] = buffer
            try:
                await write_frame(self._writer, {**payload, "cell_id": cell_id})
                try:
                    await asyncio.wait_for(buffer.done.wait(), timeout)
                except TimeoutError as exc:
                    raise KernelUnavailableError(
                        f"the kernel did not answer {payload.get('op')!r} in {timeout:.1f}s"
                    ) from exc
            finally:
                self._pending.pop(cell_id, None)
            frame = buffer.frame or {}
            if not frame.get("ok", False):
                raise KernelRejectedError(str(frame.get("error") or "the kernel refused"))
            return frame

    def _snapshot(
        self,
        *,
        variables: tuple[KernelVar, ...] = (),
        note: str | None = None,
    ) -> KernelState:
        return KernelState(
            alive=self._alive,
            cells_run=self._cells_run,
            subcalls_used=self._subcalls_used,
            restarts=self._restarts,
            variables=variables,
            bindings=BINDINGS,
            note=note if note is not None else self._dead_reason,
        )

    # -- the channel --------------------------------------------------------

    async def _adopt(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handshake, then pump frames until the kernel is gone."""
        try:
            hello = await read_frame(reader)
        except FrameError:
            writer.close()
            return
        if hello is None or hello.get("op") != OP_HELLO or hello.get("token") != self._token:
            writer.close()
            return
        if self._writer is not None:
            # One kernel per host: a second connection is refused rather than
            # silently multiplexed onto a namespace that belongs to the first.
            writer.close()
            return
        self._writer = writer
        self._alive = True
        try:
            await write_frame(
                writer,
                {
                    "op": OP_STARTUP,
                    "limits": self._limits.to_json(),
                    "max_cell_output_bytes": self._max_cell_output_bytes,
                },
            )
        except FrameError as exc:
            await self._stop(f"the kernel startup frame could not be sent: {exc}")
            return
        await self._pump(reader, writer)

    async def _pump(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            try:
                frame = await read_frame(reader)
            except FrameError as exc:
                await self._stop(f"the kernel sent a malformed frame: {exc}")
                return
            if frame is None:
                await self._stop(self._crash_note("the kernel closed the channel"))
                return
            op = frame.get("op")
            if op == OP_READY:
                self._ready_info = frame
                self._ready.set()
            elif op == OP_OUT:
                buffer = self._pending.get(str(frame.get("cell_id", "")))
                if buffer is not None:
                    stream = str(frame.get("stream", "stdout"))
                    buffer.add_output(stream, str(frame.get("text", "")))
            elif op == OP_DONE:
                buffer = self._pending.get(str(frame.get("cell_id", "")))
                if buffer is not None:
                    buffer.finish(frame)
                    buffer.done.set()
            elif op in (OP_SUBCALL, OP_SUBCALL_BATCHED):
                reply = await self._serve_subcall(frame)
                try:
                    await write_frame(
                        writer,
                        {"op": OP_SUBCALL_REPLY, "req_id": frame.get("req_id", ""), **reply},
                    )
                except FrameError as exc:
                    await self._stop(f"the sub-call reply never reached the kernel: {exc}")
                    return

    async def _serve_subcall(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Serve one sub-call request from inside a cell."""
        if self._closing:
            return {"error": "the kernel channel is closing"}
        if self._subcall is None:
            return {"error": "no sub-call handler is wired, so this kernel has no model access"}
        raw_model = frame.get("model")
        model = raw_model if isinstance(raw_model, str) else None
        if frame.get("op") == OP_SUBCALL:
            prompt = str(frame.get("prompt", ""))
            self._subcalls_used += 1
            try:
                outcome = await self._bounded(prompt, model)
            except Exception as exc:  # the cell must see a failure, not lose its kernel
                return {"error": f"sub-call failed: {type(exc).__name__}: {exc}"}
            return dict(outcome.to_frame())
        raw_prompts = frame.get("prompts")
        prompts = [str(item) for item in raw_prompts] if isinstance(raw_prompts, list) else []
        self._subcalls_used += len(prompts)
        results = await asyncio.gather(
            *(self._bounded(prompt, model) for prompt in prompts), return_exceptions=True
        )
        responses: list[str | None] = []
        failures: list[str] = []
        for result in results:
            if isinstance(result, BaseException):
                responses.append(None)
                failures.append(f"{type(result).__name__}: {result}")
            else:
                responses.append(result.text)
        reply: dict[str, Any] = {"responses": responses}
        if failures:
            reply["errors"] = failures
        return reply

    async def _bounded(self, prompt: str, model: str | None) -> SubcallOutcome:
        if self._subcall is None:
            raise KernelError("no sub-call handler is wired")
        async with self._subcall_semaphore:
            return await self._subcall(prompt, model)

    # -- process plumbing ---------------------------------------------------

    def _child_env(self) -> dict[str, str]:
        """Build the child environment: inherited, minus anything secret."""
        env = {key: value for key, value in os.environ.items() if not looks_secret(key)}
        source_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{source_root}{os.pathsep}{existing}" if existing else source_root
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env[ENV_HOST] = self._host
        env[ENV_PORT] = str(self._port)
        env[ENV_TOKEN] = self._token
        env.update(self._extra_env)
        return env

    async def _drain(self, stream: asyncio.StreamReader | None, name: str) -> None:
        """Keep the child's real streams drained, retaining a bounded tail.

        Python-level output arrives as frames; these streams only ever carry
        C-level writes, subprocesses the cell started, and the child's own fatal
        errors -- which is exactly what makes a failed start explainable.
        """
        if stream is None:
            return
        while True:
            chunk = await stream.read(8192)
            if not chunk:
                return
            tail = self._raw.setdefault(name, bytearray())
            tail.extend(chunk)
            if len(tail) > DEFAULT_RAW_TAIL_BYTES:
                del tail[: len(tail) - DEFAULT_RAW_TAIL_BYTES]

    def _crash_note(self, base: str) -> str:
        """Attach whatever the child wrote to its real stderr, if anything."""
        stderr = bytes(self._raw.get("stderr", b"")).decode("utf-8", errors="ignore").strip()
        if not stderr:
            return base
        clipped, _ = bounded_text(stderr, DEFAULT_RAW_TAIL_BYTES)
        return f"{base}: {clipped}"

    async def abort_cell(self, reason: str) -> None:
        """Stop the cell that is running right now.

        Option A has no interrupt, so the process dies with the cell and
        ``state_lost`` is reported; the in-flight ``execute`` returns a cancelled
        result rather than raising.
        """
        await self._stop(reason, cancelled=True)

    async def _stop(self, reason: str, *, cancelled: bool = False) -> None:
        """Mark the kernel dead, unblock waiters, and end the process."""
        self._alive = False
        self._dead_reason = reason
        for buffer in list(self._pending.values()):
            buffer.finish(
                {"ok": False, "error": reason, "state_lost": True, "cancelled": cancelled}
            )
            buffer.done.set()
        # A kernel killed mid-startup must not leave its starter waiting for a
        # handshake that will never arrive.
        self._ready.set()
        for task in self._drains:
            task.cancel()
        self._drains = []
        process = self._proc
        if process is not None and process.returncode is None:
            try:
                process.terminate()
                await asyncio.wait_for(process.wait(), 2.0)
            except ProcessLookupError:  # pragma: no cover - raced with exit
                pass
            except TimeoutError:
                process.kill()
                await process.wait()
        self._proc = None
        writer = self._writer
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except (ConnectionError, OSError):  # pragma: no cover - already gone
                pass
        self._writer = None


__all__ = [
    "CONTROL_TIMEOUT_SECONDS",
    "DEFAULT_CELL_OUTPUT_BYTES",
    "DEFAULT_CELL_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "ProcessKernel",
    "looks_secret",
]
