"""Tau extension that adds a session-persistent Python kernel as one tool.

The failure it fixes: every `bash python3 -c ...` call starts a cold interpreter
and throws the namespace away before the next one, so a model that needs five
dependent computations pays five interpreter starts and re-derives every value by
hand. This tool keeps one live interpreter per session: imports, dataframes, open
files, and live object handles (database connections, RPC/COM proxies) carry over
from cell to cell, and only output goes back to the conversation.

Ports Pi's `pi-pytool`. Deliberate simplifications, all of them semantic rather
than cosmetic:

* The bridge is pure stdlib and runs the cell in a persistent namespace dict,
  so a Tau install needs no `ipykernel`/`jupyter_client` venv build. Pi uses a
  Jupyter kernel because it needs ZMQ's control channel for interrupts; here the
  host sends SIGINT directly, which lands at the next bytecode boundary and gives
  the same guarantee for the case that matters (a cell that will not yield gets
  its kernel killed after a grace period, and the result says so).
* Output is buffered per cell rather than streamed.
* No namespace snapshots: handles cannot be pickled, so a rebuilt kernel starts
  empty and says so out loud.

Install by copying into `~/.tau/extensions/`, or run:

    tau -e examples/extensions/python_kernel.py
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tau_agent.messages import TextContent
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_coding.extensions import (
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    SessionShutdownEvent,
)

TOOL_NAME = "python"
COMMAND_NAME = "python"

DEFAULT_MAX_OUTPUT_CHARS = 46_080
DEFAULT_LINE_CHARS = 4_096
DEFAULT_ABORT_GRACE_MS = 5_000
DEFAULT_CELL_TIMEOUT_MS = 0
BOOT_TIMEOUT_SECONDS = 90.0
REAP_WAIT_SECONDS = 2.0

REAPED_NOTICE = (
    "<tau_python_kernel_reaped> the cell ignored the interrupt, so the kernel was "
    "killed; every name and live handle it held is gone."
)
RESET_NOTICE = (
    "<tau_python_kernel_reset> this kernel is new and empty: re-import, re-open, and "
    "re-compute before using a name from an earlier cell."
)
TIMEOUT_NOTICE = "<tau_python_cell_timeout> the cell was cut short by its own deadline."

# The Python side of the kernel. Kept as a source string so the extension stays
# one installable file: the host runs it with `<interpreter> -u -c <source>`.
_BRIDGE_SOURCE = r'''
"""Tau python-kernel bridge: one persistent namespace, one JSON line per op.

host -> bridge (stdin)
  {"op":"exec","id":<str>,"code":<str>}   run one cell, stream nothing, answer once
  {"op":"names","id":<str>}               public names currently held
  {"op":"shutdown","id":<str>}           exit 0
bridge -> host (stdout)
  {"type":"ready","pid":<int>}           first line, always
  {"type":"result","id":<str>,"status":"ok"|"error"|"aborted","stdout":<str>,
   "stderr":<str>,"value":<str|null>,"error":{name,message,stack}|null,"names":[str]}
  {"type":"names","id":<str>,"names":[str]}
  {"type":"error","id":<str|null>,"message":<str>}

An interrupt is SIGINT from the host, never an op: while a cell runs the bridge is
blocked inside exec and cannot read stdin, so a signal is the only thing that can
reach it. It takes effect at the next bytecode boundary and cannot enter a blocking
C call; the host escalates that case to a kill, and the namespace dies with the
process. stdin EOF means the host is gone, so the bridge simply stops.
"""

import ast
import builtins
import contextlib
import json
import os
import sys
import traceback

NAMESPACE = {"__name__": "__tau_python_kernel__", "__doc__": None}
CAPTURE_LIMIT = 2_000_000
VALUE_LIMIT = 4_000


class Capped:
    """Bounded capture buffer: a runaway print loop must not outgrow the pipe."""

    def __init__(self, limit):
        self._limit = limit
        self._parts = []
        self._size = 0
        self.dropped = 0

    def write(self, text):
        size = len(text)
        if self._size >= self._limit:
            self.dropped += size
            return size
        room = self._limit - self._size
        if size > room:
            self._parts.append(text[:room])
            self._parts.append("...[%d chars dropped]..." % (size - room))
            self.dropped += size - room
        else:
            self._parts.append(text)
        self._size = min(self._limit, self._size + size)
        return size

    def flush(self):
        pass

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return False

    def getvalue(self):
        if self.dropped:
            return "".join(self._parts) + "\n...[%d further chars dropped]..." % self.dropped
        return "".join(self._parts)


def use_utf8_stdio():
    """Pin the protocol pipe to UTF-8 instead of the OS locale codec."""
    for name, errors in (("stdin", "strict"), ("stdout", "backslashreplace")):
        stream = getattr(sys, name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors=errors)
        except (AttributeError, ValueError, OSError):
            pass


def emit(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def public_names():
    return sorted(name for name in NAMESPACE if not name.startswith("_"))


def error_payload(exc):
    return {
        "name": type(exc).__name__,
        "message": str(exc),
        "stack": [
            line.rstrip("\n")
            for line in traceback.format_exception(type(exc), exc, exc.__traceback__)
        ],
    }


def run_cell(code):
    out, err = Capped(CAPTURE_LIMIT), Capped(CAPTURE_LIMIT)
    status, error, value = "ok", None, None
    try:
        tree = ast.parse(code, filename="<tau-python>", mode="exec")
    except (SyntaxError, ValueError) as exc:
        return {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "value": None,
            "error": error_payload(exc),
        }
    statements = tree.body
    trailing = None
    if statements and isinstance(statements[-1], ast.Expr):
        trailing = statements[-1]
        statements = statements[:-1]
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            if statements:
                module = ast.Module(body=statements, type_ignores=tree.type_ignores)
                exec(compile(module, "<tau-python>", "exec"), NAMESPACE)
            if trailing is not None:
                expression = ast.Expression(body=trailing.value)
                ast.fix_missing_locations(expression)
                result = eval(compile(expression, "<tau-python>", "eval"), NAMESPACE)
                if result is not None:
                    value = builtins.repr(result)[:VALUE_LIMIT]
    except KeyboardInterrupt:
        status = "aborted"
    except SystemExit as exc:
        status = "error"
        error = {
            "name": "SystemExit",
            "message": "sys.exit(%r) is ignored: the kernel outlives the cell" % (exc.code,),
            "stack": [],
        }
    except BaseException as exc:
        status = "error"
        error = error_payload(exc)
    return {
        "status": status,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
        "value": value,
        "error": error,
    }


def main():
    use_utf8_stdio()
    emit({"type": "ready", "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            emit({"type": "error", "id": None, "message": "malformed protocol line: %s" % exc})
            continue
        op = request.get("op")
        request_id = request.get("id")
        if op == "exec":
            result = run_cell(str(request.get("code", "")))
            result.update({"type": "result", "id": request_id, "names": public_names()})
            emit(result)
        elif op == "names":
            emit({"type": "names", "id": request_id, "names": public_names()})
        elif op == "shutdown":
            emit({"type": "bye", "id": request_id})
            return
        elif op != "boot":
            emit({"type": "error", "id": request_id, "message": "unknown op: %r" % (op,)})


if __name__ == "__main__":
    main()
'''


class KernelGone(RuntimeError):
    """The bridge process exited; the namespace went with it."""


@dataclass(frozen=True)
class KernelSettings:
    """Resolved configuration for one kernel."""

    interpreter: str
    interpreter_source: str
    max_output_chars: int
    line_chars: int
    cell_timeout_ms: int
    abort_grace_ms: int


@dataclass
class CellOutcome:
    """One settled cell, as the host saw it."""

    status: str
    stdout: str = ""
    stderr: str = ""
    value: str | None = None
    error: dict[str, Any] | None = None
    names: list[str] = field(default_factory=list)
    timed_out: bool = False
    reaped: bool = False
    noticed_reset: bool = False


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def resolve_interpreter(env: Mapping[str, str]) -> tuple[str, str]:
    """Pick the interpreter and name the rule that picked it."""
    for variable in ("TAU_PYTHON", "PYTOOL_PYTHON", "PI_PYTHON"):
        pinned = env.get(variable)
        if pinned and pinned.strip():
            return pinned.strip(), f"{variable} pin"
    for name in ("python3", "python"):
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            return found, f"PATH {name}"
    return sys.executable, "tau interpreter"


def settings_from_env(env: Mapping[str, str]) -> KernelSettings:
    """Build kernel settings from the process environment."""
    interpreter, source = resolve_interpreter(env)
    return KernelSettings(
        interpreter=interpreter,
        interpreter_source=source,
        max_output_chars=_env_int(env, "TAU_PY_MAX_OUTPUT_CHARS", DEFAULT_MAX_OUTPUT_CHARS),
        line_chars=_env_int(env, "TAU_PY_LINE_CHARS", DEFAULT_LINE_CHARS),
        cell_timeout_ms=_env_int(env, "TAU_PY_CELL_TIMEOUT_MS", DEFAULT_CELL_TIMEOUT_MS),
        abort_grace_ms=_env_int(env, "TAU_PY_ABORT_GRACE_MS", DEFAULT_ABORT_GRACE_MS),
    )


def clip_line(line: str, line_chars: int) -> str:
    """Cut one over-long line with a visible marker."""
    if line_chars <= 0 or len(line) <= line_chars:
        return line
    return f"{line[:line_chars]} ...[{len(line) - line_chars} chars cut]"


def clip_output(text: str, max_chars: int, line_chars: int) -> str:
    """Cap cell output per line and in total, keeping both ends of the middle cut."""
    if not text:
        return ""
    clipped = "\n".join(clip_line(line, line_chars) for line in text.splitlines())
    if max_chars <= 0 or len(clipped) <= max_chars:
        return clipped
    dropped = len(clipped) - max_chars
    head = clipped[: max_chars // 2]
    tail = clipped[len(clipped) - (max_chars - len(head)) :]
    return f"{head}\n...[{dropped} chars cut]...\n{tail}"


def format_names(names: list[str], limit: int = 12) -> str:
    """Render the live namespace compactly."""
    if not names:
        return "empty"
    if len(names) <= limit:
        return f"{len(names)} names: {', '.join(names)}"
    return f"{len(names)} names: {', '.join(names[:limit])} ..."


def compose_cell_text(outcome: CellOutcome, max_chars: int, line_chars: int) -> str:
    """Render one cell result in the shape the model reads back."""
    parts: list[str] = []
    if outcome.reaped:
        parts.append(REAPED_NOTICE)
    if outcome.noticed_reset:
        parts.append(RESET_NOTICE)
    if outcome.timed_out and not outcome.reaped:
        parts.append(f"{TIMEOUT_NOTICE} names survived: {format_names(outcome.names)}.")

    status = "timeout" if outcome.timed_out and not outcome.reaped else outcome.status
    parts.append(f"status: {status}")
    if outcome.stdout:
        parts.append(f"stdout:\n{clip_output(outcome.stdout, max_chars, line_chars)}")
    if outcome.stderr:
        parts.append(f"stderr:\n{clip_output(outcome.stderr, max_chars, line_chars)}")
    if outcome.value:
        parts.append(f"value: {clip_output(outcome.value, max_chars, line_chars)}")
    if outcome.error:
        stack = outcome.error.get("stack") or [str(outcome.error.get("message", ""))]
        parts.append(f"error: {outcome.error.get('name', 'Error')}")
        parts.append(clip_output("\n".join(str(line) for line in stack), max_chars, line_chars))
    parts.append(f"namespace: {format_names(outcome.names)}")
    return "\n".join(part for part in parts if part)


class PythonKernel:
    """Owns one bridge subprocess and serializes cells through it."""

    def __init__(self, settings: KernelSettings, cwd: Path) -> None:
        self._settings = settings
        self._cwd = cwd
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_tail: list[str] = []
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._ready: asyncio.Future[dict[str, Any]] | None = None
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._names: list[str] = []
        self._reset_pending = False

    @property
    def settings(self) -> KernelSettings:
        """Return the resolved settings for this kernel."""
        return self._settings

    @property
    def names(self) -> list[str]:
        """Return the last namespace snapshot the host received."""
        return list(self._names)

    @property
    def pid(self) -> int | None:
        """Return the bridge pid, or None when no kernel is live."""
        process = self._process
        if process is None or process.returncode is not None:
            return None
        return process.pid

    @property
    def state(self) -> str:
        """Report the kernel state for diagnostics."""
        process = self._process
        if process is None:
            return "stopped"
        if process.returncode is not None:
            return "dead"
        return "ready"

    async def exec_cell(
        self,
        code: str,
        *,
        timeout_ms: int | None = None,
        grace_ms: int | None = None,
        cancel: ToolCancellationToken | None = None,
    ) -> CellOutcome:
        """Run one cell, cutting over-long cells with an interrupt then a kill."""
        limit = self._settings.cell_timeout_ms if timeout_ms is None else max(0, timeout_ms)
        grace = self._settings.abort_grace_ms if grace_ms is None else max(0, grace_ms)
        async with self._lock:
            await self._ensure_started()
            loop = asyncio.get_running_loop()
            self._sequence += 1
            cell_id = f"cell-{self._sequence}"
            future: asyncio.Future[dict[str, Any]] = loop.create_future()
            self._pending[cell_id] = future
            noticed_reset = self._reset_pending
            self._reset_pending = False
            cut_short = False
            try:
                self._write({"op": "exec", "id": cell_id, "code": code})
                if limit <= 0:
                    message = await future
                else:
                    try:
                        message = await asyncio.wait_for(asyncio.shield(future), limit / 1000)
                    except TimeoutError:
                        cut_short = True
                        self._interrupt()
                        try:
                            message = await asyncio.wait_for(asyncio.shield(future), grace / 1000)
                        except TimeoutError:
                            self._pending.pop(cell_id, None)
                            await self._reap()
                            return CellOutcome(
                                status="aborted",
                                names=self.names,
                                timed_out=True,
                                reaped=True,
                                noticed_reset=noticed_reset,
                            )
            except asyncio.CancelledError:
                self._pending.pop(cell_id, None)
                self._interrupt()
                raise
            except KernelGone:
                self._pending.pop(cell_id, None)
                return CellOutcome(
                    status="error",
                    stderr=self._stderr_text(),
                    error={
                        "name": "KernelGone",
                        "message": "the kernel process exited during the cell",
                        "stack": [],
                    },
                    names=self.names,
                    reaped=True,
                    noticed_reset=noticed_reset,
                )
            finally:
                self._pending.pop(cell_id, None)

        message["status"] = str(message.get("status", "ok"))
        error = message.get("error")
        outcome = CellOutcome(
            status=str(message["status"]),
            stdout=str(message.get("stdout") or ""),
            stderr=str(message.get("stderr") or ""),
            value=None if message.get("value") is None else str(message.get("value")),
            error=error if isinstance(error, dict) else None,
            names=[str(name) for name in (message.get("names") or [])],
            timed_out=cut_short,
            reaped=False,
            noticed_reset=noticed_reset,
        )
        self._names = list(outcome.names)
        if cancel is not None and cancel.is_cancelled():
            outcome.status = "aborted"
        return outcome

    def shutdown_now(self) -> None:
        """Ask for a clean exit and stop the process without waiting on the loop."""
        process = self._process
        self._process = None
        self._names = []
        self._reset_pending = True
        if process is None or process.returncode is not None:
            return
        with suppress(BrokenPipeError, Exception):
            if process.stdin is not None:
                process.stdin.write(b'{"op":"shutdown","id":"bye"}\n')
        with suppress(ProcessLookupError):
            process.terminate()

    async def aclose(self) -> None:
        """Stop the kernel and reap it."""
        process = self._process
        self._process = None
        if process is not None and process.returncode is None:
            with suppress(BrokenPipeError, Exception):
                if process.stdin is not None:
                    process.stdin.write(b'{"op":"shutdown","id":"bye"}\n')
                    await process.stdin.drain()
            with suppress(ProcessLookupError):
                process.terminate()
            with suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), REAP_WAIT_SECONDS)
        await self._drop_reader()
        self._names = []
        self._reset_pending = True

    async def _ensure_started(self) -> None:
        process = self._process
        ready = self._ready
        if (
            process is not None
            and process.returncode is None
            and ready is not None
            and ready.done()
            and not ready.cancelled()
        ):
            return
        self._stderr_tail = []
        self._ready = None
        process = await asyncio.create_subprocess_exec(
            self._settings.interpreter,
            "-u",
            "-c",
            _BRIDGE_SOURCE,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
            limit=8 * 1024 * 1024,
        )
        self._process = process
        loop = asyncio.get_running_loop()
        self._ready = loop.create_future()
        self._reader = asyncio.create_task(self._read_loop(process))
        asyncio.create_task(self._drain_stderr(process))
        try:
            await asyncio.wait_for(asyncio.shield(self._ready), BOOT_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            await self._reap()
            raise KernelGone("the kernel booted but never answered") from exc

    async def _read_loop(self, process: asyncio.subprocess.Process) -> None:
        stdout = process.stdout
        assert stdout is not None
        while True:
            raw = await stdout.readline()
            if not raw:
                break
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            kind = message.get("type")
            if kind == "ready":
                ready = self._ready
                if ready is not None and not ready.done():
                    ready.set_result(message)
                continue
            pending = self._pending.get(str(message.get("id")))
            if pending is not None and not pending.done():
                pending.set_result(message)
        self._fail_pending(KernelGone("the kernel process exited"))
        ready = self._ready
        if ready is not None and not ready.done():
            ready.set_exception(KernelGone(self._stderr_text() or "the kernel never became ready"))

    async def _drain_stderr(self, process: asyncio.subprocess.Process) -> None:
        stderr = process.stderr
        if stderr is None:
            return
        while True:
            chunk = await stderr.readline()
            if not chunk:
                break
            self._stderr_tail.append(chunk.decode("utf-8", "replace").rstrip("\n"))
            del self._stderr_tail[:-20]

    def _fail_pending(self, error: BaseException) -> None:
        for pending in list(self._pending.values()):
            if not pending.done():
                pending.set_exception(error)

    def _write(self, payload: Mapping[str, JSONValue]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise KernelGone("the kernel is not running")
        process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail)

    def _interrupt(self) -> None:
        process = self._process
        if process is None or process.returncode is not None:
            return
        sigint = getattr(signal, "SIGINT", None)
        if sigint is None:
            return
        with suppress(ProcessLookupError):
            process.send_signal(sigint)

    async def _reap(self) -> None:
        process = self._process
        self._process = None
        self._ready = None
        self._reset_pending = True
        self._names = []
        self._fail_pending(KernelGone("the kernel was reaped"))
        if process is not None and process.returncode is None:
            with suppress(ProcessLookupError):
                process.terminate()
            with suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), REAP_WAIT_SECONDS)
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
        await self._drop_reader()

    async def _drop_reader(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None and not reader.done():
            reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader


def _render_call(arguments: Mapping[str, JSONValue]) -> str | None:
    """Show the first line of the cell instead of dumping its source."""
    code = str(arguments.get("code", ""))
    for line in code.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= 80 else f"{stripped[:80]} ..."
    return None


def _render_result(result: AgentToolResult, *, expanded: bool) -> str | None:
    """Collapse a cell result to one transcript line unless the user expanded it."""
    if expanded:
        return None
    details = result.details if isinstance(result.details, dict) else {}
    status = str(details.get("status", "ok"))
    marker = {"ok": "[green]\u2713[/green]", "error": "[red]\u2717[/red]"}.get(
        status, "[yellow]![/yellow]"
    )
    names = details.get("names")
    live = len(names) if isinstance(names, list) else 0
    tail = f"{live} live names" if live else "no live names"
    return f"{marker} [dim]python \u00b7 {status} \u00b7 {tail}[/dim]"


def setup(tau: ExtensionAPI) -> None:
    """Register the kernel tool, its transcript renderer, and its status command."""
    settings = settings_from_env(os.environ)
    kernel: PythonKernel | None = None
    cwd = Path.cwd()

    def ensure_kernel() -> PythonKernel:
        """Return the session kernel, building it on first use."""
        nonlocal kernel
        if kernel is None:
            kernel = PythonKernel(settings, cwd)
        return kernel

    async def execute_python(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        """Run one cell in the session kernel."""
        del tool_call_id, on_update
        code = str(arguments.get("code", ""))
        max_chars = arguments.get("maxOutputChars")
        timeout_ms = arguments.get("timeoutMs")
        active = ensure_kernel()
        limit = settings.max_output_chars
        if isinstance(max_chars, int) and max_chars > 0:
            limit = max_chars
        try:
            outcome = await active.exec_cell(
                code,
                timeout_ms=timeout_ms if isinstance(timeout_ms, int) else None,
                cancel=signal,
            )
        except OSError as exc:
            details: dict[str, JSONValue] = {
                "status": "error",
                "names": [],
                "interpreter": settings.interpreter,
                "stdoutChars": 0,
                "stderrChars": 0,
            }
            text = (
                "status: error\n"
                f"error: cannot start the kernel at {settings.interpreter} "
                f"(picked by {settings.interpreter_source})\n{exc}\n"
                "Set TAU_PYTHON to a working interpreter, or run /python to see what was picked."
            )
            return AgentToolResult(content=[TextContent(text=text)], details=details)
        details = {
            "status": outcome.status,
            "timedOut": outcome.timed_out,
            "reaped": outcome.reaped,
            "names": list(outcome.names),
            "interpreter": active.settings.interpreter,
            "stdoutChars": len(outcome.stdout),
            "stderrChars": len(outcome.stderr),
        }
        text = compose_cell_text(
            outcome,
            limit,
            settings.line_chars,
        )
        return AgentToolResult(content=[TextContent(text=text)], details=details)

    tau.register_tool(
        AgentTool(
            name=TOOL_NAME,
            label="Python kernel",
            description=(
                "Run Python in a kernel that stays alive for the whole session. Names, imports, "
                "loaded data, open files and live object handles (database connections, RPC or "
                "COM proxies) carry over from one call to the next, and only output comes back. "
                "Use it for any Python work rather than running an interpreter through bash, which "
                "starts cold and keeps nothing. A cell that raises keeps the kernel and its "
                f"namespace; a cell that ignores an interrupt has its kernel killed, which empties "
                f"the namespace and is reported as reaped. Output is capped at "
                f"{settings.max_output_chars} chars per cell with an explicit marker. One "
                "interpreter runs the kernel, named by /python, and a failed import means "
                "installing into that interpreter."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source for one cell."},
                    "maxOutputChars": {
                        "type": "integer",
                        "description": f"Per-cell output cap, default {DEFAULT_MAX_OUTPUT_CHARS}.",
                    },
                    "timeoutMs": {
                        "type": "integer",
                        "description": (
                            "Wall clock for this cell; interrupt first, kill only if the cell "
                            "will not yield. Default 0 means no deadline."
                        ),
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            execute_fn=execute_python,
            prompt_snippet=(
                "Persistent Python kernel - names, data and live connections survive between calls"
            ),
            prompt_guidelines=(
                "The live kernel is the source of truth: reuse existing names instead of "
                "re-importing, reloading or recomputing them.",
                "Look before acting: print the minimal view that answers the question. The last "
                "expression's value comes back on its own.",
                "Drive blocking APIs (RPC, COM, sockets) as many short calls, never one long "
                "batch: an interrupt cannot cut a call parked inside C, and the only way back is "
                "killing the kernel and losing every name and handle.",
                "If a result carries <tau_python_kernel_reset> or says the kernel was reaped, the "
                "namespace is empty and its handles are dead: re-create them before use.",
                "Never compute through bash: `python3 -c`, heredocs and pipes start a cold "
                "interpreter and keep nothing. Files, processes, git and installs stay in bash.",
            ),
            execution_mode="sequential",
            render_call=_render_call,
            render_result=_render_result,
        )
    )

    @tau.on("session_start")
    def on_session_start(event: object, context: ExtensionContext) -> None:
        """Bind the kernel to this session's directory and drop any inherited one."""
        nonlocal kernel, cwd
        del event
        cwd = context.cwd
        if kernel is not None:
            kernel.shutdown_now()
            kernel = None

    @tau.on("session_shutdown")
    def on_session_shutdown(event: SessionShutdownEvent, context: ExtensionContext) -> None:
        """Stop the kernel with the session so no interpreter is left holding handles."""
        nonlocal kernel
        del event, context
        if kernel is not None:
            kernel.shutdown_now()
            kernel = None

    def handle_status(args: str, context: ExtensionCommandContext) -> str:
        """Report interpreter, processes, caps, and the live namespace."""
        del context
        active = ensure_kernel()
        if args.strip() == "reset":
            active.shutdown_now()
            return f"python: kernel stopped; the next cell boots a fresh, empty one at {cwd}"
        pid = active.pid
        lines = [
            f"kernel: {active.state}" + (f" \u00b7 pid {pid}" if pid is not None else ""),
            f"python: {active.settings.interpreter} ({active.settings.interpreter_source})",
            f"cwd: {cwd}",
            f"output cap: {active.settings.max_output_chars} chars per cell, "
            f"{active.settings.line_chars} per line",
            f"cell deadline: {active.settings.cell_timeout_ms} ms, abort grace "
            f"{active.settings.abort_grace_ms} ms",
            f"namespace: {format_names(active.names)}",
            "no snapshots: a rebuilt kernel starts empty",
        ]
        return "\n".join(lines)

    tau.register_command(
        COMMAND_NAME,
        handle_status,
        description="Show or reset the persistent Python kernel",
        usage="[reset]",
    )
