"""Code mode: one persistent Python kernel per session, exposed as ``python_repl``.

Code mode is the mechanism RLM assumes upstream -- context held in a variable the
model slices, and model calls written as functions inside code -- and it is
useful on its own for anything shaped like data: parsing a log, diffing two
structures, fanning one question over forty chunks.

Why it is a tool with a kernel behind it rather than a shell: ``bash`` gives the
model a text pipe, so every intermediate has to be printed to be seen. A Python
namespace lets the model hold a 5 MB list and return only the slice it decided
mattered. The economic argument for recursive work depends on that: the parent
partitions in code and only the partition's answer crosses into a context.

This module owns the seam between the harness and
:mod:`tau_coding.kernel`: it creates one kernel per session, hands the kernel a
sub-call bridge bound to the session's provider and *current* model, converts a
:class:`~tau_coding.kernel.protocol.CellResult` into an
:class:`~tau_agent.tools.AgentToolResult`, and closes the kernel when the session
closes. Policy lives here; the kernel package stays mechanism.
"""

from __future__ import annotations

import asyncio
import json
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tau_agent.messages import TextContent
from tau_agent.tools import AgentTool, AgentToolResult, ToolCancellationToken, ToolUpdateCallback
from tau_agent.types import JSONValue
from tau_coding.context_entries import ContextEntries
from tau_coding.kernel.frames import FrameError
from tau_coding.kernel.process import DEFAULT_MAX_CONCURRENT_RLMS, ProcessKernel
from tau_coding.kernel.protocol import (
    CellResult,
    ContextSource,
    Kernel,
    KernelError,
    KernelLimits,
)
from tau_coding.kernel.scaffold import RESERVED
from tau_coding.kernel.subcall import (
    DEFAULT_MAX_SUBCALLS_PER_CELL,
    SubcallBridge,
    SubcallTotals,
)
from tau_coding.rlm import RlmQuery, RlmTotals
from tau_coding.tools import ToolInputError, format_size, truncate_head

__all__ = [
    "DEFAULT_CELL_TIMEOUT_SECONDS",
    "PYTHON_REPL_TOOL_NAME",
    "CodeMode",
    "MAX_CODE_BYTES",
]

PYTHON_REPL_TOOL_NAME = "python_repl"

DEFAULT_CELL_TIMEOUT_SECONDS = 30.0
"""Per-cell budget. Overrun kills the cell *and its namespace*."""

MAX_CODE_BYTES = 64 * 1024
MAX_OUTPUT_LINES = 400
MAX_OUTPUT_BYTES = 16 * 1024

MAX_ATTACHED_FILE_BYTES = 2 * 1024 * 1024
MAX_ATTACHMENTS = 16
"""Ceilings on file handles handed to the kernel.

An attached file crosses the boundary as a *path*: its bytes never enter a
prompt. The size ceiling only stops the tool from blessing a handle a cell
would then read whole into memory -- slice a bigger file first.
"""

"""Tool-level ceiling on what a cell's text output contributes to context.

The kernel caps what it captures at a larger size; this caps what the harness
hands to the model. Both counts are reported, so a shrunken answer says how much
of itself it lost at each layer.
"""

_CANCEL_POLL_SECONDS = 0.05

_CODE_DESCRIPTION = (
    "Python to execute in the persistent kernel. Statements run as written: "
    "nothing prints implicitly, so use print() to show something and ret({...}) "
    "to return a result as data."
)

_REPL_DESCRIPTION = (
    "Execute Python in a persistent kernel tied to this session. Names bound in "
    "one call -- variables, imports, functions, objects -- are still there in the "
    "next, so large data can stay in the kernel while only the slice that matters "
    "is printed. Returns captured stdout and stderr, plus any value passed to "
    "ret(...), which arrives as JSON rather than prose. SHOW_VARS() lists bound "
    "names and their sizes without values. llm_query(prompt, model=None) and "
    "llm_query_batched(prompts, model=None) call the session's model from inside "
    "the cell; a batched failure comes back as None. attach({name: path}) "
    "binds existing files into the namespace as paths to open, so their bytes "
    "never enter the prompt, and context_index()/context_open(refs) do the same "
    "for the session's own history. A cell that exceeds its timeout is killed, "
    "which also discards the namespace."
)

_REPL_SNIPPET = "Run Python in a persistent per-session kernel (data work, batched model calls)"

_REPL_RECURSION = (
    " rlm_query(task, model=None) and rlm_query_batched(tasks, model=None) hand "
    "a task to a child with a kernel and turns of its own, and return its answer "
    "as text; use them for questions worth a sub-agent, not for a lookup."
)

_REPL_RECURSION_GUIDELINE = (
    "Use rlm_query when the sub-task is open-ended enough that solving it means "
    "writing code you would rather not see: each child is a separate kernel with "
    "its own turns, and its calls still count against this cell's budget."
)

_REPL_GUIDELINES: tuple[str, ...] = (
    "Use python_repl when the work is shaped like data: parsing, slicing, counting, "
    "comparing, or asking the same question over many chunks. Use bash for build and "
    "test commands, and write/edit for file changes.",
    "The namespace persists between calls: bind names once, then use them in later "
    "calls instead of re-reading or re-printing them.",
    "Do not print large objects to inspect them. Slice, count, and sample first, and "
    "end the cell with ret({...}) so the result comes back as data.",
    "Attach a file with attach({name: path}) instead of pasting its contents into "
    "the code: the bound name is a path you open inside the cell.",
    "For conversation history use context_index() and context_open('#3-#9') rather "
    "than asking for the transcript: you get a path to JSONL rows addressed the way "
    "a compaction summary addresses them.",
    "Keep material in kernel variables and use llm_query/llm_query_batched to ask "
    "about it in chunks; bring back answers, not the corpus.",
    "A cell that overruns its timeout is killed together with its namespace, so split "
    "long work across cells rather than into one long one.",
)

_REPL_INPUT_SCHEMA: dict[str, JSONValue] = {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": _CODE_DESCRIPTION},
        "description": {
            "type": "string",
            "description": (
                "Brief present-participle summary of what the cell does, such as "
                "'Partitioning the changelog by file'"
            ),
        },
        "timeout": {
            "type": "number",
            "description": (
                f"Per-cell budget in seconds (default {DEFAULT_CELL_TIMEOUT_SECONDS:g}). "
                "Exceeding it kills the cell and clears the namespace."
            ),
        },
        "attach": {
            "type": "object",
            "propertyNames": {"type": "string"},
            "additionalProperties": {"type": "string"},
            "description": (
                "Files to expose in the namespace as paths, given as a mapping of "
                'name to path, such as {"log": "tests/fixtures/app.log"}. The '
                "name must be a Python identifier and the path may be relative to "
                "the session cwd. Contents stay on disk: the cell opens the path."
            ),
        },
    },
    "required": ["code", "description"],
}


class CodeMode:
    """One kernel per session, plus the ``python_repl`` tool that drives it.

    The kernel object is created lazily, so a session that never calls the tool
    never pays for a process. ``/reload`` re-builds tools but keeps this object --
    and therefore the namespace -- because the namespace is the model's working
    memory, not a resource file.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        subcall: SubcallBridge | None = None,
        context: ContextSource | None = None,
        limits: KernelLimits | None = None,
        cell_timeout_seconds: float = DEFAULT_CELL_TIMEOUT_SECONDS,
        max_concurrent_subcalls: int = 4,
        max_calls_per_cell: int | None = DEFAULT_MAX_SUBCALLS_PER_CELL,
        recursion_enabled: bool = False,
        kernel_factory: Callable[[], Kernel] | None = None,
    ) -> None:
        self._cwd = cwd
        self._subcall = subcall
        self._context = context
        self._limits = limits
        self._cell_timeout = cell_timeout_seconds
        self._max_concurrent_subcalls = max(1, max_concurrent_subcalls)
        self._max_calls_per_cell = max_calls_per_cell
        # The recursion ceilings live in RlmLimits, with whoever builds the
        # RlmQuery; this class only routes requests and reports what they cost.
        # ``recursion_enabled`` is asked for up front because the tool description
        # is built before the session has a harness to answer recursions through:
        # the promise is made early and the runner arrives before the first cell.
        self._recursion_enabled = recursion_enabled
        self._rlm: RlmQuery | None = None
        self._kernel_factory = kernel_factory or self._default_kernel
        self._kernel: Kernel | None = None
        self._tool: AgentTool | None = None
        self._disposed = False

    # -- wiring -------------------------------------------------------------

    def bind_subcall(self, bridge: SubcallBridge) -> None:
        """Attach the host-side answerer for kernel sub-calls.

        A session builds this after its harness exists, because a sub-call resolves
        the live provider and model rather than the startup ones. Binding happens
        before the first cell: the kernel captures the bridge when it starts, and a
        kernel created without one cannot gain ``llm_query`` afterwards.
        """
        if self._kernel is not None:
            raise RuntimeError("cannot bind sub-calls after this kernel has run a cell")
        if self._max_calls_per_cell is not None:
            bridge.max_calls_per_cell = self._max_calls_per_cell
        self._subcall = bridge

    def bind_context(self, source: ContextSource) -> None:
        """Attach the session's live context for ``context_index``/``context_open``.

        Same ordering rule as :meth:`bind_subcall`: the kernel captures the
        source when it starts, so a session binds it before the first cell. The
        source is a live view rather than a snapshot, because the context worth
        searching is the one the next model call will see -- after a compaction
        or a branch switch, not the one that existed at construction.
        """

        if self._kernel is not None:
            raise RuntimeError("cannot bind a session context after this kernel has run a cell")
        self._context = source

    def bind_recursion(self, rlm: RlmQuery | None) -> None:
        """Attach the host-side recursion runner for ``rlm_query``.

        Wiring is the caller's job for the same reason the sub-call bridge is:
        a child kernel needs a factory that can make kernels, and only the owner
        of kernel construction can supply one. A code mode constructed without
        ``recursion_enabled`` never advertises the names and never accepts a
        runner, so the promise in the description and the binding in the
        namespace cannot drift apart.
        """
        if self._kernel is not None:
            raise RuntimeError("cannot bind recursion after this kernel has run a cell")
        if rlm is not None and not self._recursion_enabled:
            raise RuntimeError(
                "this code mode was not created with recursion enabled; the tool "
                "description and the namespace bindings have to agree"
            )
        self._rlm = rlm

    def tool(self) -> AgentTool:
        """Return the session's ``python_repl`` tool."""
        if self._tool is None:
            self._tool = AgentTool(
                name=PYTHON_REPL_TOOL_NAME,
                label="python",
                description=_REPL_DESCRIPTION
                + (_REPL_RECURSION if self._recursion_enabled else ""),
                parameters=_REPL_INPUT_SCHEMA,
                execute_fn=self._execute,
                prompt_snippet=_REPL_SNIPPET,
                prompt_guidelines=(
                    _REPL_GUIDELINES + (_REPL_RECURSION_GUIDELINE,)
                    if self._recursion_enabled
                    else _REPL_GUIDELINES
                ),
                # One namespace, so two cells cannot interleave inside it.
                execution_mode="sequential",
                render_call=_render_call,
            )
        return self._tool

    async def dispose(self) -> None:
        """Close the kernel if one was ever started. Closing is permanent."""
        self._disposed = True
        kernel = self._kernel
        self._kernel = None
        if kernel is not None:
            await kernel.close()
        context = self._context
        if isinstance(context, ContextEntries):
            context.dispose()

    @property
    def started(self) -> bool:
        """Whether this session has run a cell yet."""
        return self._kernel is not None

    @property
    def subcall_bridge(self) -> SubcallBridge | None:
        """The host-side answerer for kernel sub-calls, if code mode has one."""
        return self._subcall

    @property
    def recursion(self) -> RlmQuery | None:
        """The host-side recursion runner, if this code mode was given one."""
        return self._rlm

    def _default_kernel(self) -> Kernel:
        return ProcessKernel(
            cwd=self._cwd,
            subcall=self._subcall,
            context=self._context,
            rlm=self._rlm,
            max_concurrent_subcalls=self._max_concurrent_subcalls,
            max_concurrent_rlms=DEFAULT_MAX_CONCURRENT_RLMS,
            cell_timeout_seconds=self._cell_timeout,
            limits=self._limits,
        )

    def _kernel_or_new(self) -> Kernel:
        if self._kernel is None:
            self._kernel = self._kernel_factory()
        return self._kernel

    async def _attach(
        self, kernel: Kernel, requested: Mapping[str, JSONValue]
    ) -> list[tuple[str, int]]:
        """Bind every requested file handle before the cell runs.

        A handle is not a durable resource: a kernel killed by an overrun loses
        its namespace, so the cell restates what it needs and this re-binds it.
        Re-attaching unconditionally on each call makes that recovery automatic
        instead of contingent on noticing the kernel restarted. The whole cell
        fails -- rather than running half-attached -- because model code that
        reads a missing name gets a confusing NameError instead of the reason
        the handle was refused.
        """

        attached: list[tuple[str, int]] = []
        for name, raw in requested.items():
            attached.append(await _attach_one(kernel, self._cwd, name, raw))
        return attached

    # -- execution ----------------------------------------------------------

    async def _execute(
        self,
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, on_update
        code = _code_arg(arguments)
        timeout = _timeout_arg(arguments, default=self._cell_timeout)
        if self._disposed:
            return _failure_result(
                "code mode is closed for this session; start a new session to run Python again"
            )
        if self._recursion_enabled and self._rlm is None:
            # The description promised rlm_query and nothing was wired to answer
            # it. That is a wiring bug, so it fails the call in words rather than
            # letting the model meet a NameError for a name the tool advertised.
            return _failure_result(
                "recursion was advertised for this session but never wired, so "
                "this python_repl cannot answer rlm_query"
            )
        if signal is not None and signal.is_cancelled():
            raise ToolInputError("Cell cancelled")

        kernel = self._kernel_or_new()
        attachments = await self._attach(kernel, _attach_arg(arguments))
        bridge = self._subcall
        rlm = self._rlm
        if bridge is not None:
            bridge.begin_cell(signal)
        if rlm is not None:
            rlm.begin_cell(signal)
        watch: asyncio.Task[None] | None = None
        try:
            if signal is not None:
                watch = asyncio.create_task(self._watch_cancellation(kernel, signal))
            result = await kernel.execute(code, timeout=timeout)
        except (KernelError, FrameError, OSError) as exc:
            return _failure_result(f"the code-mode kernel could not run this cell: {exc}")
        finally:
            if watch is not None:
                watch.cancel()
            totals = bridge.take_totals() if bridge is not None else None
            recursion = rlm.take_totals() if rlm is not None else None
            if bridge is not None:
                bridge.end_cell()

        return _describe(
            result,
            totals,
            kernel,
            timeout=timeout,
            attachments=attachments,
            recursion=recursion,
        )

    async def _watch_cancellation(self, kernel: Kernel, signal: ToolCancellationToken) -> None:
        """Abort the cell when the harness cancels it.

        A Python cell cannot be interrupted from outside, so cancellation is a
        kill and the namespace goes with it -- which is why the result says so.
        """
        while not signal.is_cancelled():
            await asyncio.sleep(_CANCEL_POLL_SECONDS)
        await kernel.abort_cell("the cell was cancelled")


# -- reporting --------------------------------------------------------------


def _render_call(arguments: Mapping[str, JSONValue]) -> str | None:
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        return None
    lines = [line for line in code.strip().splitlines() if line.strip()]
    first = lines[0].strip()
    if len(lines) > 1:
        return f"{first}  (+{len(lines) - 1} more line{'s' if len(lines) > 2 else ''})"
    return first


def _describe(
    result: CellResult,
    totals: SubcallTotals | None,
    kernel: Kernel,
    *,
    timeout: float,
    attachments: Sequence[tuple[str, int]] = (),
    recursion: RlmTotals | None = None,
) -> AgentToolResult:
    blocks: list[str] = []
    output = result.stdout
    if result.stderr:
        output = f"{output}\n{result.stderr}" if output else result.stderr
    if output.strip():
        truncation = truncate_head(
            output.strip(), max_lines=MAX_OUTPUT_LINES, max_bytes=MAX_OUTPUT_BYTES
        )
        blocks.append(truncation.content)
        if truncation.truncated:
            note = f"[showing first {truncation.output_lines} of {truncation.total_lines} lines"
            if truncation.truncated_by == "bytes":
                note += f" ({format_size(MAX_OUTPUT_BYTES)} limit)"
            note += "]"
            blocks.append(note)
    elif result.ok:
        blocks.append("(no output)")

    if result.returned:
        if result.ret_error:
            blocks.append(f"[ret rejected: {result.ret_error}]")
        else:
            blocks.append(f"[ret] {_as_json(result.ret)}")

    if attachments:
        rows = ", ".join(f"{name} ({format_size(size)})" for name, size in attachments)
        blocks.append(f"[attached as paths: {rows}]")

    if totals is not None and totals.calls:
        parts = [f"{totals.calls} sub-call{'' if totals.calls == 1 else 's'}"]
        if totals.models:
            parts.append(f"via {', '.join(totals.models)}")
        parts.append(f"{totals.requested_tokens} in / {totals.output_tokens} out tokens")
        if totals.cost:
            parts.append(f"~${totals.cost:.4f}")
        if totals.failures:
            parts.append(f"{totals.failures} failed")
        blocks.append(f"[{' '.join(parts)}]")

    if recursion is not None and (recursion.children or recursion.fallbacks or recursion.refusals):
        parts = []
        if recursion.children:
            plural = "" if recursion.children == 1 else "s"
            parts.append(
                f"{recursion.children} recursion{plural}: {recursion.turns} turns, "
                f"{recursion.cells} child cell{'' if recursion.cells == 1 else 's'}, "
                f"depth {recursion.deepest}"
            )
        if recursion.fallbacks:
            parts.append(
                f"{recursion.fallbacks} answered directly at the depth ceiling (not counted above)"
            )
        if recursion.refusals:
            parts.append(f"{recursion.refusals} refused by the recursion budget")
        blocks.append(f"[{' | '.join(parts)}]")

    if not result.ok and result.error:
        blocks.append(f"[error] {result.error.strip()}")
    if result.timed_out:
        blocks.append(
            f"[cell killed after exceeding its {timeout:g}s budget; the namespace is "
            "gone; split the work into smaller cells]"
        )
    elif result.cancelled:
        blocks.append("[cell cancelled]")
    elif result.state_lost:
        blocks.append("[the kernel restarted: names bound by this cell did not survive]")

    dropped = result.omitted_bytes
    if dropped:
        blocks.append(f"[{format_size(dropped)} of this cell's output was capped and not returned]")

    details: dict[str, JSONValue] = {
        "ok": result.ok,
        "duration_ms": result.duration_ms,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "state_lost": result.state_lost,
        "returned": result.returned,
        "ret": result.ret,
        "ret_error": result.ret_error,
        "error": result.error,
        "output_omitted_bytes": result.omitted_bytes,
        "kernel": {
            "alive": kernel.alive,
            "cells_run": kernel.cells_run,
            "subcalls_used": kernel.subcalls_used,
            "recursions_used": getattr(kernel, "recursions_used", 0),
            "restarts": kernel.restarts,
        },
    }
    if attachments:
        details["attachments"] = [{"name": name, "bytes": size} for name, size in attachments]
    if totals is not None:
        details["subcalls"] = totals.to_json()
    if recursion is not None:
        details["recursion"] = recursion.to_json()

    return AgentToolResult(content=[TextContent(text="\n\n".join(blocks))], details=details)


def _failure_result(message: str) -> AgentToolResult:
    return AgentToolResult(
        content=[TextContent(text=f"[error] {message}")],
        details={"ok": False, "error": message},
    )


def _as_json(value: JSONValue) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - frames already encode cleanly
        return str(value)


def _code_arg(arguments: Mapping[str, JSONValue]) -> str:
    raw = arguments.get("code")
    if not isinstance(raw, str) or not raw.strip():
        raise ToolInputError("code must be a non-empty string")
    if len(raw.encode("utf-8")) > MAX_CODE_BYTES:
        raise ToolInputError(
            f"code is {format_size(len(raw.encode('utf-8')))}; the limit is "
            f"{format_size(MAX_CODE_BYTES)}. Split the work across cells -- the "
            "namespace persists between them."
        )
    return raw


def _attach_arg(arguments: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    raw = arguments.get("attach")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ToolInputError("attach must be an object mapping a name to a file path")
    if len(raw) > MAX_ATTACHMENTS:
        raise ToolInputError(
            f"attach names {len(raw)} files; the limit is {MAX_ATTACHMENTS}. "
            "Attach what this cell reads and drop the rest -- paths persist in "
            "the namespace across cells."
        )
    return raw


async def _attach_one(kernel: Kernel, cwd: Path, name: object, raw: JSONValue) -> tuple[str, int]:
    """Resolve one handle and bind it in the kernel; raise with the reason why not."""

    if not isinstance(name, str) or not name.isidentifier():
        raise ToolInputError(f"attach name {name!r} must be a Python identifier")
    if name in RESERVED:
        raise ToolInputError(
            f"attach name {name!r} belongs to a kernel binding; choose another name"
        )
    if not isinstance(raw, str) or not raw.strip():
        raise ToolInputError(f"attach path for {name!r} must be a non-empty string")
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else cwd / candidate
    try:
        info = path.stat()
    except OSError as exc:
        raise ToolInputError(f"attach path for {name!r} could not be read: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ToolInputError(
            f"attach path for {name!r} is not a regular file; the kernel binds "
            "files, not directories or devices"
        )
    if info.st_size > MAX_ATTACHED_FILE_BYTES:
        raise ToolInputError(
            f"attach path for {name!r} is {format_size(info.st_size)}; the limit is "
            f"{format_size(MAX_ATTACHED_FILE_BYTES)}. Slice the file with bash first "
            "-- a handle this large would be read whole inside the cell."
        )
    try:
        await kernel.bind(name, path)
    except (KernelError, FrameError, OSError) as exc:
        raise ToolInputError(f"the kernel refused to attach {name!r}: {exc}") from exc
    return name, info.st_size


def _timeout_arg(arguments: Mapping[str, JSONValue], *, default: float) -> float:
    raw = arguments.get("timeout")
    if raw is None:
        return default
    if not isinstance(raw, (int, float)) or isinstance(raw, bool):
        raise ToolInputError("timeout must be a number")
    value = float(raw)
    if value <= 0:
        raise ToolInputError("timeout must be greater than 0")
    return value
