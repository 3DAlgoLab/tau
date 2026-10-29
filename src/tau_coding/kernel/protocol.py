"""The kernel boundary: frame vocabulary, results, and the host-facing protocol.

Two rules shape everything here.

**Values do not cross.** A variable created by model code stays in the kernel
namespace. The host receives bounded output, a size-only inventory of what is
bound, and whatever the code explicitly reports through ``ret()``. That is the
whole point of an RLM-style harness: a 5 MB list can be sliced in the kernel and
only the slice the model chose to print reaches anybody's context.

**The kernel is disposable, the namespace is not durable.** A runaway cell can
only be stopped by killing the process, and killing it destroys the namespace.
Every result therefore carries ``state_lost`` so the caller can tell the model
the truth instead of pretending continuity.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Protocol

from tau_agent.types import JSONValue

# Host -> kernel.
OP_EXEC = "exec"
OP_STATE = "state"
OP_RESET = "reset"
OP_BIND = "bind"
OP_SHUTDOWN = "shutdown"
# Kernel -> host.
OP_HELLO = "hello"
OP_OUT = "out"
OP_DONE = "done"
OP_STARTUP = "startup"
OP_READY = "ready"
OP_SUBCALL = "subcall"
OP_SUBCALL_BATCHED = "subcall_batched"
OP_SUBCALL_REPLY = "subcall_reply"
OP_CONTEXT = "context"
OP_CONTEXT_REPLY = "context_reply"

ENV_HOST = "TAU_KERNEL_HOST"
ENV_PORT = "TAU_KERNEL_PORT"
ENV_TOKEN = "TAU_KERNEL_TOKEN"

HOST_DEFAULT = "127.0.0.1"


class KernelError(RuntimeError):
    """Base class for kernel failures."""


class KernelUnavailableError(KernelError):
    """Raised when a kernel is needed but the process is not running."""


class KernelRejectedError(KernelError):
    """Raised when the kernel refuses an operation and says why."""


@dataclass(frozen=True, slots=True)
class KernelLimits:
    """Best-effort resource ceilings applied inside the kernel child.

    These are ``setrlimit`` values enforced by the OS rather than polite
    timeouts: they stop a cell from pinning all memory or writing an unbounded
    file. ``RLIMIT_AS`` in particular can break imports of large native
    extensions, so it stays off unless asked for. Resource limits are POSIX
    only; on other platforms they are reported as not applied.
    """

    cpu_seconds: int | None = None
    address_space_mb: int | None = None
    file_size_mb: int | None = None
    open_files: int | None = None
    max_processes: int | None = None

    def to_json(self) -> dict[str, JSONValue]:
        """Return the limits as a JSON mapping for the startup frame."""
        payload: dict[str, JSONValue] = {}
        for entry in dataclass_fields(self):
            value = getattr(self, entry.name)
            if value is not None:
                payload[entry.name] = value
        return payload


@dataclass(frozen=True, slots=True)
class KernelVar:
    """One entry in the namespace inventory.

    ``shallow_bytes`` is exactly that: a shallow ``sys.getsizeof``. It is an
    accounting hint for the model and for status commands, never a promise about
    the object graph, and no value is ever included.
    """

    name: str
    type: str
    length: int | None
    shallow_bytes: int

    @staticmethod
    def from_frame(frame: Mapping[str, Any]) -> KernelVar:
        """Rebuild one inventory row from a ``state`` frame."""
        length = frame.get("length")
        size = frame.get("bytes")
        return KernelVar(
            name=str(frame.get("name", "")),
            type=str(frame.get("type", "unknown")),
            length=length if isinstance(length, int) else None,
            shallow_bytes=size if isinstance(size, int) else 0,
        )


@dataclass(frozen=True, slots=True)
class KernelState:
    """What the host can know about a kernel without reading its variables."""

    alive: bool
    cells_run: int
    subcalls_used: int
    restarts: int
    variables: tuple[KernelVar, ...] = ()
    bindings: tuple[str, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True)
class CellResult:
    """One executed cell: bounded observations, never the namespace itself."""

    cell_id: str
    ok: bool
    duration_ms: int
    stdout: str
    stderr: str
    stdout_omitted_bytes: int = 0
    stderr_omitted_bytes: int = 0
    error: str | None = None
    ret: JSONValue | None = None
    returned: bool = False
    ret_error: str | None = None
    timed_out: bool = False
    cancelled: bool = False
    state_lost: bool = False

    @property
    def omitted_bytes(self) -> int:
        """Total output bytes dropped across both streams."""
        return self.stdout_omitted_bytes + self.stderr_omitted_bytes


@dataclass(frozen=True, slots=True)
class SubcallOutcome:
    """One host-computed completion handed back to the kernel.

    ``metadata`` is deliberately opaque so this layer stays independent of
    :mod:`tau_ai`; the code-mode tool fills it with model and usage facts.
    """

    text: str
    model: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def to_frame(self) -> dict[str, JSONValue]:
        """Render the outcome as a reply frame body."""
        metadata = dict(self.metadata)
        if self.model is not None:
            metadata.setdefault("model", self.model)
        return {"text": self.text, "metadata": metadata}


SubcallFn = Callable[[str, str | None], Awaitable[SubcallOutcome]]
"""Host-side completion the kernel can ask for, one prompt at a time."""


class ContextSource(Protocol):
    """Host-side supplier of addressed session-context handles.

    A kernel asks for two things and receives only JSON: a bounded index of the
    active context, and a **path** to a file holding the slice it named. Row
    text travels through the file, never through a frame, which is what keeps a
    200 KB history branch out of anybody's prompt. The kernel package stays
    mechanism: it forwards these requests and never inspects a session type.

    Reply contract (both keys optional except where noted):

    * ``index`` -> ``{"rows": [...], "total": int, "omitted": int, ...}``
    * ``open``  -> ``{"path": str, "written": int, "omitted": int, ...}``
    """

    async def index(self, query: str | None) -> Mapping[str, JSONValue]:
        """Describe the active context in bounded rows; never its full text."""
        ...

    async def open(self, refs: tuple[str, ...]) -> Mapping[str, JSONValue]:
        """Materialize the named entries as a file the kernel can read."""
        ...


class Kernel(Protocol):
    """The execution environment the code-mode tool needs.

    :class:`~tau_coding.kernel.process.ProcessKernel` is the option-A
    implementation. The protocol exists so an ``ipykernel``-backed kernel can be
    dropped in later -- hard per-cell interrupt, surviving namespace -- without
    touching anything above this line.
    """

    @property
    def alive(self) -> bool:
        """Whether the kernel process is currently usable."""
        ...

    @property
    def subcalls_used(self) -> int:
        """Sub-completions served for this kernel so far."""
        ...

    @property
    def cells_run(self) -> int:
        """Cells executed by this kernel object, including killed ones."""
        ...

    @property
    def restarts(self) -> int:
        """Times the kernel was restarted after being killed."""
        ...

    async def start(self) -> None:
        """Start the kernel, or restart it after it was killed."""
        ...

    async def execute(self, code: str, *, timeout: float | None = None) -> CellResult:
        """Run one cell in the persistent namespace."""
        ...

    async def abort_cell(self, reason: str) -> None:
        """Stop the cell that is running right now.

        The in-flight :meth:`execute` returns with ``cancelled`` set. Whether the
        namespace survives is implementation-specific and reported through
        ``state_lost``: option A has no interrupt, so it kills the process, while
        an ``ipykernel`` kernel interrupts the cell and keeps the namespace.
        """
        ...

    async def state(self) -> KernelState:
        """Inventory bound names and sizes; never their values."""
        ...

    async def bind(self, name: str, path: Path) -> None:
        """Expose a file path inside the namespace as a handle to read."""
        ...

    async def reset(self) -> None:
        """Drop the namespace and re-create the bindings."""
        ...

    async def close(self) -> None:
        """Shut the kernel down and release its channel."""
        ...


def bounded_text(text: str, max_bytes: int) -> tuple[str, int]:
    """Clip ``text`` to ``max_bytes`` of UTF-8 and report what was dropped.

    Walks characters, never byte offsets, so a multibyte character cannot be cut
    into an invalid sequence -- the same rule the compaction renderer uses.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, 0
    keep = encoded[: max(0, max_bytes)]
    clipped = keep.decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(clipped.encode("utf-8"))
    return clipped, omitted


__all__ = [
    "ENV_HOST",
    "ENV_PORT",
    "ENV_TOKEN",
    "HOST_DEFAULT",
    "OP_BIND",
    "OP_CONTEXT",
    "OP_CONTEXT_REPLY",
    "OP_DONE",
    "OP_EXEC",
    "OP_HELLO",
    "OP_OUT",
    "OP_READY",
    "OP_RESET",
    "OP_SHUTDOWN",
    "OP_STARTUP",
    "OP_STATE",
    "OP_SUBCALL",
    "OP_SUBCALL_BATCHED",
    "OP_SUBCALL_REPLY",
    "CellResult",
    "ContextSource",
    "Kernel",
    "KernelError",
    "KernelLimits",
    "KernelRejectedError",
    "KernelUnavailableError",
    "KernelState",
    "KernelVar",
    "SubcallFn",
    "SubcallOutcome",
    "bounded_text",
]
