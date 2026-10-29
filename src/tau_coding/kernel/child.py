"""The kernel child process: one persistent namespace, one control channel.

Run as ``python -m tau_coding.kernel.child``. The host gives this process a
localhost channel address in its environment; everything else travels as frames.
The process lives for the whole session so model code can build state across
cells -- that persistence is what makes an RLM-style traversal cheap, since the
input never has to re-enter a prompt.

Three invariants:

* **Cell errors are data.** Anything a cell raises is formatted and reported; it
  never takes the kernel down, so one bad cell cannot destroy the session.
* **Output is bounded.** Streamed text is capped in UTF-8 bytes and the omitted
  remainder is counted, so a runaway loop says how much it swallowed.
* **Only the control channel is trusted.** A second, unrelated connection can
  join the host's port but is refused there by the handshake token.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import platform
import socket
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from typing import Any

from tau_coding.kernel.frames import (
    FrameError,
    read_frame_sync,
    send_frame_sync,
)
from tau_coding.kernel.protocol import (
    ENV_CONNECT_TIMEOUT,
    ENV_HOST,
    ENV_PORT,
    ENV_TOKEN,
    HOST_DEFAULT,
    OP_BIND,
    OP_CONTEXT,
    OP_CONTEXT_REPLY,
    OP_DONE,
    OP_EXEC,
    OP_HELLO,
    OP_OUT,
    OP_READY,
    OP_RESET,
    OP_RLM,
    OP_RLM_BATCHED,
    OP_RLM_REPLY,
    OP_SHUTDOWN,
    OP_STARTUP,
    OP_STATE,
    OP_SUBCALL,
    OP_SUBCALL_BATCHED,
    OP_SUBCALL_REPLY,
    bounded_text,
)
from tau_coding.kernel.scaffold import (
    BINDINGS,
    CONTEXT_INDEX,
    CONTEXT_OPEN,
    LLM_QUERY,
    LLM_QUERY_BATCHED,
    RECURSION_BINDINGS,
    RESERVED,
    RET,
    RLM_QUERY,
    RLM_QUERY_BATCHED,
    SHOW_VARS,
)

CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_CELL_OUTPUT_BYTES = 64 * 1024
MAX_ERROR_BYTES = 8 * 1024
MAX_RET_BYTES = 32 * 1024
_FLUSH_BYTES = 4096

_RLIMITS: dict[str, str] = {
    "cpu_seconds": "RLIMIT_CPU",
    "address_space_mb": "RLIMIT_AS",
    "file_size_mb": "RLIMIT_FSIZE",
    "open_files": "RLIMIT_NOFILE",
    "max_processes": "RLIMIT_NPROC",
}
_RLIMIT_MB_TO_BYTES = 1024 * 1024


class KernelSubcallError(RuntimeError):
    """Raised inside a cell when the host could not serve a sub-call."""


class _Channel:
    """One framed connection to the host, safe across threads the cell spawns."""

    def __init__(self, sock: socket.socket, reader: io.BufferedReader) -> None:
        self._sock = sock
        self._reader = reader
        self._write_lock = threading.Lock()
        self._request_lock = threading.Lock()
        self._request_seq = 0
        self.cell_id = ""

    def send(self, payload: dict[str, Any]) -> bool:
        """Send one frame, tagging it with the cell that produced it."""
        if self.cell_id:
            payload = {**payload, "cell_id": self.cell_id}
        with self._write_lock:
            try:
                send_frame_sync(self._sock, payload)
            except (OSError, FrameError):
                return False
        return True

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a sub-call request and block for its reply.

        Sub-calls are serialized: the reply is read here, outside the main frame
        loop, so two threads cannot race for it. Parallelism comes from
        :func:`llm_query_batched`, which asks for many completions in one
        request and lets the host fan them out.
        """
        with self._request_lock:
            self._request_seq += 1
            req_id = f"r{self._request_seq}"
            if not self.send({**payload, "req_id": req_id}):
                raise KernelSubcallError("kernel channel closed")
            try:
                reply = read_frame_sync(self._reader)
            except FrameError as exc:
                raise KernelSubcallError(f"sub-call reply was malformed: {exc}") from exc
            if reply is None:
                raise KernelSubcallError("kernel channel closed mid-request")
            if (
                reply.get("op") not in (OP_SUBCALL_REPLY, OP_CONTEXT_REPLY, OP_RLM_REPLY)
                or reply.get("req_id") != req_id
            ):
                raise KernelSubcallError("unexpected frame while waiting for a host reply")
            return reply


class _Budget:
    """A byte allowance that keeps a head and a tail, and counts the middle out.

    A flat head allowance is what made a child believe a big print *was* the file:
    it saw the first 64 KiB, the rest was dropped inside the kernel, and the row it
    was looking for sat past the cut. So the allowance is split -- most of it spent
    as the cell streams, the rest reserved and refilled from the end, so the last
    thing a cell prints survives as well as the first. Every byte given up is
    counted, and the tail frame names the count.
    """

    def __init__(self, limit: int, *, head_fraction: float = 0.75) -> None:
        self.head_limit = max(1, int(limit * head_fraction))
        self.tail_limit = max(1, limit - self.head_limit)
        self.omitted_bytes = 0
        self._head_used = 0
        self._tail: list[str] = []
        self._tail_bytes = 0

    def allow(self, text: str) -> str:
        """Return the part of ``text`` that belongs to the head; hold back the rest."""
        if not text:
            return ""
        room = self.head_limit - self._head_used
        if room <= 0:
            self._hold(text)
            return ""
        clipped, _omitted = bounded_text(text, room)
        self._head_used += len(clipped.encode("utf-8"))
        if len(clipped) < len(text):
            self._hold(text[len(clipped) :])
        return clipped

    def _hold(self, text: str) -> None:
        """Keep the most recent ``tail_limit`` bytes; account for everything older."""
        if not text:
            return
        self._tail.append(text)
        self._tail_bytes += len(text.encode("utf-8"))
        while self._tail_bytes > self.tail_limit:
            excess = self._tail_bytes - self.tail_limit
            oldest = self._tail[0]
            size = len(oldest.encode("utf-8"))
            if size <= excess:
                self._tail.pop(0)
                self.omitted_bytes += size
                self._tail_bytes -= size
                continue
            keep = size - excess
            dropped, kept = oldest[:keep], oldest[keep:]
            self._tail[0] = kept
            self.omitted_bytes += len(dropped.encode("utf-8"))
            self._tail_bytes -= len(dropped.encode("utf-8"))

    def take_tail(self) -> str:
        """Return the reserved tail once, with the middle it stands in for named."""
        if not self._tail:
            return ""
        text = "".join(self._tail)
        dropped = self.omitted_bytes
        self._tail.clear()
        self._tail_bytes = 0
        if not dropped:
            return text
        marker = f"[middle omitted: {dropped:,} UTF-8 bytes; this is the end of the output]\n"
        # The marker is charged to the tail, so the cell's total observation never
        # exceeds the budget the caller asked for: announcing the loss may not be
        # what pushes a run back over the line.
        room = max(0, self.tail_limit - len(marker.encode("utf-8")))
        clipped, extra = bounded_text(text, room)
        self.omitted_bytes = dropped + extra
        return f"{marker}{clipped}"


class _FrameStream(io.TextIOBase):
    """``sys.stdout``/``sys.stderr`` replacement that forwards frames to the host.

    The kernel's real standard streams stay free for C-level writes and for
    subprocesses the cell starts, which the host drains separately.
    """

    def __init__(self, channel: _Channel, name: str, budget: _Budget) -> None:
        self._channel = channel
        self._name = name
        self._budget = budget
        self._parts: list[str] = []
        self._pending = 0

    def writable(self) -> bool:
        return True

    def write(self, text: str) -> int:
        if not text:
            return 0
        allowed = self._budget.allow(text)
        self._parts.append(allowed)
        self._pending += len(allowed)
        if self._pending >= _FLUSH_BYTES:
            self.flush()
        return len(text)

    def flush(self) -> None:
        if not self._parts:
            return
        text = "".join(self._parts)
        self._parts.clear()
        self._pending = 0
        self._channel.send({"op": OP_OUT, "stream": self._name, "text": text})

    def close(self) -> None:
        self.flush()
        tail = self._budget.take_tail()
        if tail:
            self._channel.send({"op": OP_OUT, "stream": self._name, "text": tail})


def _format_exception() -> str:
    """Render the active exception, keeping the tail when it is too long."""
    text = traceback.format_exc()
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_ERROR_BYTES:
        return text
    tail = encoded[-MAX_ERROR_BYTES:].decode("utf-8", errors="ignore")
    dropped = len(encoded) - len(tail.encode("utf-8"))
    return f"[omitted {dropped} UTF-8 bytes of traceback]\n{tail}"


def _apply_limits(limits: Mapping[str, Any]) -> list[str]:
    """Apply ``setrlimit`` ceilings in the child; report what was not applied."""
    notes: list[str] = []
    try:
        import resource
    except ImportError:  # pragma: no cover - POSIX-only platforms
        return ["resource limits are unavailable on this platform"]
    for name, attr in _RLIMITS.items():
        raw = limits.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            notes.append(f"{name}: invalid value")
            continue
        which = getattr(resource, attr, None)
        if which is None:
            notes.append(f"{attr}: not supported here")
            continue
        if name.endswith("_mb"):
            value *= _RLIMIT_MB_TO_BYTES
        try:
            current, hard = resource.getrlimit(which)
            target = value if current == resource.RLIM_INFINITY else min(current, value)
            if hard != resource.RLIM_INFINITY:
                target = min(target, hard)
            resource.setrlimit(which, (target, max(target, 1)))
        except (OSError, ValueError) as exc:  # pragma: no cover - platform specific
            notes.append(f"{attr}: {exc}")
    return notes


def _size_of(value: object) -> int:
    try:
        return sys.getsizeof(value)
    except TypeError:  # pragma: no cover - exotic objects
        return 0


def _length_of(value: object) -> int | None:
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return None


def _inventory(ns: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe bound names without touching their values."""
    rows: list[dict[str, Any]] = []
    for name, value in list(ns.items()):
        if name.startswith("_") or name in RESERVED:
            continue
        length = _length_of(value)
        rows.append(
            {
                "name": name,
                "type": type(value).__name__,
                "length": length,
                "bytes": _size_of(value),
            }
        )
    rows.sort(key=lambda row: str(row["name"]))
    return rows


def _safe_json(value: Any) -> tuple[Any, str | None]:
    """Return ``value`` if it is data, otherwise the reason it is not."""
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        return None, f"ret() needs JSON data, not {type(value).__name__}: {exc}"
    if len(encoded) > MAX_RET_BYTES:
        return None, f"ret() result is {len(encoded)} bytes, the limit is {MAX_RET_BYTES}"
    return value, None


class _Kernel:
    """The child's namespace and frame dispatch."""

    def __init__(
        self,
        channel: _Channel,
        max_cell_output_bytes: int,
        bindings: Sequence[str] | None = None,
    ) -> None:
        self._channel = channel
        self._max_cell_output_bytes = max(1, max_cell_output_bytes)
        # The host decides which optional bindings this namespace gets, and the
        # child reports back what it actually bound: a name advertised in the
        # ready frame but missing here is a NameError for whoever trusted it.
        self.bindings: tuple[str, ...] = (
            tuple(bindings)
            if bindings is not None
            else tuple(name for name in BINDINGS if name not in RECURSION_BINDINGS)
        )
        self._ns: dict[str, Any] = {}
        self._reassert()

    def _reassert(self) -> None:
        """(Re)bind the scaffold names so a cell cannot clobber them."""
        ns = self._ns
        ns.setdefault("__name__", "__tau_kernel__")
        ns.setdefault("__doc__", None)
        ns["__builtins__"] = builtins
        ns[LLM_QUERY] = self._llm_query
        ns[LLM_QUERY_BATCHED] = self._llm_query_batched
        if RLM_QUERY in self.bindings:
            ns[RLM_QUERY] = self._rlm_query
        if RLM_QUERY_BATCHED in self.bindings:
            ns[RLM_QUERY_BATCHED] = self._rlm_query_batched
        ns[CONTEXT_INDEX] = self._context_index
        ns[CONTEXT_OPEN] = self._context_open
        ns[SHOW_VARS] = self._show_vars
        ns[RET] = self._ret_noop

    def _fresh(self) -> None:
        self._ns = {}
        self._reassert()

    def _show_vars(self) -> dict[str, str]:
        """Report names and types bound in this kernel; values stay here."""
        return {
            name: type(value).__name__
            for name, value in self._ns.items()
            if not name.startswith("_")
        }

    def _context_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Ask the host for context rows or a slice path, serialized like a sub-call."""
        reply = self._channel.request({"op": OP_CONTEXT, **payload})
        error = reply.get("error")
        if error is not None:
            raise KernelSubcallError(str(error))
        return reply

    def _context_index(self, query: str | None = None) -> dict[str, Any]:
        if query is not None and not isinstance(query, str):
            raise TypeError("context_index(query) needs a string or None")
        return self._context_request({"action": "index", "query": query})

    def _context_open(self, *refs: str) -> str:
        named = [ref for ref in refs if isinstance(ref, str) and ref.strip()]
        if not named:
            raise TypeError(
                "context_open(refs...) needs references such as '#3', '12-20', "
                "'last:5', 'all', or an entry id"
            )
        reply = self._context_request({"action": "open", "refs": named})
        path = reply.get("path")
        if not isinstance(path, str) or not path:
            raise KernelSubcallError("the context reply carried no file path")
        note = reply.get("note")
        if isinstance(note, str) and note:
            print(f"[context] {note}")
        return path

    def _ret_noop(self, value: object) -> None:  # pragma: no cover - replaced per cell
        del value

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        if not isinstance(prompt, str):
            raise TypeError("llm_query(prompt) needs a string prompt")
        reply = self._channel.request({"op": OP_SUBCALL, "prompt": prompt, "model": model})
        error = reply.get("error")
        if error is not None:
            raise KernelSubcallError(str(error))
        text = reply.get("text", "")
        return text if isinstance(text, str) else str(text)

    def _llm_query_batched(self, prompts: list[str], model: str | None = None) -> list[str | None]:
        if not isinstance(prompts, list):
            raise TypeError("llm_query_batched(prompts) needs a list of strings")
        reply = self._channel.request(
            {"op": OP_SUBCALL_BATCHED, "prompts": list(prompts), "model": model}
        )
        error = reply.get("error")
        if error is not None:
            raise KernelSubcallError(str(error))
        responses = reply.get("responses", [])
        if not isinstance(responses, list):
            raise KernelSubcallError("sub-call batch reply was malformed")
        return [None if item is None else str(item) for item in responses]

    def _rlm_query(self, task: str, model: str | None = None) -> str:
        if not isinstance(task, str):
            raise TypeError("rlm_query(task) needs a string task")
        reply = self._channel.request({"op": OP_RLM, "task": task, "model": model})
        error = reply.get("error")
        if error is not None:
            raise KernelSubcallError(str(error))
        text = reply.get("text", "")
        return text if isinstance(text, str) else str(text)

    def _rlm_query_batched(self, tasks: list[str], model: str | None = None) -> list[str | None]:
        if not isinstance(tasks, list):
            raise TypeError("rlm_query_batched(tasks) needs a list of strings")
        reply = self._channel.request({"op": OP_RLM_BATCHED, "tasks": list(tasks), "model": model})
        error = reply.get("error")
        if error is not None:
            raise KernelSubcallError(str(error))
        responses = reply.get("responses", [])
        if not isinstance(responses, list):
            raise KernelSubcallError("recursion batch reply was malformed")
        return [None if item is None else str(item) for item in responses]

    def execute(self, frame: Mapping[str, Any]) -> None:
        """Run one cell and report bounded observations."""
        cell_id = str(frame.get("cell_id", ""))
        code = str(frame.get("code", ""))
        self._channel.cell_id = cell_id
        returned: dict[str, Any] = {}

        def ret(value: object) -> None:
            returned["value"] = value

        self._ns[RET] = ret
        budgets = (_Budget(self._max_cell_output_bytes), _Budget(self._max_cell_output_bytes))
        previous = (sys.stdout, sys.stderr)
        streams = (
            _FrameStream(self._channel, "stdout", budgets[0]),
            _FrameStream(self._channel, "stderr", budgets[1]),
        )
        sys.stdout, sys.stderr = streams
        started = time.perf_counter()
        ok, error = True, None
        try:
            exec(compile(code, f"<cell {cell_id}>", "exec"), self._ns, self._ns)
        except SystemExit as exc:
            ok = False
            error = f"SystemExit({exc.code}) was raised; sys.exit() does not end the kernel"
        except BaseException:  # the cell is model-written: report, keep running
            ok = False
            error = _format_exception()
        finally:
            # close(), not flush(): the reserved tail of each stream is released
            # there, and it has to reach the host before the done frame does.
            for stream in streams:
                stream.close()
            sys.stdout, sys.stderr = previous
            self._channel.cell_id = ""
            self._reassert()

        done: dict[str, Any] = {
            "op": OP_DONE,
            "cell_id": cell_id,
            "ok": ok,
            "ms": int((time.perf_counter() - started) * 1000),
            "stdout_omitted": budgets[0].omitted_bytes,
            "stderr_omitted": budgets[1].omitted_bytes,
        }
        if error is not None:
            done["error"] = error
        if "value" in returned:
            value, ret_error = _safe_json(returned["value"])
            done["returned"] = True
            if ret_error is None:
                done["ret"] = value
            else:
                done["ret_error"] = ret_error
        self._channel.send(done)

    def handle(self, frame: Mapping[str, Any]) -> bool:
        """Dispatch one control frame; return ``False`` to exit the loop."""
        op = frame.get("op")
        cell_id = str(frame.get("cell_id", ""))
        if op == OP_EXEC:
            self.execute(frame)
        elif op == OP_STATE:
            self._channel.send(
                {"op": OP_DONE, "cell_id": cell_id, "ok": True, "vars": _inventory(self._ns)}
            )
        elif op == OP_RESET:
            self._fresh()
            self._channel.send({"op": OP_DONE, "cell_id": cell_id, "ok": True})
        elif op == OP_BIND:
            self._bind(frame, cell_id)
        elif op == OP_SHUTDOWN:
            return False
        else:
            self._channel.send(
                {
                    "op": OP_DONE,
                    "cell_id": cell_id,
                    "ok": False,
                    "error": f"unknown kernel operation: {op!r}",
                }
            )
        return True

    def _bind(self, frame: Mapping[str, Any], cell_id: str) -> None:
        """Expose a file path as a variable the cell can read."""
        name = str(frame.get("name", ""))
        path = str(frame.get("path", ""))
        if not name.isidentifier() or name in RESERVED:
            self._channel.send(
                {"op": OP_DONE, "cell_id": cell_id, "ok": False, "error": f"cannot bind {name!r}"}
            )
            return
        self._ns[name] = path
        self._channel.send({"op": OP_DONE, "cell_id": cell_id, "ok": True})


def _connect_timeout() -> float:
    """How long the kernel waits to *reach* the host -- never for an answer.

    The host leaves this alone; a test shrinks it to prove that an answer landing
    later than the connect timeout still reaches its cell.
    """

    raw = os.environ.get(ENV_CONNECT_TIMEOUT, "")
    try:
        value = float(raw)
    except ValueError:
        return CONNECT_TIMEOUT_SECONDS
    return value if value > 0 else CONNECT_TIMEOUT_SECONDS


def _fatal(message: str) -> None:
    print(f"tau kernel: {message}", file=sys.__stderr__, flush=True)


def main() -> int:
    """Connect, run cells until told to stop, and report setup problems loudly."""
    try:
        host = os.environ.get(ENV_HOST, HOST_DEFAULT)
        port = int(os.environ[ENV_PORT])
        token = os.environ[ENV_TOKEN]
    except (KeyError, ValueError):
        _fatal("the host did not pass a channel address")
        return 2

    try:
        sock = socket.create_connection((host, port), timeout=_connect_timeout())
    except OSError as exc:
        _fatal(f"cannot reach the host on {host}:{port}: {exc}")
        return 2
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # ``create_connection`` leaves the socket timed, and that timeout would then
    # apply to every frame read: a sub-call whose provider turn outlives it dies
    # inside the cell *after* the host has paid for the completion, and the reply
    # still in flight desynchronises the next frame read. Reads are blocking from
    # here; the host's per-cell budget is what bounds a wedged host, because it is
    # the host that kills this process.
    sock.settimeout(None)
    reader = sock.makefile("rb")
    channel = _Channel(sock, reader)

    try:
        send_frame_sync(sock, {"op": OP_HELLO, "token": token, "pid": os.getpid()})
    except (OSError, FrameError) as exc:
        _fatal(f"cannot open the channel with the host: {exc}")
        return 2
    startup = read_frame_sync(reader)
    if startup is None or startup.get("op") != OP_STARTUP:
        _fatal("the host closed the channel before startup")
        return 2

    limits = startup.get("limits")
    notes = _apply_limits(limits) if isinstance(limits, Mapping) else []
    max_output = startup.get("max_cell_output_bytes", DEFAULT_CELL_OUTPUT_BYTES)
    try:
        max_output_bytes = int(max_output)
    except (TypeError, ValueError):
        max_output_bytes = DEFAULT_CELL_OUTPUT_BYTES

    raw_bindings = startup.get("bindings")
    bindings = (
        tuple(name for name in raw_bindings if isinstance(name, str))
        if isinstance(raw_bindings, (list, tuple))
        else None
    )
    kernel = _Kernel(channel, max_output_bytes, bindings)
    channel.send(
        {
            "op": OP_READY,
            "pid": os.getpid(),
            "python": platform.python_version(),
            "limits": notes,
            "bindings": list(kernel.bindings),
        }
    )

    while True:
        try:
            frame = read_frame_sync(reader)
        except FrameError as exc:
            _fatal(f"malformed control frame: {exc}")
            return 3
        if frame is None:
            return 0
        if not kernel.handle(frame):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
