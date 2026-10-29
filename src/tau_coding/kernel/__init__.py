"""A code-mode execution environment: one persistent Python kernel per session.

This package is mechanism only -- nothing in it is wired into a session or a
tool. The code-mode extension at ``examples/extensions/`` is what puts these
primitives in front of a model, and deleting that file takes the capability away
again.

The shape follows the original Recursive Language Models design: the harness
gives the model a code environment, the input lives as data the code can slice,
and sub-completions are ordinary function calls inside that environment. What
this layer guarantees is the boundary the design depends on: model code runs in
its own process, its output is byte-bounded and counted, its sub-calls are
concurrency-bounded, and its namespace is inventoryable without ever being read.
"""

from tau_coding.kernel.frames import (
    MAX_FRAME_BYTES,
    FrameError,
    FrameTooLargeError,
    decode_frame,
    encode_frame,
    read_frame,
    read_frame_sync,
    send_frame_sync,
    write_frame,
)
from tau_coding.kernel.process import (
    CONTROL_TIMEOUT_SECONDS,
    DEFAULT_CELL_OUTPUT_BYTES,
    DEFAULT_CELL_TIMEOUT_SECONDS,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ProcessKernel,
    looks_secret,
)
from tau_coding.kernel.protocol import (
    CellResult,
    Kernel,
    KernelError,
    KernelLimits,
    KernelRejectedError,
    KernelState,
    KernelUnavailableError,
    KernelVar,
    SubcallFn,
    SubcallOutcome,
)
from tau_coding.kernel.scaffold import BINDING_HELP, BINDINGS, GUIDELINES

__all__ = [
    "BINDINGS",
    "BINDING_HELP",
    "CONTROL_TIMEOUT_SECONDS",
    "DEFAULT_CELL_OUTPUT_BYTES",
    "DEFAULT_CELL_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "GUIDELINES",
    "MAX_FRAME_BYTES",
    "CellResult",
    "FrameError",
    "FrameTooLargeError",
    "Kernel",
    "KernelError",
    "KernelLimits",
    "KernelRejectedError",
    "KernelState",
    "KernelUnavailableError",
    "KernelVar",
    "ProcessKernel",
    "SubcallFn",
    "SubcallOutcome",
    "decode_frame",
    "encode_frame",
    "looks_secret",
    "read_frame",
    "read_frame_sync",
    "send_frame_sync",
    "write_frame",
]
