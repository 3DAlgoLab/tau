"""Names the kernel injects, and the discipline they encode.

The kernel child defines its bindings as ordinary closures (see
:mod:`tau_coding.kernel.child`); nothing callable crosses the process boundary.
What *does* need to cross is the model's expectations, so the binding names and
their help text live here where the code-mode tool can render them into a prompt
and the kernel can restore them after a cell overwrites them.

The rule the bindings encode: work stays in the kernel, only decisions come back
out. ``print`` is for the operator's eye and is byte-bounded; ``ret()`` is the
one channel for a typed result.
"""

from __future__ import annotations

LLM_QUERY = "llm_query"
LLM_QUERY_BATCHED = "llm_query_batched"
CONTEXT_INDEX = "context_index"
CONTEXT_OPEN = "context_open"
RET = "ret"
SHOW_VARS = "SHOW_VARS"

BINDINGS: tuple[str, ...] = (
    LLM_QUERY,
    LLM_QUERY_BATCHED,
    CONTEXT_INDEX,
    CONTEXT_OPEN,
    RET,
    SHOW_VARS,
)
"""Names reserved in the kernel namespace, in documentation order."""

BINDING_HELP: tuple[str, ...] = (
    "llm_query(prompt, model=None) -> str -- ask the model for one plain "
    "completion; the reply is text, and the big context stays where it is.",
    "llm_query_batched(prompts, model=None) -> list[str | None] -- the same "
    "call for many prompts at once; a failed entry is None so one bad prompt "
    "cannot sink the batch.",
    "context_index(query=None) -> dict -- describe the active session context "
    "in bounded rows (address, entry id, role, bytes, first line) without "
    "loading any of it; the addresses are the 1-based (#N) a compaction "
    "summary cites.",
    "context_open(*refs) -> str -- materialize addressed context entries as a "
    'JSONL file and return its path. Refs: "#N", "A-B", "last:N", '
    '"first:N", "all", or an entry id. Each row carries its address, '
    "entry id, role, text, and how many bytes were clipped.",
    "ret(value) -- report the cell's result. Only JSON data crosses the boundary; last call wins.",
    f"{SHOW_VARS}() -> dict -- names, types and sizes bound in this kernel. "
    "Values stay in the kernel.",
)

GUIDELINES: tuple[str, ...] = (
    "Keep large data in the kernel: slice, count and grep it there, and print "
    "only the part you are deciding on.",
    "Files are handles, not payloads: use the bound path and read what you need.",
    "Session history is a handle too: context_index() to see it and "
    "context_open(refs) to read it, and only the slice you chose comes back.",
    "Fan out with llm_query_batched rather than a loop of single calls.",
    "End the cell with ret({...}) so the result is data, not prose to re-read.",
)

RESERVED = BINDINGS + ("__name__", "__doc__", "__builtins__", "__spec__", "__loader__")
"""Namespace entries the kernel re-asserts after every cell."""


def help_block() -> str:
    """Return the bindings rendered for a tool description."""
    return "\n".join(BINDING_HELP)


__all__ = [
    "BINDINGS",
    "BINDING_HELP",
    "CONTEXT_INDEX",
    "CONTEXT_OPEN",
    "GUIDELINES",
    "LLM_QUERY",
    "LLM_QUERY_BATCHED",
    "RESERVED",
    "RET",
    "SHOW_VARS",
    "help_block",
]
