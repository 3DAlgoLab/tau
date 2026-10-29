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
RET = "ret"
SHOW_VARS = "SHOW_VARS"

BINDINGS: tuple[str, ...] = (LLM_QUERY, LLM_QUERY_BATCHED, RET, SHOW_VARS)
"""Names reserved in the kernel namespace, in documentation order."""

BINDING_HELP: tuple[str, ...] = (
    "llm_query(prompt, model=None) -> str -- ask the model for one plain "
    "completion; the reply is text, and the big context stays where it is.",
    "llm_query_batched(prompts, model=None) -> list[str | None] -- the same "
    "call for many prompts at once; a failed entry is None so one bad prompt "
    "cannot sink the batch.",
    "ret(value) -- report the cell's result. Only JSON data crosses the boundary; last call wins.",
    f"{SHOW_VARS}() -> dict -- names, types and sizes bound in this kernel. "
    "Values stay in the kernel.",
)

GUIDELINES: tuple[str, ...] = (
    "Keep large data in the kernel: slice, count and grep it there, and print "
    "only the part you are deciding on.",
    "Files are handles, not payloads: use the bound path and read what you need.",
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
    "GUIDELINES",
    "LLM_QUERY",
    "LLM_QUERY_BATCHED",
    "RESERVED",
    "RET",
    "SHOW_VARS",
    "help_block",
]
