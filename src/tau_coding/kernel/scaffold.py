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

from collections.abc import Sequence

LLM_QUERY = "llm_query"
LLM_QUERY_BATCHED = "llm_query_batched"
RLM_QUERY = "rlm_query"
RLM_QUERY_BATCHED = "rlm_query_batched"
CONTEXT_INDEX = "context_index"
CONTEXT_OPEN = "context_open"
RET = "ret"
SHOW_VARS = "SHOW_VARS"

RECURSION_BINDINGS: tuple[str, ...] = (RLM_QUERY, RLM_QUERY_BATCHED)
"""
Names bound only when the host wired a recursion.

A sub-call is one provider request; a recursion is a child with a kernel of
its own. The second is opt-in per session, so it is not part of the
always-bound set: the tool description and the kernel reporting its own
bindings both have to say what is actually there, and a name advertised but
missing is a NameError for whoever trusted the documentation.
"""

BINDINGS: tuple[str, ...] = (
    LLM_QUERY,
    LLM_QUERY_BATCHED,
    *RECURSION_BINDINGS,
    CONTEXT_INDEX,
    CONTEXT_OPEN,
    RET,
    SHOW_VARS,
)
"""Every name the kernel may bind, in documentation order."""

BINDING_HELP: tuple[str, ...] = (
    "llm_query(prompt, model=None) -> str -- ask the model for one plain "
    "completion; the reply is text, and the big context stays where it is.",
    "llm_query_batched(prompts, model=None) -> list[str | None] -- the same "
    "call for many prompts at once; a failed entry is None so one bad prompt "
    "cannot sink the batch.",
    "rlm_query(task, model=None) -> str -- hand a task to a child that works "
    "in its own kernel: it writes code, sees the output, and answers when "
    "done. At the depth ceiling it degrades to a single plain completion.",
    "rlm_query_batched(tasks, model=None) -> list[str | None] -- several tasks "
    "at once, each with its own child; a task that fails comes back as None.",
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
    "Use rlm_query when the sub-task has to compute: it gets a kernel and turns "
    "of its own. Use llm_query when reading and answering is enough.",
    "End the cell with ret({...}) so the result is data, not prose to re-read.",
)

RESERVED = BINDINGS + ("__name__", "__doc__", "__builtins__", "__spec__", "__loader__")
"""Namespace entries the kernel re-asserts after every cell."""


def help_block(bindings: Sequence[str] = BINDINGS) -> str:
    """
    Return the help lines for the names that are actually bound.

    A description that promises rlm_query to a session without recursion
    would send the model into a NameError, so a tool renders this from the
    names its kernel really bound rather than from the catalogue.
    """
    bound = set(bindings)
    return "\n".join(line for line in BINDING_HELP if line.split("(", 1)[0].split(" ")[0] in bound)


__all__ = [
    "BINDINGS",
    "BINDING_HELP",
    "CONTEXT_INDEX",
    "CONTEXT_OPEN",
    "GUIDELINES",
    "LLM_QUERY",
    "LLM_QUERY_BATCHED",
    "RECURSION_BINDINGS",
    "RESERVED",
    "RET",
    "RLM_QUERY",
    "RLM_QUERY_BATCHED",
    "SHOW_VARS",
    "help_block",
]
