"""Typed, prose-blind normalization of one active-context window.

Compaction decisions must be reproducible, so the engine reads only typed
structure -- roles, block types, tool names, JSON arguments, error flags, exit
codes -- and never infers anything from prose. Prose is copied, excerpted, and
sampled by later folds; it is never classified.

Assistant thinking blocks are deliberate scratchpad, not commitments, so they
never enter the event stream and no fold can leak truncated deliberation that
reads like fact. :func:`count_erased_thinking_blocks` keeps that erasure
auditable in compaction details.

Every event carries two addresses: ``entry_id``, the stable address a later
reader expands from the append-only log, and ``source_entry_id``, the entry that
actually carried the content. They differ when one entry yields several events
(an assistant message with text plus tool calls).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Union

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
    content_text,
)
from tau_agent.types import JSONValue
from tau_coding.compaction.bounds import clip_utf8, utf8_bytes
from tau_coding.context_window import COMPACTION_SUMMARY_PREFIX

# Mirrors ``tau_agent.session.memory._format_branch_summary``. The test suite
# pins this constant to the live formatter so a replay-format change cannot
# silently turn a branch summary into an apparent user instruction.
BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:"
)

_MAX_DETAILS_DEPTH = 12
_MAX_DETAILS_NODES = 256
_MAX_DETAILS_ITEMS = 64
_MAX_DETAILS_STRING_BYTES = 1024
_MAX_DETAILS_BYTES = 8 * 1024

#: One active-context entry paired with the message it replays, matching
#: ``CodingSession._active_context_rows()``.
type ContextRow = tuple[str, AgentMessage]


@dataclass(frozen=True, slots=True)
class UserEvent:
    """A user turn, verbatim from the transcript."""

    index: int
    entry_id: str
    source_entry_id: str
    text: str
    kind: Literal["user"] = "user"


@dataclass(frozen=True, slots=True)
class AssistantTextEvent:
    """One assistant text block; thinking blocks are never present."""

    index: int
    entry_id: str
    source_entry_id: str
    text: str
    kind: Literal["assistant_text"] = "assistant_text"


@dataclass(frozen=True, slots=True)
class PriorSummaryEvent:
    """A summary an earlier compaction or branch already produced.

    Carried as typed structure rather than re-derivable prose: dropping it would
    make the second compaction of a session forget the first one, because the
    raw entries it summarized are already gone from the active context.
    """

    index: int
    entry_id: str
    source_entry_id: str
    summary: str
    source: Literal["compaction", "branch"]
    kind: Literal["prior_summary"] = "prior_summary"


@dataclass(frozen=True, slots=True)
class CustomMessageEvent:
    """An extension-injected message that participates in model context."""

    index: int
    entry_id: str
    source_entry_id: str
    custom_type: str
    text: str
    display: bool
    details: dict[str, JSONValue] | None = None
    kind: Literal["custom_message"] = "custom_message"


@dataclass(frozen=True, slots=True)
class ToolCallEvent:
    """A tool call as requested by the assistant."""

    index: int
    entry_id: str
    source_entry_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, JSONValue]
    kind: Literal["tool_call"] = "tool_call"


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """A tool result paired to its call by id."""

    index: int
    entry_id: str
    source_entry_id: str
    tool_call_id: str
    tool_name: str
    is_error: bool
    text: str
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class BashEvent:
    """A shell execution: the command from the originating call plus status."""

    index: int
    entry_id: str
    source_entry_id: str
    tool_call_id: str
    command: str
    is_error: bool
    exit_code: int | None = None
    error: str | None = None
    kind: Literal["bash"] = "bash"


CompactionEvent = Union[  # noqa: UP007 - Union keeps the discriminator explicit
    UserEvent,
    AssistantTextEvent,
    PriorSummaryEvent,
    CustomMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    BashEvent,
]


def first_line(text: str) -> str:
    """Return the first line of ``text`` without scanning it all."""
    stripped = text.lstrip()
    newline = stripped.find("\n")
    return stripped if newline < 0 else stripped[:newline]


class _Drop:
    """Sentinel marking a JSON value the sanitizer ceilings rejected."""


_DROP = _Drop()


@dataclass(slots=True)
class _SanitizerState:
    nodes: int = 0


def _bounded_json(value: JSONValue, state: _SanitizerState, depth: int = 0) -> JSONValue | _Drop:
    """Copy a JSON value under node, depth, and per-collection ceilings.

    Extension-owned ``details`` are untrusted in size and shape. A payload that
    exceeds a ceiling is dropped whole rather than silently truncated: a partial
    payload would reach the summary with no marker saying what is missing.
    Returning the sentinel instead of ``None`` keeps a legitimate JSON ``null``
    distinguishable from a value a ceiling rejected.
    """
    state.nodes += 1
    if state.nodes > _MAX_DETAILS_NODES:
        return _DROP
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return clip_utf8(value, _MAX_DETAILS_STRING_BYTES)
    if isinstance(value, (int, float)):
        return value
    if depth >= _MAX_DETAILS_DEPTH:
        return _DROP
    if isinstance(value, list):
        if len(value) > _MAX_DETAILS_ITEMS:
            return _DROP
        items: list[JSONValue] = []
        for item in value:
            bounded = _bounded_json(item, state, depth + 1)
            if isinstance(bounded, _Drop):
                return _DROP
            items.append(bounded)
        return items
    if isinstance(value, dict):
        if len(value) > _MAX_DETAILS_ITEMS:
            return _DROP
        bounded_items: dict[str, JSONValue] = {}
        for key in sorted(value):
            bounded = _bounded_json(value[key], state, depth + 1)
            if isinstance(bounded, _Drop):
                return _DROP
            bounded_items[key] = bounded
        return bounded_items
    return _DROP


def _bounded_details(details: JSONValue) -> dict[str, JSONValue] | None:
    """Sanitize one extension ``details`` payload, or drop it entirely."""
    if not isinstance(details, dict):
        return None
    sanitized = _bounded_json(details, _SanitizerState())
    if not isinstance(sanitized, dict):
        return None
    try:
        encoded = json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    if utf8_bytes(encoded) > _MAX_DETAILS_BYTES:
        return None
    return sanitized


def _pending_command(arguments: dict[str, JSONValue]) -> str:
    command = arguments.get("command")
    return command if isinstance(command, str) else ""


def normalize_rows(rows: Sequence[ContextRow]) -> list[CompactionEvent]:
    """Normalize active-context rows into a typed event stream.

    Tool results are paired back to their calls by id, which is how a bash
    result can carry the command that produced it. Pairing is structural; the
    text of a result never influences classification.
    """
    events: list[CompactionEvent] = []
    pending_calls: dict[str, dict[str, JSONValue]] = {}
    index = 0

    def next_index() -> int:
        """Return the 1-based position of the event being constructed."""
        nonlocal index
        index += 1
        return index

    def push(event: CompactionEvent) -> None:
        events.append(event)

    for entry_id, message in rows:
        if isinstance(message, UserMessage):
            text = content_text(message.content)
            if text.startswith(COMPACTION_SUMMARY_PREFIX):
                push(
                    PriorSummaryEvent(
                        index=next_index(),
                        entry_id=entry_id,
                        source_entry_id=entry_id,
                        summary=text.removeprefix(COMPACTION_SUMMARY_PREFIX),
                        source="compaction",
                    )
                )
            elif text.startswith(BRANCH_SUMMARY_PREFIX):
                push(
                    PriorSummaryEvent(
                        index=next_index(),
                        entry_id=entry_id,
                        source_entry_id=entry_id,
                        summary=text.removeprefix(BRANCH_SUMMARY_PREFIX),
                        source="branch",
                    )
                )
            else:
                push(
                    UserEvent(
                        index=next_index(),
                        entry_id=entry_id,
                        source_entry_id=entry_id,
                        text=text,
                    )
                )
            continue

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextContent) and block.text:
                    push(
                        AssistantTextEvent(
                            index=next_index(),
                            entry_id=entry_id,
                            source_entry_id=entry_id,
                            text=block.text,
                        )
                    )
                elif isinstance(block, ToolCall):
                    pending_calls[block.id] = dict(block.arguments)
                    push(
                        ToolCallEvent(
                            index=next_index(),
                            entry_id=entry_id,
                            source_entry_id=entry_id,
                            tool_call_id=block.id,
                            name=block.name,
                            arguments=dict(block.arguments),
                        )
                    )
            continue

        if isinstance(message, ToolResultMessage):
            pending = pending_calls.get(message.tool_call_id)
            if message.tool_name == "bash":
                command = _pending_command(pending) if pending else ""
                push(
                    BashEvent(
                        index=next_index(),
                        entry_id=entry_id,
                        source_entry_id=entry_id,
                        tool_call_id=message.tool_call_id,
                        command=command,
                        is_error=message.is_error,
                        error=message.text if message.is_error else None,
                    )
                )
            else:
                push(
                    ToolResultEvent(
                        index=next_index(),
                        entry_id=entry_id,
                        source_entry_id=entry_id,
                        tool_call_id=message.tool_call_id,
                        tool_name=message.tool_name,
                        is_error=message.is_error,
                        text=message.text,
                    )
                )
            continue

        if isinstance(message, BashExecutionMessage):
            failed = message.exit_code not in (None, 0) or message.cancelled
            push(
                BashEvent(
                    index=next_index(),
                    entry_id=entry_id,
                    source_entry_id=entry_id,
                    tool_call_id="",
                    command=message.command,
                    is_error=failed,
                    exit_code=message.exit_code,
                    error=message.output if failed and message.output else None,
                )
            )
            continue

        if isinstance(message, CustomMessage):
            push(
                CustomMessageEvent(
                    index=next_index(),
                    entry_id=entry_id,
                    source_entry_id=entry_id,
                    custom_type=message.custom_type,
                    text=content_text(message.content),
                    display=message.display,
                    details=_bounded_details(message.details),
                )
            )
            continue

        if isinstance(message, (CompactionSummaryMessage, BranchSummaryMessage)):
            push(
                PriorSummaryEvent(
                    index=next_index(),
                    entry_id=entry_id,
                    source_entry_id=entry_id,
                    summary=message.summary,
                    source="branch" if isinstance(message, BranchSummaryMessage) else "compaction",
                )
            )

    return events


def count_erased_thinking_blocks(rows: Sequence[ContextRow]) -> int:
    """Count assistant thinking parts erased by normalization.

    The walk mirrors the selection in :func:`normalize_rows`, so the count
    reported in compaction details cannot disagree with what was dropped.
    """
    total = 0
    for _entry_id, message in rows:
        if not isinstance(message, AssistantMessage):
            continue
        total += sum(1 for block in message.content if isinstance(block, ThinkingContent))
    return total
