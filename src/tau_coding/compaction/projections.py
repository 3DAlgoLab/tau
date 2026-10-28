"""Deterministic section folds over a typed compaction event stream.

Each fold is a pure function of the event stream, so the same session always
compacts to the same summary. Nothing here inspects prose to decide what
matters: sections are built from roles, tool names, JSON arguments, and error
flags. This is graded decay -- the oldest turns collapse to one line, the most
recent events stay as a collapsed transcript with stable ``(#N)`` references,
and the last action is surfaced in ``[Current Status]``.

Salience is recomputed rather than remembered. :func:`project_outstanding`
re-derives which errors are still open by matching each failure's typed
identity against a later success, so no state has to survive between
compactions.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from tau_agent.types import JSONValue
from tau_coding.compaction.bounds import (
    AddressedSample,
    clip_utf8,
    omission_line,
    sample_addressed,
    sample_addressed_from,
)
from tau_coding.compaction.dialogue import project_dialogue, recent_dialogue
from tau_coding.compaction.events import (
    AssistantTextEvent,
    BashEvent,
    CompactionEvent,
    CustomMessageEvent,
    PriorSummaryEvent,
    ToolCallEvent,
    ToolResultEvent,
    UserEvent,
    first_line,
)

# Ceilings are byte ceilings, matching how the renderer budgets the summary.
MAX_LINE_BYTES = 140
MAX_USER_ONELINER_BYTES = 120
MAX_EARLIER_USER_BYTES = 80
MAX_STATUS_LINE_BYTES = 140
MAX_TRANSCRIPT_LINE_BYTES = 100
MAX_TRANSCRIPT_CMD_BYTES = 80
MAX_DETAIL_BYTES = 160

#: Tau tools that name a path in their arguments.
FILE_TOOLS = frozenset({"read", "edit", "write"})
#: Tau tools whose success means a file's content changed.
MODIFYING_TOOLS = frozenset({"edit", "write"})

MAX_REQUESTS = 24
MAX_FILES_PER_KIND = 24
MAX_ACTIVITY = 48
MAX_OUTSTANDING = 32
#: Open errors outrank resolved ones, so they get the larger share.
MAX_UNRESOLVED = 24
MAX_RESOLVED = MAX_OUTSTANDING - MAX_UNRESOLVED
MAX_EARLIER_TURNS = 32
TRANSCRIPT_WINDOW = 40

Outcome = Literal["succeeded", "failed"]


@dataclass(frozen=True, slots=True)
class Sections:
    """Rendered lines per summary section, in render order.

    ``transcript`` is rendered last but allocated before the footer, and an
    empty section is skipped by the renderer rather than emitted as a bare
    header.
    """

    dialogue: tuple[str, ...] = ()
    requests: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    activity: tuple[str, ...] = ()
    outstanding: tuple[str, ...] = ()
    earlier_turns: tuple[str, ...] = ()
    status: tuple[str, ...] = ()
    transcript: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectionOmittedCounts:
    """Items dropped by each fold's sampler.

    The counts pair with the address markers in the rendered sections: an
    omission is only acceptable when the summary says how many items it covers
    and which source entries address them.
    """

    requests: int = 0
    files: int = 0
    activity: int = 0
    outstanding: int = 0
    earlier_turns: int = 0
    transcript: int = 0
    dialogue_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Sections plus the accounting a compaction stores as provenance."""

    sections: Sections
    omitted: ProjectionOmittedCounts


@dataclass(frozen=True, slots=True)
class _Section:
    lines: tuple[str, ...]
    omitted: int = 0


@dataclass(frozen=True, slots=True)
class Operation:
    """One tool operation paired back to the call that requested it.

    Operation identity -- tool, path, or command -- is what lets a later success
    resolve an earlier failure without reading the error text.
    """

    index: int
    entry_id: str
    tool: str
    arguments: dict[str, JSONValue]
    outcome: Outcome
    error: str | None = None

    @property
    def ref(self) -> str:
        """Return the tool reference used in rendered one-liners."""
        return self.tool


def truncate(text: str, max_bytes: int) -> str:
    """Collapse whitespace and clip to a UTF-8 byte ceiling.

    One-liners are compared and sampled after this fold, so line breaks and
    runs of spaces in the source cannot change what a section contains.
    """
    return clip_utf8(" ".join(text.split()), max_bytes)


def quoted(text: str, max_bytes: int) -> str:
    """Return a JSON-quoted one-liner so quoting cannot be ambiguous."""
    return json.dumps(truncate(text, max_bytes), ensure_ascii=False)


def _path_of(arguments: dict[str, JSONValue]) -> str | None:
    for key in ("path", "file", "dir"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _command_of(arguments: dict[str, JSONValue]) -> str | None:
    value = arguments.get("command")
    if isinstance(value, str) and value.strip():
        return value
    return None


def _primary_of(arguments: dict[str, JSONValue]) -> str | None:
    """Return the argument that identifies a call, without reading prose."""
    for key in ("command", "pattern", "query", "id", "name", "action"):
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _calls_by_id(events: Sequence[CompactionEvent]) -> dict[str, ToolCallEvent]:
    return {event.tool_call_id: event for event in events if isinstance(event, ToolCallEvent)}


def collect_operations(events: Sequence[CompactionEvent]) -> list[Operation]:
    """Collect tool operations in chronological order.

    A result with no matching call is dropped: without its arguments the
    operation has no identity, so it cannot be sampled by address or resolved by
    a later success.
    """
    calls = _calls_by_id(events)
    operations: list[Operation] = []
    for event in events:
        if isinstance(event, ToolResultEvent):
            if event.tool_name == "bash":
                continue
            call = calls.get(event.tool_call_id)
            if call is None:
                continue
            operations.append(
                Operation(
                    index=event.index,
                    entry_id=call.entry_id,
                    tool=call.name,
                    arguments=call.arguments,
                    outcome="failed" if event.is_error else "succeeded",
                    error=event.text if event.is_error else None,
                )
            )
        elif isinstance(event, BashEvent):
            operations.append(
                Operation(
                    index=event.index,
                    entry_id=event.entry_id,
                    tool="bash",
                    arguments={"command": event.command} if event.command else {},
                    outcome="failed" if event.is_error else "succeeded",
                    error=event.error,
                )
            )
    return sorted(operations, key=lambda operation: operation.index)


def is_file_operation(operation: Operation) -> bool:
    """Return whether the operation names a path in a known file tool."""
    return operation.tool in FILE_TOOLS


def is_bash_operation(operation: Operation) -> bool:
    """Return whether the operation is a shell execution."""
    return operation.tool == "bash"


def _common_root(paths: Sequence[str]) -> str:
    """Return the longest common path prefix by segment, with a trailing slash."""
    if not paths:
        return ""
    split = [
        [segment for segment in path.replace("\\", "/").split("/") if segment] for path in paths
    ]
    first = split[0]
    common = 0
    while common < len(first) and all(
        len(parts) > common and parts[common] == first[common] for parts in split[1:]
    ):
        common += 1
    return f"{'/'.join(first[:common])}/" if common else ""


def _strip_root(root: str, path: str) -> str:
    return path.replace(root, "", 1) if root and path.startswith(root) else path


@dataclass(frozen=True, slots=True)
class _AddressedFile:
    """A sampled file plus the entry that addresses the operation naming it."""

    path: str
    entry_id: str


def _project_requests(
    events: Sequence[CompactionEvent],
    protected: frozenset[str],
) -> _Section:
    """Fold older user turns into addressed one-liners.

    Turns already carried verbatim by the dialogue section are excluded: they
    would otherwise consume the budget twice and, worse, read as a historical
    paraphrase of a live instruction.
    """

    def historical() -> Iterable[UserEvent]:
        for event in events:
            if isinstance(event, UserEvent) and event.entry_id not in protected:
                yield event

    sampled = sample_addressed_from(historical(), MAX_REQUESTS)
    lines: list[str] = []
    for position, user in enumerate(sampled.values):
        if sampled.omitted and position == sampled.split_index:
            lines.append(
                omission_line(
                    sampled.omitted,
                    sampled.omitted_first_entry_id,
                    sampled.omitted_last_entry_id,
                    "user scope changes",
                )
            )
        line = truncate(first_line(user.text), MAX_USER_ONELINER_BYTES)
        if line:
            lines.append(f"- {line} [entry {user.entry_id}]")
    return _Section(lines=tuple(lines), omitted=sampled.omitted)


def _project_files(events: Sequence[CompactionEvent]) -> _Section:
    """Fold successful file operations into per-kind address lists.

    Tau tool results carry no typed "created" flag, so the fold never claims a
    file was created: that would require reading result prose, and the engine is
    prose-blind. A successful ``write`` is therefore reported as written, which
    covers both cases without asserting anything unprovable.
    """
    read: dict[str, _AddressedFile] = {}
    modified: dict[str, _AddressedFile] = {}
    written: dict[str, _AddressedFile] = {}

    for operation in collect_operations(events):
        if not is_file_operation(operation) or operation.outcome != "succeeded":
            continue
        path = _path_of(operation.arguments)
        if not path:
            continue
        address = _AddressedFile(path=path, entry_id=operation.entry_id)
        if operation.tool == "write":
            written.setdefault(path, address)
        elif operation.tool == "edit":
            modified.setdefault(path, address)
        else:
            read.setdefault(path, address)

    changed = set(modified) | set(written)
    read_only = [item for path, item in read.items() if path not in changed]

    sampled_written = sample_addressed(written.values(), MAX_FILES_PER_KIND)
    sampled_modified = sample_addressed(modified.values(), MAX_FILES_PER_KIND)
    sampled_read = sample_addressed(read_only, MAX_FILES_PER_KIND)
    all_sampled = [*sampled_written.values, *sampled_modified.values, *sampled_read.values]
    if not all_sampled:
        return _Section(())

    root = _common_root([item.path for item in all_sampled])
    lines: list[str] = []
    omitted = 0
    if root:
        lines.append(f"(under {root})")

    def append_kind(header: str, sampled: AddressedSample[_AddressedFile]) -> None:
        nonlocal omitted
        if not sampled.values and not sampled.omitted:
            return
        lines.append(header)
        omitted += sampled.omitted
        for position, item in enumerate(sampled.values):
            if sampled.omitted and position == sampled.split_index:
                lines.append(
                    "  "
                    + omission_line(
                        sampled.omitted,
                        sampled.omitted_first_entry_id,
                        sampled.omitted_last_entry_id,
                        "file addresses",
                    )
                )
            lines.append(f"  {_strip_root(root, item.path)} [entry {item.entry_id}]")

    append_kind("Written:", sampled_written)
    append_kind("Modified:", sampled_modified)
    append_kind("Read:", sampled_read)
    return _Section(lines=tuple(lines), omitted=omitted)


@dataclass(frozen=True, slots=True)
class _ActivityItem:
    line: str
    entry_id: str


def _project_activity(
    events: Sequence[CompactionEvent],
) -> _Section:
    """Fold non-file tool operations into a sampled activity record.

    File operations live in ``[Files And Changes]`` with their paths, so the
    activity fold covers everything else -- shell executions above all, since
    those are the operations a reader most often needs to re-check. The
    transcript shows only a rolling window, while this fold samples the whole
    cumulative stream, head and tail, so old commands stay addressable.
    """
    items: list[_ActivityItem] = []
    for operation in collect_operations(events):
        if is_file_operation(operation):
            continue
        primary = _primary_of(operation.arguments)
        detail = f" ({truncate(first_line(primary), 72)})" if primary else ""
        items.append(
            _ActivityItem(
                line=f"- {operation.ref}{detail} -> {operation.outcome}",
                entry_id=operation.entry_id,
            )
        )
    sampled = sample_addressed(items, MAX_ACTIVITY)
    lines: list[str] = []
    for position, item in enumerate(sampled.values):
        if sampled.omitted and position == sampled.split_index:
            lines.append(
                omission_line(
                    sampled.omitted,
                    sampled.omitted_first_entry_id,
                    sampled.omitted_last_entry_id,
                    "activity records",
                )
            )
        lines.append(f"{item.line} [entry {item.entry_id}]")
    return _Section(lines=tuple(lines), omitted=sampled.omitted)


@dataclass(frozen=True, slots=True)
class _ErrorItem:
    index: int
    entry_id: str
    description: str
    resolved: bool


def _identity_of(operation: Operation) -> str:
    """Return the typed identity that lets a later success resolve a failure."""
    if is_file_operation(operation):
        path = _path_of(operation.arguments)
        if path:
            return f"file\0{operation.tool}\0{path}"
    if is_bash_operation(operation):
        command = _command_of(operation.arguments)
        if command is not None:
            return f"bash\0{operation.tool}\0{command}"
    return f"generic\0{operation.ref}\0{json.dumps(operation.arguments, sort_keys=True)}"


def project_outstanding(events: Sequence[CompactionEvent]) -> _Section:
    """Fold failed operations into open and resolved error records.

    A failure counts as resolved only when a later success has exactly the same
    typed identity -- same tool and path, or same tool and command. Matching is
    never done on error text, because two different failures can share a
    message and two runs of the same fix can differ.
    """
    operations = collect_operations(events)
    successes = [
        (operation.index, _identity_of(operation))
        for operation in operations
        if operation.outcome == "succeeded"
    ]
    items: list[_ErrorItem] = []
    for operation in operations:
        if operation.outcome == "succeeded":
            continue
        path = _path_of(operation.arguments) if is_file_operation(operation) else None
        command = _command_of(operation.arguments) if is_bash_operation(operation) else None
        if path:
            subject = f"{operation.ref} {path}"
        elif command is not None:
            subject = f"{operation.ref}: {truncate(first_line(command), MAX_LINE_BYTES)}"
        else:
            subject = operation.ref
        signature = (
            f": {truncate(first_line(operation.error), MAX_LINE_BYTES)}"
            if operation.error
            else f": {operation.outcome}"
        )
        identity = _identity_of(operation)
        items.append(
            _ErrorItem(
                index=operation.index,
                entry_id=operation.entry_id,
                description=f"{subject}{signature}",
                resolved=any(
                    index > operation.index and key == identity for index, key in successes
                ),
            )
        )

    if not items:
        return _Section(())
    unresolved = sample_addressed([item for item in items if not item.resolved], MAX_UNRESOLVED)
    resolved = sample_addressed([item for item in items if item.resolved], MAX_RESOLVED)
    lines: list[str] = []

    def append(sampled: AddressedSample[_ErrorItem], noun: str) -> None:
        for position, item in enumerate(sampled.values):
            if sampled.omitted and position == sampled.split_index:
                lines.append(
                    omission_line(
                        sampled.omitted,
                        sampled.omitted_first_entry_id,
                        sampled.omitted_last_entry_id,
                        noun,
                    )
                )
            marker = " [RESOLVED]" if item.resolved else ""
            lines.append(f"- {item.description}{marker} [entry {item.entry_id}]")

    append(unresolved, "open error records")
    append(resolved, "resolved error records")
    return _Section(
        lines=tuple(lines),
        omitted=unresolved.omitted + resolved.omitted,
    )


@dataclass(frozen=True, slots=True)
class _Turn:
    entry_id: str
    context_line: str
    tools: str


def _turn_tool_name(event: CompactionEvent, call_ids: frozenset[str]) -> str | None:
    """Return the tool name one turn's histogram should count for this event.

    A result counts only when the call that requested it is not in the stream,
    so one execution contributes one count instead of two.
    """
    if isinstance(event, ToolCallEvent):
        return event.name
    if isinstance(event, ToolResultEvent):
        return event.tool_name if event.tool_call_id not in call_ids else None
    if isinstance(event, BashEvent):
        return "bash" if event.tool_call_id not in call_ids else None
    return None


def _custom_message_line(event: CustomMessageEvent, max_bytes: int) -> str:
    """Render an extension-injected message by type, visibility, and payload."""
    visibility = "visible" if event.display else "hidden"
    suffix = ""
    if event.details is not None:
        payload = json.dumps(event.details, ensure_ascii=False, sort_keys=True)
        suffix = f" details={truncate(payload, MAX_DETAIL_BYTES)}"
    return (
        f"custom {quoted(event.custom_type, 80)} ({visibility}): "
        f"{quoted(event.text, max_bytes)}{suffix}"
    )


def _project_earlier_turns(events: Sequence[CompactionEvent]) -> _Section:
    """Collapse every turn before the last to one addressed line.

    A turn is one user (or extension-injected) message plus the tool histogram
    that followed it. The histogram is computed from typed call names, so an old
    turn still says what was done there even when its prose is gone.
    """

    def turns() -> Iterable[_Turn]:
        call_ids = frozenset(_calls_by_id(events))
        context: UserEvent | CustomMessageEvent | None = None
        counts: dict[str, int] = {}
        order: list[str] = []

        def completed() -> _Turn | None:
            if context is None:
                return None
            context_line = (
                quoted(first_line(context.text), MAX_EARLIER_USER_BYTES)
                if isinstance(context, UserEvent)
                else _custom_message_line(context, MAX_EARLIER_USER_BYTES)
            )
            return _Turn(
                entry_id=context.entry_id,
                context_line=context_line,
                tools=" ".join(f"{name}:{counts[name]}" for name in order),
            )

        for event in events:
            if isinstance(event, (UserEvent, CustomMessageEvent)):
                turn = completed()
                if turn is not None:
                    yield turn
                context = event
                counts = {}
                order = []
                continue
            if context is None:
                continue
            name = _turn_tool_name(event, call_ids)
            if name is None:
                continue
            if name not in counts:
                order.append(name)
            counts[name] = counts.get(name, 0) + 1

    sampled = sample_addressed_from(turns(), MAX_EARLIER_TURNS)
    lines: list[str] = []
    for position, turn in enumerate(sampled.values):
        if sampled.omitted and position == sampled.split_index:
            lines.append(
                omission_line(
                    sampled.omitted,
                    sampled.omitted_first_entry_id,
                    sampled.omitted_last_entry_id,
                    "earlier turns",
                )
            )
        lines.append(
            f"{turn.context_line}{f' | {turn.tools}' if turn.tools else ''} [entry {turn.entry_id}]"
        )

    # Cumulative re-derivation normally leaves no prior summary in the source.
    # When a caller projects only the active context, an earlier summary is the
    # sole evidence of older history, so it is kept as an address, never as text
    # that could be mistaken for this session's own claims.
    for event in events:
        if isinstance(event, PriorSummaryEvent):
            lines.append(f"- prior {event.source} summary [entry {event.entry_id}]")
    return _Section(lines=tuple(lines), omitted=sampled.omitted)


def _project_status(events: Sequence[CompactionEvent]) -> _Section:
    """Bridge the summarized window into the kept tail.

    These are the last facts of the summarized window: the request in flight,
    the last file change, the last command, and the last assistant line. They
    let the next turn continue without re-reading history.
    """
    last_context: UserEvent | CustomMessageEvent | None = None
    for event in reversed(events):
        if isinstance(event, (UserEvent, CustomMessageEvent)):
            last_context = event
            break

    lines: list[str] = []
    if isinstance(last_context, UserEvent):
        request = truncate(first_line(last_context.text), MAX_STATUS_LINE_BYTES)
        lines.append(f"Last request: {request}")
    elif last_context is not None:
        lines.append(f"Last context: {_custom_message_line(last_context, MAX_STATUS_LINE_BYTES)}")

    last_change: Operation | None = None
    last_command: Operation | None = None
    for operation in collect_operations(events):
        if is_file_operation(operation) and operation.tool in MODIFYING_TOOLS:
            last_change = operation
        elif is_bash_operation(operation):
            last_command = operation
    if last_change is not None:
        path = _path_of(last_change.arguments) or ""
        lines.append(f"Last change: {last_change.tool}{f' {path}' if path else ''}")
    if last_command is not None:
        command = _command_of(last_command.arguments)
        detail = f" {truncate(first_line(command), MAX_STATUS_LINE_BYTES)}" if command else ""
        lines.append(f"Last command: {last_command.ref}{detail} -> {last_command.outcome}")

    for event in reversed(events):
        if isinstance(event, AssistantTextEvent):
            note = truncate(first_line(event.text), MAX_STATUS_LINE_BYTES)
            if note:
                lines.append(f"Last note: {note}")
            break
    return _Section(lines=tuple(lines))


def _summarize_args(name: str, arguments: dict[str, JSONValue]) -> str:
    """Render the identifying argument set for a transcript line."""
    primary = (
        _command_of(arguments) if name == "bash" else _path_of(arguments) or _primary_of(arguments)
    )
    if primary:
        return truncate(first_line(primary), MAX_TRANSCRIPT_CMD_BYTES)
    rendered = [
        f"{key}={truncate(str(value), 40)}"
        for key, value in list(arguments.items())[:2]
        if isinstance(value, (str, int, float, bool)) or value is None
    ]
    return " ".join(rendered)


def _project_transcript(events: Sequence[CompactionEvent]) -> _Section:
    """Render a rolling window of the newest events with ``(#N)`` pointers.

    The pointer is the normalized event index, which is stable for a given
    source stream. A reader can quote ``(#184)`` to point at one event without
    the summary storing that event's content.
    """
    window = events[-TRANSCRIPT_WINDOW:]
    lines: list[str] = []
    omitted = len(events) - len(window)
    if omitted > 0:
        lines.append(
            omission_line(
                omitted,
                events[0].entry_id if events else None,
                events[omitted - 1].entry_id,
                "transcript events",
            )
        )
    for event in window:
        ref = f"(#{event.index})"
        if isinstance(event, UserEvent):
            detail = truncate(first_line(event.text), MAX_TRANSCRIPT_LINE_BYTES)
            lines.append(f"{ref} user: {detail}")
        elif isinstance(event, AssistantTextEvent):
            detail = truncate(first_line(event.text), MAX_TRANSCRIPT_LINE_BYTES)
            lines.append(f"{ref} assistant: {detail}")
        elif isinstance(event, PriorSummaryEvent):
            lines.append(f"{ref} prior {event.source} summary [entry {event.entry_id}]")
        elif isinstance(event, CustomMessageEvent):
            lines.append(f"{ref} {_custom_message_line(event, MAX_TRANSCRIPT_LINE_BYTES)}")
        elif isinstance(event, ToolCallEvent):
            lines.append(f"{ref} {event.name}({_summarize_args(event.name, event.arguments)})")
        elif isinstance(event, ToolResultEvent):
            status = "error" if event.is_error else "ok"
            detail = truncate(first_line(event.text), MAX_TRANSCRIPT_LINE_BYTES)
            lines.append(f"{ref} -> {status}: {detail}")
        elif isinstance(event, BashEvent):
            status = "error" if event.is_error else "ok"
            command = truncate(first_line(event.command), MAX_TRANSCRIPT_CMD_BYTES)
            lines.append(f"{ref} bash({command}) -> {status}")
    return _Section(lines=tuple(lines), omitted=omitted)


def project_with_metadata(
    events: Sequence[CompactionEvent],
    retained_dialogue: Sequence[CompactionEvent] = (),
) -> ProjectionResult:
    """Project every fold over one source stream.

    ``retained_dialogue`` is the kept tail, consulted only to pair a summarized
    request with a reply that stays verbatim in context. Tool history in the
    tail is never folded here.
    """
    exchanges = recent_dialogue(events, retained_dialogue)
    dialogue = project_dialogue(exchanges)
    requests = _project_requests(
        events, frozenset(exchange.user.entry_id for exchange in exchanges)
    )
    files = _project_files(events)
    activity = _project_activity(events)
    outstanding = project_outstanding(events)
    earlier_turns = _project_earlier_turns(events)
    status = _project_status(events)
    transcript = _project_transcript(events)
    return ProjectionResult(
        sections=Sections(
            dialogue=dialogue.lines,
            requests=requests.lines,
            files=files.lines,
            activity=activity.lines,
            outstanding=outstanding.lines,
            earlier_turns=earlier_turns.lines,
            status=status.lines,
            transcript=transcript.lines,
        ),
        omitted=ProjectionOmittedCounts(
            requests=requests.omitted,
            files=files.omitted,
            activity=activity.omitted,
            outstanding=outstanding.omitted,
            earlier_turns=earlier_turns.omitted,
            transcript=transcript.omitted,
            dialogue_bytes=dialogue.omitted_bytes,
        ),
    )


def project(events: Sequence[CompactionEvent]) -> Sections:
    """Project the folds and return only the sections."""
    return project_with_metadata(events).sections
