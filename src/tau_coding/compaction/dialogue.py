"""Recent user/assistant exchanges kept verbatim inside a compaction summary.

Dialogue is the one section that must not decay into a one-liner: a truncated
instruction reads like a completed instruction, and the model will act on the
truncated part. The dialogue fold therefore keeps whole exchanges (the last few
eligible user turns and the assistant reply that answered each one) and shortens
text only when the section itself exceeds its byte allocation.

Recency is measured in eligible user exchanges, not in tool events. An
assistant entry contributes one response no matter how many text blocks it
contains, and thinking blocks never reach here because normalization erases
them upstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tau_coding.compaction.bounds import bounded_excerpt, utf8_bytes
from tau_coding.compaction.events import AssistantTextEvent, CompactionEvent, UserEvent

#: How many user exchanges survive into the dialogue section.
MAX_RECENT_EXCHANGES = 3

# Three exchanges plus the section heading fit inside the dialogue allocation
# in ``render.SUMMARY_SECTIONS``; growing either without the other turns a
# protected section into a shrink request.
MAX_EXCHANGE_BYTES = 3968

_RETAINED_SUFFIX = "; retained raw"


@dataclass(slots=True)
class DialogueMessage:
    """One side of an exchange, tagged with whether its raw entry is kept.

    ``retained`` marks content that is still present in the retained tail of the
    context. Its text is not re-rendered -- only its address is -- so a
    summarized request and the raw reply that followed it can both appear without
    duplicating content the model already sees.
    """

    entry_id: str
    text: str
    retained: bool


@dataclass(slots=True)
class DialogueExchange:
    """A user turn paired with the assistant response that answered it."""

    user: DialogueMessage
    assistant: DialogueMessage | None = None


def recent_dialogue(
    events: Sequence[CompactionEvent],
    retained_tail: Sequence[CompactionEvent] = (),
) -> list[DialogueExchange]:
    """Return the last ``MAX_RECENT_EXCHANGES`` exchanges across two windows.

    ``events`` is the summarized window and ``retained_tail`` the context kept
    verbatim. An exchange pairs a user turn with the assistant response that
    *followed* it, so the response labelled in the summary is the one that
    answered that instruction. An exchange opened in the summarized window stays
    open into the tail, which is how a summarized request keeps the reply that
    compaction left in context (the reply then appears as an address, not as
    duplicated text).

    An exchange is dropped only when the model still sees both sides verbatim:
    a retained request with a retained reply needs no summary of itself.
    Filtering happens before the cap, so tail replies cannot evict summarized
    context.
    """
    exchanges: list[DialogueExchange] = []
    current: DialogueExchange | None = None

    def close() -> None:
        nonlocal current
        if current is None:
            return
        assistant = current.assistant
        if not current.user.retained or (assistant is not None and not assistant.retained):
            exchanges.append(current)
            while len(exchanges) > MAX_RECENT_EXCHANGES:
                exchanges.pop(0)
        current = None

    def visit(source: Sequence[CompactionEvent], retained: bool) -> None:
        nonlocal current
        for event in source:
            if isinstance(event, UserEvent):
                close()
                current = DialogueExchange(
                    user=DialogueMessage(
                        entry_id=event.entry_id,
                        text=event.text,
                        retained=retained,
                    )
                )
                continue
            if not isinstance(event, AssistantTextEvent) or not event.text.strip():
                continue
            if current is None:
                # Assistant prose with no visible request in either window is a
                # reply to content the fold cannot attribute, so it is skipped
                # rather than attached to an unrelated instruction.
                continue
            if current.assistant is None:
                current.assistant = DialogueMessage(
                    entry_id=event.entry_id,
                    text=event.text,
                    retained=retained,
                )
            elif current.assistant.entry_id == event.entry_id:
                # One entry's text blocks are one response, not two.
                current.assistant.text = f"{current.assistant.text}\n{event.text}"

    visit(events, False)
    visit(retained_tail, True)
    close()
    return exchanges


@dataclass(frozen=True, slots=True)
class DialogueExcerpt:
    """One rendered dialogue item: its labelled heading and bounded text."""

    heading: str
    text: str
    omitted_bytes: int


@dataclass(frozen=True, slots=True)
class _ExchangeBudget:
    """Byte budget left for one exchange after its headings."""

    available: int
    user: int
    assistant: int


def _exchange_budget(exchange: DialogueExchange) -> _ExchangeBudget:
    """Split one exchange's budget between the request and its reply.

    The user's own words get at least half even when the assistant reply is
    long, because the reply is labelled as unverified history while the request
    is an instruction. Short replies leave the remainder available to the
    request.
    """
    user_heading = _heading("User instruction", exchange.user)
    assistant_heading = (
        _heading("Historical assistant response (not a verified outcome)", exchange.assistant)
        if exchange.assistant is not None
        else ""
    )
    # 8 bytes of slack for the separators and newlines the excerpt is joined with.
    available = max(0, MAX_EXCHANGE_BYTES - utf8_bytes(user_heading + assistant_heading) - 8)
    reply_bytes = (
        utf8_bytes(exchange.assistant.text)
        if exchange.assistant is not None and not exchange.assistant.retained
        else 0
    )
    user_budget = (
        available if exchange.assistant is None else max(available // 2, available - reply_bytes)
    )
    user_budget = min(user_budget, available)
    return _ExchangeBudget(
        available=available,
        user=int(user_budget),
        assistant=max(0, available - utf8_bytes(exchange.user.text)),
    )


def _heading(role: str, message: DialogueMessage) -> str:
    suffix = _RETAINED_SUFFIX if message.retained else ""
    return f"{role} [entry {message.entry_id}{suffix}]:"


def dialogue_excerpts(exchange: DialogueExchange) -> list[DialogueExcerpt]:
    """Render one exchange as heading-labelled, byte-bounded excerpts.

    The request comes first, then the response that answered it. Text whose raw
    entry is still in the retained tail is replaced by an empty body: the address
    is what the summary must contribute, and repeating the bytes would spend the
    budget twice on one message.
    """
    budget = _exchange_budget(exchange)
    user_text = "" if exchange.user.retained else exchange.user.text
    user = bounded_excerpt(user_text, budget.user)
    excerpts = [
        DialogueExcerpt(
            heading=_heading("User instruction", exchange.user),
            text=user.text,
            omitted_bytes=user.omitted_bytes,
        )
    ]
    if exchange.assistant is not None:
        reply_text = "" if exchange.assistant.retained else exchange.assistant.text
        reply = bounded_excerpt(reply_text, budget.assistant)
        excerpts.append(
            DialogueExcerpt(
                heading=_heading(
                    "Historical assistant response (not a verified outcome)",
                    exchange.assistant,
                ),
                text=reply.text,
                omitted_bytes=reply.omitted_bytes,
            )
        )
    return excerpts


@dataclass(frozen=True, slots=True)
class DialogueProjection:
    """Dialogue section lines plus the exact bytes clipped away."""

    lines: tuple[str, ...]
    omitted_bytes: int = 0


def project_dialogue(exchanges: Sequence[DialogueExchange]) -> DialogueProjection:
    """Project exchanges into section lines, conserving clipped bytes."""
    omitted_bytes = 0
    lines: list[str] = []
    for exchange in exchanges:
        items: list[str] = []
        for excerpt in dialogue_excerpts(exchange):
            omitted_bytes += excerpt.omitted_bytes
            items.append(f"{excerpt.heading}\n{excerpt.text}")
        lines.append("\n\n".join(items))
    return DialogueProjection(lines=tuple(lines), omitted_bytes=omitted_bytes)
