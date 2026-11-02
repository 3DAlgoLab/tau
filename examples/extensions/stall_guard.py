"""Tau extension that recovers model turns that end with no tool call.

The failure it fixes: an OpenAI-compatible endpoint can return finish_reason
``stop`` with the tool-call markup sitting inside ``content`` text instead of
the structured ``tool_calls`` array. Tau then sees an ordinary final answer: no
tool runs, no error is raised, and the run settles silently. Two sibling shapes
settle the same way -- ``stop_reason`` ``toolUse`` with no call block, and a
message carrying only thinking. Tau's built-in retry only covers provider
errors, so none of these are retried.

At the settlement boundary this guard inspects the assistant message that ends
the turn and, for those three shapes, injects one corrective message so the run
makes exactly one more model request -- bounded per run and per session.

A ``length`` stop is deliberately never retried. That stop means the provider
truncated the response at its token budget -- usually because the prompt nearly
fills the context window -- so another request only adds tokens to an already
full prompt. Observed live: a thinking-only turn at 131,064 of 131,072 tokens
was "recovered" this way and the retry was rejected with HTTP 400 ("request
exceeds the context window"). Truncation needs room, and the session owns that:
when a run settles on a truncated turn that produced nothing visible, Tau
compacts and continues it once. This guard's part is to say out loud what the
silent stop meant, never to add tokens to it.

Tau has no Pi-style ``agent_before_settle`` decision hook. The boundary here is
the ``turn_end`` of a turn whose assistant message requested no tool call:
``run_agent_loop`` has emitted that turn but has not yet drained its follow-up
queue, and Tau dispatches agent events to extensions inline from the harness
listener, so a follow-up queued from this handler is consumed by the same run.

Install by copying into `~/.tau/extensions/`, or run:

    tau -e examples/extensions/stall_guard.py
"""

from __future__ import annotations

import re

from tau_agent.messages import AssistantMessage, ThinkingContent
from tau_coding.events import QueueUpdateEvent
from tau_coding.extensions import (
    CustomMessageView,
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    MessageRenderOptions,
    TurnEndEvent,
)

MAX_PER_RUN = 2
MAX_PER_SESSION = 8
CUSTOM_TYPE = "stall-guard:retry"
# A message that both leaked markup and carried a real answer is left alone.
PROSE_OVERRIDE_CHARS = 600
# Provider tool-call grammars that must never appear as prose.
LEAKED_MARKUP = re.compile(
    r"<\s*(?:tool_call|tool_use|function=|parameter=|invoke\b|antml:|/?think\b|im_start|im_end)"
    r"|<\|[^|\n]{0,16}\|",
    re.IGNORECASE,
)
_FENCED_CODE = re.compile(r"```[\s\S]*?```")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_TAGS = re.compile(r"<[^>]*>")
_WHITESPACE = re.compile(r"\s+")

REASON_MARKUP = "tool-call markup written as plain text"
REASON_PHANTOM = 'stop_reason "toolUse" with no tool call block'
REASON_THINKING_ONLY = "thinking only, with no answer and no tool call"


def _strip_code(text: str) -> str:
    """Drop fenced blocks and inline spans: quoted markup is documentation.

    A message that quotes a call grammar inside code is explaining the tool
    interface, not leaking one into prose.
    """
    return _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", text))


def _prose_length(text: str) -> int:
    """Return the length of the message's real prose, ignoring tags."""
    return len(_WHITESPACE.sub(" ", _TAGS.sub(" ", text)).strip())


def _stall_reason(message: AssistantMessage) -> str | None:
    """Return why this settled assistant turn looks like a stalled call.

    ``error``/``aborted``/``length`` are excluded up front: none of them is a
    stall, and a correction cannot fix a provider-side truncation. Retrying a
    full context is what turns a silent truncation into a 400 context overflow.
    """
    if message.stop_reason in {"error", "aborted", "length"}:
        return None
    text = message.text
    leaked = LEAKED_MARKUP.search(_strip_code(text)) is not None and (
        _prose_length(text) <= PROSE_OVERRIDE_CHARS
    )
    phantom = message.stop_reason == "toolUse"
    thinking_only = not text.strip() and any(
        isinstance(block, ThinkingContent) for block in message.content
    )
    if leaked:
        return REASON_MARKUP
    if phantom:
        return REASON_PHANTOM
    if thinking_only:
        return REASON_THINKING_ONLY
    return None


def _correction(reason: str) -> str:
    """Return the model-facing message that asks for a real tool call."""
    return (
        f"[stall-guard] Your previous response ended as {reason}, so nothing was "
        "executed and the run was about to stop. Re-issue the call as a real tool "
        "call through the tool interface. Never write tool-call markup into prose "
        "text, and never name a tool that is absent from your tool list. If you "
        "genuinely meant to answer in prose rather than call a tool, answer now and "
        "say why."
    )


def _render_retry(view: CustomMessageView, options: MessageRenderOptions) -> str:
    """Render the injected correction as one compact transcript line."""
    details = view.details if isinstance(view.details, dict) else {}
    reason = str(details.get("reason", REASON_MARKUP))
    attempt = details.get("attempt", "?")
    limit = details.get("max_per_run", MAX_PER_RUN)
    line = f"[yellow]\u27f2 stall-guard[/yellow] [dim]{reason} \u2014 retry {attempt}/{limit}[/dim]"
    if options.expanded:
        content = view.content if isinstance(view.content, str) else str(view.content)
        line = f"{line}\n[dim]{content}[/dim]"
    return line


class _Budget:
    """Retry budget and queue awareness for one extension load generation."""

    def __init__(self) -> None:
        self.per_run = 0
        self.per_session = 0
        self.pending_queued = 0
        self.exhausted_notified = False

    def reset_run(self) -> None:
        """Start a fresh per-run budget."""
        self.per_run = 0
        self.pending_queued = 0

    def spend(self) -> bool:
        """Consume one retry, or report that the budget is already spent."""
        if self.per_run >= MAX_PER_RUN or self.per_session >= MAX_PER_SESSION:
            return False
        self.per_run += 1
        self.per_session += 1
        return True


def setup(tau: ExtensionAPI) -> None:
    """Subscribe the settlement guard, its renderer, and its status command."""
    budget = _Budget()

    @tau.on("agent_start")
    def on_agent_start(event: object, context: ExtensionContext) -> None:
        del event, context
        budget.reset_run()

    @tau.on("queue_update")
    def on_queue_update(event: QueueUpdateEvent, context: ExtensionContext) -> None:
        del context
        budget.pending_queued = len(event.steering) + len(event.follow_up)

    @tau.on("message_start")
    def on_message_start(event: object, context: ExtensionContext) -> None:
        """Count a queued message as drained: it already resumes the run."""
        del context
        message = getattr(event, "message", None)
        if getattr(message, "role", None) in ("user", "custom"):
            budget.pending_queued = 0

    @tau.on("turn_end")
    def on_turn_end(event: TurnEndEvent, context: ExtensionContext) -> None:
        message = event.message
        if not isinstance(message, AssistantMessage) or event.tool_results:
            return
        if message.tool_calls:
            return
        # A queued message already resumes the run on its own, so there is
        # nothing to rescue; Tau emits no event when the loop drains it.
        if budget.pending_queued > 0:
            return
        if message.stop_reason == "length":
            # Truncation is not a stall and no retry can fix it: the provider
            # cut the response at its token budget, usually because the prompt
            # nearly fills the context window. The session compacts and continues
            # a truncated turn that showed nothing, so explain the silent stop
            # instead of adding tokens to it.
            context.ui.notify(
                "stall-guard: response truncated at the token limit; "
                "not retrying (Tau will compact context instead)",
                "warning",
            )
            return

        # `_stall_reason` also refuses `length`; the remaining stop-reason
        # policy (error/aborted) lives there so classifier and tests agree.
        reason = _stall_reason(message)
        if reason is None:
            return

        if not budget.spend():
            if not budget.exhausted_notified:
                budget.exhausted_notified = True
                context.ui.notify(
                    "stall-guard: retry budget exhausted, leaving the run settled",
                    "warning",
                )
            return

        context.ui.notify(
            f"stall-guard: {reason} \u2014 forcing one retry "
            f"({budget.per_run}/{MAX_PER_RUN} this run, "
            f"{budget.per_session}/{MAX_PER_SESSION} this session)",
            "warning",
        )
        tau.send_custom_message(
            _correction(reason),
            custom_type=CUSTOM_TYPE,
            details={
                "reason": reason,
                "attempt": budget.per_run,
                "max_per_run": MAX_PER_RUN,
                "session_attempt": budget.per_session,
                "max_per_session": MAX_PER_SESSION,
            },
            deliver_as="follow_up",
        )

    def handle_status(args: str, context: ExtensionCommandContext) -> str:
        """Report, or reset, the retry budget."""
        del context
        if args.strip() == "reset":
            budget.reset_run()
            budget.per_session = 0
            budget.exhausted_notified = False
        return (
            f"stall-guard: run {budget.per_run}/{MAX_PER_RUN}, "
            f"session {budget.per_session}/{MAX_PER_SESSION}"
        )

    tau.register_message_renderer(CUSTOM_TYPE, _render_retry)
    tau.register_command(
        "stall-guard",
        handle_status,
        description="Show or reset the stall-guard retry budget",
        usage="[reset]",
    )
