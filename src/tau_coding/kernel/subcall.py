"""Answer kernel sub-calls with provider requests, using the session's own model.

A code cell can ask for a completion through ``llm_query`` /
``llm_query_batched``. Those bindings stop at the kernel boundary; this module is
the host side that turns one prompt into one provider request. It is deliberately
free of session types: the session hands in thunks for the provider and the model,
so a sub-call always uses the *current* model rather than the one selected when
the kernel started.

Two properties are load-bearing:

* Sub-calls request **no tools**. A sub-call is a leaf, so recursion is
  structurally impossible until an explicit ``rlm_query`` arrives with a depth
  guard. Without this, one cell could spawn a full agent that spawns a kernel
  that spawns an agent, with nothing counting the spend.
* Every sub-call is **tallied**. Sub-calls are invisible to session usage
  analytics because they are not session messages; the ledger here is what keeps
  their cost from vanishing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tau_agent.messages import Usage, UserMessage, sum_usage
from tau_agent.provider import CancellationToken, ModelProvider
from tau_agent.provider_events import AssistantDoneEvent, AssistantErrorEvent, TextDeltaEvent
from tau_agent.types import JSONValue
from tau_coding.kernel.protocol import SubcallOutcome, bounded_text

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

__all__ = [
    "DEFAULT_MAX_SUBCALLS_PER_CELL",
    "DEFAULT_SUBCALL_MAX_RESPONSE_BYTES",
    "SUBCALL_SYSTEM_PROMPT",
    "SubcallBridge",
    "SubcallTotals",
]

DEFAULT_SUBCALL_MAX_RESPONSE_BYTES = 32 * 1024

DEFAULT_MAX_SUBCALLS_PER_CELL = 32
"""Ceiling on provider requests one cell may ask for.

Concurrency is bounded in the kernel; this bounds spend. A cell that wants more
than this should have asked fewer, larger questions, and a cell that asks in a
loop should hit a wall before it hits a provider bill.
"""

SUBCALL_SYSTEM_PROMPT = (
    "You are a sub-model called from inside a program. Your answer is parsed by "
    "code, not read by a human: return exactly what was asked for, with no "
    "preamble, no hedging, and no offer to help further. If the request asks for "
    "structured data, return only that data. If you cannot answer, say so in one "
    "short sentence instead of guessing."
)


@dataclass(frozen=True, slots=True)
class SubcallTotals:
    """What the sub-call ledger accumulated over a window."""

    calls: int = 0
    failures: int = 0
    models: tuple[str, ...] = ()
    usage: Usage = field(default_factory=Usage)

    @property
    def requested_tokens(self) -> int:
        return self.usage.input + self.usage.cache_read + self.usage.cache_write

    @property
    def output_tokens(self) -> int:
        return self.usage.output

    @property
    def cost(self) -> float:
        return self.usage.cost.total

    def to_json(self) -> dict[str, JSONValue]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "models": list(self.models),
            "input_tokens": self.requested_tokens,
            "cached_tokens": self.usage.cache_read,
            "output_tokens": self.usage.output,
            "reasoning_tokens": self.usage.reasoning,
            "estimated_cost": round(self.cost, 6),
        }


class SubcallBridge:
    """Turn one prompt from inside a kernel into one provider completion.

    The provider and the model are resolved per call so the bridge can outlive a
    ``/model`` change. ``begin_cell`` opens the window a cell's sub-calls belong
    to: it hands in the tool's cancellation token, so a cell the user stops stops
    paying for in-flight sub-calls rather than merely stopping to read their
    answers, and it enforces ``max_calls_per_cell`` -- a cell is free to fan out,
    but a bug in model-written code must not be able to spend without a ceiling.
    """

    def __init__(
        self,
        *,
        provider: Callable[[], ModelProvider],
        model: Callable[[], str],
        system: str = SUBCALL_SYSTEM_PROMPT,
        max_response_bytes: int = DEFAULT_SUBCALL_MAX_RESPONSE_BYTES,
        max_calls_per_cell: int | None = DEFAULT_MAX_SUBCALLS_PER_CELL,
    ) -> None:
        self._provider = provider
        self._model = model
        self._system = system
        self.max_response_bytes = max_response_bytes
        self.max_calls_per_cell = max_calls_per_cell
        self._signal: CancellationToken | None = None
        self._calls = 0
        self._failures = 0
        self._models: list[str] = []
        self._usages: list[Usage] = []

    # -- per-cell scope -----------------------------------------------------

    def begin_cell(self, signal: CancellationToken | None) -> None:
        """Open the window a cell's sub-calls belong to, starting the tally fresh."""
        self._signal = signal
        self._calls = 0
        self._failures = 0
        self._models = []
        self._usages = []

    def end_cell(self) -> None:
        """Leave the window; sub-calls outside a cell are never cancelled."""
        self._signal = None

    def take_totals(self) -> SubcallTotals:
        """Return the current window's totals and reset the ledger."""
        totals = self.totals()
        self._calls = 0
        self._failures = 0
        self._models = []
        self._usages = []
        return totals

    def totals(self) -> SubcallTotals:
        """Return the totals for the window that is currently open."""
        return SubcallTotals(
            calls=self._calls,
            failures=self._failures,
            models=tuple(self._models),
            usage=sum_usage(self._usages) if self._usages else Usage(),
        )

    # -- the sub-call itself ------------------------------------------------

    async def __call__(self, prompt: str, model: str | None = None) -> SubcallOutcome:
        """Stream one completion for ``prompt``.

        Raises when the provider reports an error or the cell is cancelled: the
        kernel turns a single raised sub-call into a cell error, and a raised
        entry inside a batch into a ``None`` hole. A failed attempt still cost the
        request that failed, so it counts against ``max_calls_per_cell`` too.
        """
        target = model or self._model()
        limit = self.max_calls_per_cell
        if limit is not None and self._calls >= limit:
            raise RuntimeError(
                f"this cell used its sub-call budget of {limit}; partition the work "
                "into fewer, larger prompts or run the next batch in a new cell"
            )
        signal = self._signal
        text_parts: list[str] = []
        final_text: str | None = None
        usages: list[Usage] = []
        response_provider: str | None = None

        stream: AsyncIterator[object] = self._provider().stream_response(
            model=target,
            system=self._system,
            messages=[UserMessage(content=prompt)],
            tools=[],
            signal=signal,
        )
        async for event in stream:
            if isinstance(event, TextDeltaEvent):
                text_parts.append(event.delta)
            elif isinstance(event, AssistantDoneEvent):
                final_text = event.message.text
                usages.append(event.message.usage)
                response_provider = event.message.response_provider
            elif isinstance(event, AssistantErrorEvent):
                self._calls += 1
                self._failures += 1
                self._note_model(target)
                reason = event.error.error_message or event.reason
                raise RuntimeError(f"sub-call to {target} failed: {reason}")

        if signal is not None and signal.is_cancelled():
            self._calls += 1
            self._failures += 1
            raise RuntimeError(f"sub-call to {target} was cancelled")

        self._calls += 1
        self._note_model(target)
        if usages:
            self._usages.extend(usages)

        text = final_text if final_text is not None else "".join(text_parts)
        text, omitted = bounded_text(text, self.max_response_bytes)
        metadata: dict[str, JSONValue] = {
            "model": target,
            "empty": not text.strip(),
        }
        if omitted:
            metadata["omitted_bytes"] = omitted
        if response_provider:
            metadata["response_provider"] = response_provider
        return SubcallOutcome(text=text, model=target, metadata=metadata)

    def _note_model(self, model: str) -> None:
        if model not in self._models:
            self._models.append(model)
