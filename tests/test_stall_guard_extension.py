"""Behavioral tests for the shipped `stall_guard` example extension.

The load-bearing behavior is end to end: a model turn that ends with no tool
call must not settle while a corrective follow-up can still be drained by the
same run. Those tests drive a real `CodingSession` with `FakeProvider`, so the
harness event dispatch, the extension follow-up queue, and the loop's queue
drain are exercised together. Queue awareness, budgets, the transcript
renderer, and the slash command are checked directly against the runtime.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from pi_event_helpers import assistant_done, assistant_start
from tau_agent.events import AgentStartEvent, MessageStartEvent
from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    CustomMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    assistant_content,
)
from tau_agent.session import JsonlSessionStorage
from tau_agent.tools import (
    AgentTool,
    AgentToolResult,
    ToolCancellationToken,
    ToolUpdateCallback,
)
from tau_agent.types import JSONValue
from tau_ai import FakeProvider
from tau_ai.events import AssistantMessageEvent
from tau_coding import CodingSession, CodingSessionConfig, TauResourcePaths
from tau_coding.events import QueueUpdateEvent
from tau_coding.extensions import (
    ExtensionRuntime,
    NullUiBridge,
    TurnEndEvent,
)

pytestmark = pytest.mark.anyio

EXTENSION_PATH = Path(__file__).resolve().parents[1] / "examples" / "extensions" / "stall_guard.py"
CUSTOM_TYPE = "stall-guard:retry"

LEAKED_REPLY = (
    "Time to look at the file.\n\n"
    '<invoke name="read">\n<parameter name="path">notes.md</parameter>\n</invoke>'
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class RecordingUiBridge(NullUiBridge):
    """Keep extension notifications observable in tests."""

    def __init__(self) -> None:
        self.notifications: list[tuple[str, str]] = []

    @property
    def has_ui(self) -> bool:
        return True

    def notify(self, message: str, level: str = "info") -> None:
        self.notifications.append((message, level))


class QueuedSession:
    """Minimal bound-session fake for runtime-level guard decisions."""

    def __init__(self, tmp_path: Path) -> None:
        self.cwd = tmp_path
        self.model = "fake"
        self.provider_name = "fake"
        self.available_providers: tuple[str, ...] = ("fake",)
        self.available_model_choices: tuple[object, ...] = ()
        self.inference_provider: str | None = None
        self.inference_provider_mode = "automatic"
        self.session_id = "session-1"
        self.session_name: str | None = None
        self.thinking_level = "medium"
        self.system_prompt = "You are Tau."
        self.project_trust_resolution: object | None = None
        self.is_running = True
        self.messages: tuple[AgentMessage, ...] = ()
        self.queued: list[tuple[str, str | None]] = []

    def queue_steering_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        self.queued.append((content, custom_type))

    def queue_follow_up_message(
        self,
        content: str,
        *,
        custom_type: str | None = None,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        self.queued.append((content, custom_type))

    async def append_custom_entry(self, namespace: str, data: dict[str, JSONValue]) -> None:
        del namespace, data

    async def set_label(self, target_id: str, label: str | None) -> None:
        del target_id, label

    def set_inference_provider(self, route: str | None) -> str:
        self.inference_provider = route
        return route or "automatic"


def _stream(
    message: AssistantMessage,
    *,
    finish_reason: str | None = None,
) -> list[AssistantMessageEvent]:
    return [assistant_start(model="fake"), assistant_done(message, finish_reason=finish_reason)]


def _stalled_message() -> AssistantMessage:
    return AssistantMessage(model="fake", content=LEAKED_REPLY)


def _answer(text: str) -> AssistantMessage:
    return AssistantMessage(model="fake", content=text)


def _probe_tool() -> AgentTool:
    async def executor(
        tool_call_id: str,
        arguments: Mapping[str, JSONValue],
        signal: ToolCancellationToken | None = None,
        on_update: ToolUpdateCallback | None = None,
    ) -> AgentToolResult:
        del tool_call_id, arguments, signal, on_update
        return AgentToolResult(content=[TextContent(text="probe ran")], details={})

    return AgentTool(
        name="probe",
        label="Probe",
        description="Fake tool used to build a normal tool round.",
        parameters={"type": "object", "properties": {}},
        execute_fn=executor,
    )


def _resource_paths(tmp_path: Path) -> TauResourcePaths:
    paths = TauResourcePaths(
        root=tmp_path / "home-tau",
        cwd=tmp_path / "project",
        agents_root=tmp_path / "home-agents",
    )
    assert paths.cwd is not None
    paths.cwd.mkdir(parents=True, exist_ok=True)
    return paths


def _load_runtime(tmp_path: Path) -> ExtensionRuntime:
    runtime = ExtensionRuntime()
    runtime.load(
        _resource_paths(tmp_path),
        extra_paths=(EXTENSION_PATH,),
        include_resource_dirs=False,
    )
    assert not [diagnostic for diagnostic in runtime.diagnostics if diagnostic.severity == "error"]
    return runtime


def _bound_runtime(tmp_path: Path) -> tuple[ExtensionRuntime, QueuedSession]:
    runtime = _load_runtime(tmp_path)
    session = QueuedSession(tmp_path)
    runtime.bind(session)  # type: ignore[arg-type]
    return runtime, session


async def _load_session(
    tmp_path: Path,
    provider: FakeProvider,
    *,
    tools: list[AgentTool] | None = None,
) -> tuple[CodingSession, RecordingUiBridge]:
    paths = _resource_paths(tmp_path)
    session = await CodingSession.load(
        CodingSessionConfig(
            provider=provider,
            model="fake",
            storage=JsonlSessionStorage(tmp_path / "session.jsonl"),
            cwd=tmp_path / "project",
            resource_paths=paths,
            extension_paths=(EXTENSION_PATH,),
            tools=tools,
        )
    )
    ui = RecordingUiBridge()
    session.extension_runtime.set_ui_bridge(ui)
    return session, ui


def _corrections(messages: tuple[AgentMessage, ...]) -> list[CustomMessage]:
    return [
        message
        for message in messages
        if isinstance(message, CustomMessage) and message.custom_type == CUSTOM_TYPE
    ]


def _turn_end(
    message: AssistantMessage,
    *,
    tool_results: Sequence[ToolResultMessage] = (),
) -> TurnEndEvent:
    return TurnEndEvent(turn_index=0, message=message, tool_results=list(tool_results))


def _attempt(message: CustomMessage) -> int:
    """Read the guard's attempt counter out of a loosely typed details payload."""

    details = message.details
    assert isinstance(details, dict)
    value = details["attempt"]
    assert isinstance(value, int)
    return value


# -- end to end through a real coding session ----------------------------------


async def test_shipped_example_loads_without_diagnostics(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)

    assert runtime.extension_names == ("stall_guard",)


async def test_leaked_markup_turn_forces_one_retry(tmp_path: Path) -> None:
    provider = FakeProvider([_stream(_stalled_message()), _stream(_answer("Read notes.md."))])
    session, ui = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]

    assert len(provider.calls) == 2
    corrections = _corrections(session.messages)
    assert len(corrections) == 1
    assert "[stall-guard]" in corrections[0].text
    assert "tool-call markup written as plain text" in str(corrections[0].details)
    # The corrective message reaches the model as context on the retry request.
    assert any("[stall-guard]" in message.text for message in provider.calls[1][2])
    assert [level for _, level in ui.notifications] == ["warning"]
    assert "forcing one retry (1/2" in ui.notifications[0][0]

    await session.aclose()


async def test_phantom_tool_use_stop_reason_forces_one_retry(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _stream(_answer("Reading the file now."), finish_reason="tool_calls"),
            _stream(_answer("Read it.")),
        ]
    )
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]

    assert len(provider.calls) == 2
    corrections = _corrections(session.messages)
    assert len(corrections) == 1
    assert 'stop_reason "toolUse"' in str(corrections[0].details)

    await session.aclose()


async def test_thinking_only_turn_forces_one_retry(tmp_path: Path) -> None:
    thinking = AssistantMessage(
        model="fake",
        content=[ThinkingContent(thinking="I should read the file next.")],
    )
    provider = FakeProvider([_stream(thinking), _stream(_answer("Read it."))])
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]

    assert len(provider.calls) == 2
    corrections = _corrections(session.messages)
    assert len(corrections) == 1
    assert "thinking only" in str(corrections[0].details)

    await session.aclose()


async def test_plain_prose_answer_is_not_rescued(tmp_path: Path) -> None:
    provider = FakeProvider([_stream(_answer("The answer is 4."))])
    session, ui = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("What is 2 + 2?")]

    assert len(provider.calls) == 1
    assert _corrections(session.messages) == []
    assert ui.notifications == []

    await session.aclose()


async def test_markup_documented_inside_a_real_answer_is_not_rescued(tmp_path: Path) -> None:
    explained = AssistantMessage(
        model="fake",
        content=(
            "Tau calls tools through structured blocks, never prose. You will "
            'never see a line such as <invoke name="read"> in a real answer, '
            "because the harness parses tool calls out of the provider's "
            "tool_calls array instead of out of the message text. A tool call is "
            "emitted by the provider as a typed ToolCall block, executed by the "
            "loop, and returned as a toolResult message the model reads on the "
            "next turn. That is also why a stalled turn has to be rescued at the "
            "turn boundary rather than in the response itself: nothing inside a "
            "well-formed reply signals the mistake, and the only reliable signal "
            "is that the turn ends with no call at all. The retry budget stays "
            "small on purpose, so a model repeating the same mistake cannot keep "
            "spending turns on it, and a genuinely finished answer is left alone."
        ),
    )
    provider = FakeProvider([_stream(explained)])
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Explain how tool calls work")]

    assert len(provider.calls) == 1
    assert _corrections(session.messages) == []

    await session.aclose()


async def test_markup_quoted_in_a_code_block_is_not_rescued(tmp_path: Path) -> None:
    documented = AssistantMessage(
        model="fake",
        content=(
            "The shape looks like this:\n\n```text\n"
            '<invoke name="read">\n<parameter name="path">notes.md</parameter>\n'
            "</invoke>\n```\n\nBut Tau sends that as a structured block, "
            "not as text."
        ),
    )
    provider = FakeProvider([_stream(documented)])
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Show what a call looks like")]

    assert len(provider.calls) == 1
    assert _corrections(session.messages) == []

    await session.aclose()


async def test_normal_tool_round_is_not_rescued(tmp_path: Path) -> None:
    calling = AssistantMessage(
        model="fake",
        content=assistant_content("", [ToolCall(id="call-1", name="probe", arguments={})]),
    )
    provider = FakeProvider([_stream(calling), _stream(_answer("Probe finished."))])
    session, _ = await _load_session(tmp_path, provider, tools=[_probe_tool()])

    _ = [event async for event in session.prompt("Use the probe tool")]

    assert len(provider.calls) == 2
    assert any(isinstance(message, ToolResultMessage) for message in session.messages)
    assert _corrections(session.messages) == []

    await session.aclose()


async def test_retry_budget_is_bounded_per_run(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _stream(_stalled_message()),
            _stream(_stalled_message()),
            _stream(_stalled_message()),
            _stream(_answer("Finally answered.")),
        ]
    )
    session, ui = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]

    # One initial request plus MAX_PER_RUN (2) retries, then the run settles.
    assert len(provider.calls) == 3
    assert len(_corrections(session.messages)) == 2
    assert [message for message, level in ui.notifications if level == "warning"] == [
        "stall-guard: tool-call markup written as plain text "
        "— forcing one retry (1/2 this run, 1/8 this session)",
        "stall-guard: tool-call markup written as plain text "
        "— forcing one retry (2/2 this run, 2/8 this session)",
        "stall-guard: retry budget exhausted, leaving the run settled",
    ]

    await session.aclose()


async def test_second_run_gets_a_fresh_run_budget(tmp_path: Path) -> None:
    provider = FakeProvider(
        [
            _stream(_stalled_message()),
            _stream(_answer("First answer.")),
            _stream(_stalled_message()),
            _stream(_answer("Second answer.")),
        ]
    )
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]
    _ = [event async for event in session.prompt("Read it again")]

    assert len(provider.calls) == 4
    corrections = _corrections(session.messages)
    assert len(corrections) == 2
    attempts = [_attempt(item) for item in corrections]
    assert attempts == [1, 1]

    await session.aclose()


async def test_status_command_reports_and_resets_the_budget(tmp_path: Path) -> None:
    provider = FakeProvider(
        [_stream(_stalled_message()), _stream(_stalled_message()), _stream(_stalled_message())]
    )
    session, _ = await _load_session(tmp_path, provider)

    _ = [event async for event in session.prompt("Read notes.md")]

    spent = session.handle_command("/stall-guard")
    assert spent.handled is True
    spent_message = spent.message
    assert spent_message is not None
    assert "run 2/2" in spent_message
    assert "session 2/8" in spent_message

    reset = session.handle_command("/stall-guard reset")
    assert reset.handled is True
    reset_message = reset.message
    assert reset_message is not None
    assert "run 0/2" in reset_message
    assert "session 0/8" in reset_message

    await session.aclose()


# -- runtime-level guard decisions --------------------------------------------


async def test_queued_follow_up_suppresses_the_rescue(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    await runtime.emit_event(QueueUpdateEvent(steering=(), follow_up=("next task",)))
    await runtime.emit_event(_turn_end(_stalled_message()))

    assert session.queued == []


async def test_injected_correction_reenables_the_guard(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    await runtime.emit_event(_turn_end(_stalled_message()))
    # A queued message that the run then drains clears the pending count.
    await runtime.emit_event(QueueUpdateEvent(steering=(), follow_up=("correction",)))
    await runtime.emit_event(
        MessageStartEvent(message=CustomMessage(custom_type=CUSTOM_TYPE, content="correction"))
    )
    await runtime.emit_event(_turn_end(_stalled_message()))

    assert [custom_type for _, custom_type in session.queued] == [CUSTOM_TYPE, CUSTOM_TYPE]


async def test_error_and_aborted_turns_are_left_to_tau_retries(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    for stop_reason in ("error", "aborted"):
        await runtime.emit_event(
            _turn_end(
                AssistantMessage(
                    model="fake",
                    content=LEAKED_REPLY,
                    stop_reason=stop_reason,
                    error_message="provider said no",
                )
            )
        )

    assert session.queued == []


async def test_turn_with_tool_results_is_not_rescued(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    await runtime.emit_event(
        _turn_end(
            _answer("calling tools"),
            tool_results=[ToolResultMessage(tool_call_id="call-1", tool_name="read", content="ok")],
        )
    )

    assert session.queued == []


async def test_agent_start_resets_the_run_budget(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    for _ in range(4):
        await runtime.emit_event(_turn_end(_stalled_message()))
    assert len(session.queued) == 2

    await runtime.emit_event(AgentStartEvent())
    await runtime.emit_event(_turn_end(_stalled_message()))

    assert len(session.queued) == 3


async def test_session_budget_caps_retries_across_runs(tmp_path: Path) -> None:
    runtime, session = _bound_runtime(tmp_path)

    for _ in range(5):
        for _ in range(3):
            await runtime.emit_event(_turn_end(_stalled_message()))
        await runtime.emit_event(AgentStartEvent())

    assert len(session.queued) == 8


async def test_renderer_line_is_compact_and_expands_on_request(tmp_path: Path) -> None:
    runtime = _load_runtime(tmp_path)
    details = {"reason": "thinking only", "attempt": 1, "max_per_run": 2}

    compact = runtime.render_custom_message(CUSTOM_TYPE, "correction text", details, False)
    expanded = runtime.render_custom_message(CUSTOM_TYPE, "correction text", details, True)

    assert compact is not None
    assert "stall-guard" in compact
    assert "thinking only" in compact
    assert "retry 1/2" in compact
    assert "correction text" not in compact
    assert expanded is not None
    assert "correction text" in expanded
