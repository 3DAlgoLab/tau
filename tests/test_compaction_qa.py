"""Tests for ground-truth QA probes over a rendered deterministic summary."""

from __future__ import annotations

from tau_agent.messages import (
    AgentMessage,
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from tau_coding.compaction.events import ContextRow, normalize_rows
from tau_coding.compaction.projections import MAX_FILES_PER_KIND, project_with_metadata
from tau_coding.compaction.qa import check_probes, generate_probes, qa_report
from tau_coding.compaction.render import POINTER_LINE, RenderOptions, render_summary_with_metadata


def _assistant(*text: str, calls: tuple[ToolCall, ...] = ()) -> AssistantMessage:
    content: list[object] = [TextContent(text=body) for body in text]
    content.extend(calls)
    return AssistantMessage(content=content)  # type: ignore[arg-type]


def _call(call_id: str, name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(arguments))  # type: ignore[arg-type]


def _result(
    call_id: str,
    name: str,
    text: str,
    *,
    is_error: bool = False,
) -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=call_id,
        tool_name=name,
        content=[TextContent(text=text)],
        is_error=is_error,
    )


def _session() -> list[ContextRow]:
    return [
        ("e1", UserMessage(content="Port the deterministic compaction engine into Tau.")),
        ("e2", _assistant("Reading the reference sources first.")),
        ("e3", _assistant(calls=(_call("c1", "read", path="compaction/render.ts"),))),
        ("e4", _result("c1", "read", "export const render = ...")),
        ("e5", _assistant(calls=(_call("c2", "bash", command="pytest tests -q"),))),
        ("e6", _result("c2", "bash", "2 failed", is_error=True)),
        ("e7", _assistant(calls=(_call("c3", "bash", command="pytest tests -q"),))),
        ("e8", _result("c3", "bash", "all passed")),
        ("e9", UserMessage(content="Now write the renderer. 한국어 주석도 확인해 주세요.")),
        ("e10", _assistant("Writing render.py now.")),
        ("e11", _assistant(calls=(_call("c4", "write", path="src/pkg/render.py"),))),
        ("e12", _result("c4", "write", "wrote file")),
        ("e13", _assistant(calls=(_call("c5", "edit", path="src/pkg/projections.py"),))),
        ("e14", _result("c5", "edit", "replaced 1 block")),
        ("e15", UserMessage(content="Run the gates.")),
        ("e16", _assistant(calls=(_call("c6", "bash", command="uv run mypy"),))),
        ("e17", _result("c6", "bash", "9 errors found", is_error=True)),
        ("e18", _assistant("Fixing the annotations.")),
        ("e19", UserMessage(content="한국어로 요약도 검토해 주세요.")),
        ("e20", _assistant("요약 검토를 다음 단계로 남깁니다.")),
    ]


def _render(rows: list[ContextRow]) -> tuple[str, int]:
    """Project and render a whole-window compaction, returning text and cut index."""
    events = normalize_rows(rows)
    projection = project_with_metadata(events)
    rendered = render_summary_with_metadata(
        projection.sections,
        RenderOptions(
            first_entry_id=rows[0][0],
            last_entry_id=rows[-1][0],
            last_timestamp="2026-10-28T09:00:00Z",
            request_lines=("Keep the migration order [entry e1]",),
        ),
    )
    return rendered.summary, len(events)


def test_a_full_render_answers_every_probe() -> None:
    summary, cut_index = _render(_session())
    events = normalize_rows(_session())

    report = qa_report(events, cut_index, summary)

    assert report.ok, [failure.reason for failure in report.failures]
    assert report.probe_count >= 12
    classes = {probe.probe_class for probe in generate_probes(events, cut_index)}
    assert classes == {"content", "address"}


def test_probes_are_derived_from_events_not_from_the_summary() -> None:
    rows = _session()
    summary, cut_index = _render(rows)
    events = normalize_rows(rows)

    first = generate_probes(events, cut_index)
    second = generate_probes(events, cut_index)
    assert first == second

    answers = [probe.answer for probe in first]
    assert any("projections.py" in answer for answer in answers)
    assert any("uv run mypy" in answer for answer in answers)

    stripped = summary.replace("projections.py", "something-else.py")
    report = qa_report(events, cut_index, stripped)
    assert report.content_score < 1.0
    assert any("projections.py" in failure.reason for failure in report.failures)


def test_an_unresolved_error_must_keep_its_signature_but_a_resolved_one_need_not() -> None:
    rows = _session()
    summary, cut_index = _render(rows)
    events = normalize_rows(rows)

    probes = {probe.id: probe for probe in generate_probes(events, cut_index)}
    unresolved = [probe for probe in probes.values() if probe.id.startswith("unresolved-error:")]
    assert [probe.answer for probe in unresolved] == ["bash: uv run mypy: 9 errors found"]

    # The earlier failure was answered by a later success with the same identity.
    assert "bash: pytest tests -q: 2 failed" in summary
    assert all(probe.answer != "bash: pytest tests -q: 2 failed" for probe in unresolved)


def test_omitted_files_stay_addressed_by_a_marker_the_summary_must_keep() -> None:
    rows: list[ContextRow] = [("e0", UserMessage(content="Touch many files."))]
    for index in range(MAX_FILES_PER_KIND + 6):
        call_id = f"c{index}"
        rows.append(
            (
                f"w{index}",
                _assistant(calls=(_call(call_id, "write", path=f"src/mod{index}.py"),)),
            )
        )
        rows.append((f"r{index}", _result(call_id, "write", "wrote file")))

    events = normalize_rows(rows)
    projection = project_with_metadata(events)
    rendered = render_summary_with_metadata(
        projection.sections,
        RenderOptions(first_entry_id="e0", last_entry_id=rows[-1][0]),
    )

    report = qa_report(events, len(events), rendered.summary)
    assert report.ok, [failure.reason for failure in report.failures]

    address_probes = [
        probe for probe in generate_probes(events, len(events)) if probe.id.startswith("modified")
    ]
    assert any(probe.answer.startswith("… omitted 6 file addresses") for probe in address_probes)


def test_cjk_facts_are_probed_and_survive_the_render() -> None:
    rows: list[ContextRow] = [
        ("e1", UserMessage(content="요약 알고리즘을 포팅해 주세요.")),
        ("e2", _assistant(calls=(_call("c1", "bash", command="pytest tests -q"),))),
        ("e3", _result("c1", "bash", "테스트 실패", is_error=True)),
        ("e4", UserMessage(content="한국어 오류도 요약에 남기세요.")),
    ]

    events = normalize_rows(rows)
    projection = project_with_metadata(events)
    rendered = render_summary_with_metadata(
        projection.sections,
        RenderOptions(first_entry_id="e1", last_entry_id="e4"),
    )

    report = qa_report(events, len(events), rendered.summary)
    assert report.ok, [failure.reason for failure in report.failures]
    answers = [probe.answer for probe in generate_probes(events, len(events))]
    assert any("한국어 오류도 요약에 남기세요" in answer for answer in answers)
    assert any("테스트 실패" in answer for answer in answers)


def test_losing_only_the_pointer_is_scored_as_an_address_failure() -> None:
    rows = _session()
    summary, cut_index = _render(rows)
    events = normalize_rows(rows)

    without_footer = summary.replace(POINTER_LINE, "")
    report = qa_report(events, cut_index, without_footer)

    assert report.address_score < 1.0
    assert report.content_score == 1.0
    assert report.score < 1.0
    assert not report.ok


def test_probes_can_be_checked_without_a_report() -> None:
    rows = _session()
    summary, cut_index = _render(rows)
    events = normalize_rows(rows)

    probes = generate_probes(events, cut_index)
    passed, failed = check_probes(summary, probes)

    assert len(passed) == len(probes)
    assert failed == []


def test_an_empty_window_produces_no_probes() -> None:
    events = normalize_rows(_session())

    assert generate_probes(events, 0) == []
    report = qa_report(events, None, "")
    assert report.probe_count == 0
    assert report.ok
    assert report.score == 1.0


def test_transcript_pointers_address_events_the_summary_did_not_carry() -> None:
    rows: list[ContextRow] = [
        (f"u{index}", UserMessage(content=f"request number {index}")) for index in range(60)
    ]

    events = normalize_rows(rows)
    projection = project_with_metadata(events)
    rendered = render_summary_with_metadata(
        projection.sections,
        RenderOptions(first_entry_id="u0", last_entry_id=rows[-1][0]),
    )

    report = qa_report(events, len(events), rendered.summary)
    assert report.ok, [failure.reason for failure in report.failures]

    answers = [probe.answer for probe in generate_probes(events, len(events))]
    assert "(#60)" in answers
    assert any(answer.startswith("… omitted 20 transcript events") for answer in answers)


def test_probe_answers_are_never_empty_strings() -> None:
    events = normalize_rows(_session())

    for probe in generate_probes(events, len(events)):
        assert probe.answer, probe.id
        passed, failed = check_probes("", [probe])
        assert passed == [] and failed, probe.id


def test_messages_are_never_rewritten_by_the_engine_path() -> None:
    rows = _session()
    before: list[AgentMessage] = [message for _, message in rows]

    _render(rows)

    assert [message for _, message in rows] == before
