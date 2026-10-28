"""Tests for the byte-budgeted renderer that assembles compaction sections."""

from __future__ import annotations

from tau_agent.messages import UserMessage
from tau_coding.compaction.bounds import MAX_SUMMARY_BYTES, utf8_bytes
from tau_coding.compaction.dialogue import MAX_EXCHANGE_BYTES, MAX_RECENT_EXCHANGES
from tau_coding.compaction.projections import Sections
from tau_coding.compaction.render import (
    CARRY_FORWARD_HEADER,
    MAX_DIALOGUE_BYTES,
    MAX_RENDERED_LINE_BYTES,
    POINTER_LINE,
    REQUEST_HEADER,
    RenderOptions,
    format_timestamp,
    message_timestamp,
    render_summary,
    render_summary_with_metadata,
)


def _lines(count: int, size: int, *, label: str = "item") -> tuple[str, ...]:
    return tuple(f"{label} {index} {'x' * (size - 12)}" for index in range(count))


def _history_line_count(summary: str) -> int:
    return sum(1 for line in summary.splitlines() if line.startswith("item "))


def test_dialogue_ceiling_covers_the_exchange_budget() -> None:
    heading_slack = 44
    assert MAX_RECENT_EXCHANGES * (MAX_EXCHANGE_BYTES + heading_slack) <= MAX_DIALOGUE_BYTES


def test_summary_stays_inside_the_global_byte_ceiling() -> None:
    sections = Sections(
        dialogue=_lines(6, 4000, label="dialogue"),
        requests=_lines(400, 300),
        files=_lines(400, 300),
        activity=_lines(400, 300),
        outstanding=_lines(400, 300),
        earlier_turns=_lines(400, 300),
        status=_lines(400, 300),
        transcript=_lines(400, 300),
    )

    rendered = render_summary_with_metadata(
        sections,
        RenderOptions(request_lines=_lines(60, 300, label="request"), carry_lines=_lines(60, 300)),
    )

    assert utf8_bytes(rendered.summary) <= MAX_SUMMARY_BYTES
    assert rendered.request_omitted_bytes > 0
    assert rendered.carry_omitted_bytes > 0
    assert rendered.dialogue_omitted_bytes > 0


def test_footer_carries_the_expansion_pointer_and_source_range() -> None:
    summary = render_summary(
        Sections(status=("Last request: fix the bug",)),
        RenderOptions(
            first_entry_id="a1",
            last_entry_id="z99",
            last_timestamp="2026-10-28T09:00:00Z",
        ),
    )

    assert "[compacted 2026-10-28T09:00:00Z; cumulative source entries a1 → z99]" in summary
    assert POINTER_LINE in summary


def test_branch_summaries_are_labelled_as_structural() -> None:
    summary = render_summary(
        Sections(status=("Last request: continue",)),
        RenderOptions(first_entry_id="a1", last_entry_id="b2", summary_kind="branch"),
    )

    assert "branch summarized" in summary
    assert "structural source entries a1 → b2" in summary


def test_empty_sections_leave_no_header_behind() -> None:
    summary = render_summary(Sections(files=("Written:", "  app.py [entry e3]")))

    assert "[Files And Changes]" in summary
    for header in ("[Historical requests]", "[Tool Activity]", "[Current Status]"):
        assert header not in summary


def test_history_shrinks_by_dropping_lines_not_cutting_one_mid_stream() -> None:
    lines = _lines(500, 200)
    summary = render_summary(Sections(files=lines))

    block = summary.split("[Files And Changes]\n", 1)[1].split("\n\n", 1)[0]
    rendered = block.splitlines()

    assert any(line.startswith("… omitted ") for line in rendered)
    assert len(rendered) < len(lines)
    assert all(utf8_bytes(line) <= MAX_RENDERED_LINE_BYTES for line in rendered)
    assert lines[0] in rendered, "the head of the section must survive"
    assert lines[-1] in rendered, "the tail of the section must survive"


def test_protected_sections_keep_every_item_under_pressure() -> None:
    requests = _lines(40, 400, label="request")
    dialogue = _lines(4, 3000, label="dialogue")

    rendered = render_summary_with_metadata(
        Sections(dialogue=dialogue, requests=requests, files=_lines(200, 300)),
        RenderOptions(request_lines=requests),
    )

    summary = rendered.summary
    assert summary.count(REQUEST_HEADER) == 1
    for index in range(40):
        assert f"request {index}" in summary, f"protected item {index} vanished"
    assert "[Historical requests]" in summary
    assert rendered.request_omitted_bytes > 0


def test_protected_content_is_placed_before_history_in_the_budget() -> None:
    dialogue = tuple(f"dialogue {index}: {'가' * 1500}" for index in range(4))
    history = {
        name: _lines(60, 200)
        for name in (
            "requests",
            "files",
            "activity",
            "outstanding",
            "earlier_turns",
            "status",
            "transcript",
        )
    }
    roomy = render_summary(Sections(**history))
    pressured = render_summary(Sections(dialogue=dialogue, **history))

    assert _history_line_count(roomy) > _history_line_count(pressured) > 0
    assert utf8_bytes(roomy) <= MAX_SUMMARY_BYTES
    assert utf8_bytes(pressured) <= MAX_SUMMARY_BYTES
    for index in range(4):
        assert f"dialogue {index}:" in pressured


def test_carry_forward_block_appears_only_when_it_has_items() -> None:
    without = render_summary(Sections(status=("Last request: go",)))
    with_carry = render_summary(
        Sections(status=("Last request: go",)),
        RenderOptions(carry_lines=("- keep the migration branch [carry:0]",)),
    )

    assert CARRY_FORWARD_HEADER not in without
    assert f"{CARRY_FORWARD_HEADER}\n\n- keep the migration branch [carry:0]" in with_carry


def test_transcript_is_rendered_behind_a_separator_header() -> None:
    summary = render_summary(Sections(transcript=("(#1) user: first", "(#2) read(a.py)")))

    assert summary.startswith("---\n(#1) user: first\n(#2) read(a.py)")


def test_render_is_deterministic() -> None:
    sections = Sections(
        requests=_lines(30, 240),
        files=_lines(30, 240),
        outstanding=_lines(30, 240),
        transcript=_lines(30, 240),
    )
    options = RenderOptions(first_entry_id="e1", last_entry_id="e90")

    assert render_summary(sections, options) == render_summary(sections, options)


def test_section_headers_are_reported_for_provenance() -> None:
    rendered = render_summary_with_metadata(
        Sections(
            dialogue=("User instruction [entry e1]:\ngo",),
            files=("Written:", "  app.py [entry e2]"),
            transcript=("(#1) user: go",),
        ),
        RenderOptions(carry_lines=("- focus [carry:0]",)),
    )

    assert rendered.section_headers == (
        "[Recent user directions and discussion]",
        "[Carry Forward]",
        "[Files And Changes]",
    )


def test_timestamps_render_in_utc_and_unknown_time_is_explicit() -> None:
    assert format_timestamp(0) == "1970-01-01T00:00:00Z"
    assert format_timestamp(None) == "(unknown time)"
    assert message_timestamp(UserMessage(content="hi", timestamp=1_762_000_000_000)) == (
        1_762_000_000_000
    )


def test_unaddressed_source_renders_an_explicit_empty_range() -> None:
    summary = render_summary(Sections(status=("Last request: go",)))

    assert "(no entries)" in summary
