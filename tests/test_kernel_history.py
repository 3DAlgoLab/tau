"""Deterministic history tests: bounded size, and no loss that is not counted.

The history block is what a child sees of its own earlier turns. It is rendered
without a model, so every property below is checkable exactly: no judge, no
sampling, no flake.
"""

from __future__ import annotations

from tau_coding.compaction.bounds import utf8_bytes
from tau_coding.kernel.history import (
    DEFAULT_MAX_HISTORY_BYTES,
    collapse,
    render_history,
)


def collapse_lines(count: int, width: int = 40) -> str:
    return "\n".join(f"row-{i}-{'v' * width}" for i in range(count))


def test_a_small_observation_is_passed_through_untouched() -> None:
    text, omitted = collapse("first line\nlast line", 512)
    assert text == "first line\nlast line"
    assert omitted == 0


def test_a_big_observation_keeps_its_first_and_last_line() -> None:
    body = "HEADER LINE\n" + collapse_lines(2000) + "\nTHE ROW THAT MATTERS"
    text, omitted = collapse(body, 512)

    assert text.startswith("HEADER LINE")
    assert text.endswith("THE ROW THAT MATTERS")
    assert "middle omitted" in text
    assert omitted == utf8_bytes(body) - utf8_bytes(text.splitlines()[0]) - utf8_bytes(
        text.splitlines()[-1]
    )


def test_the_collapsed_form_states_what_it_dropped() -> None:
    body = collapse_lines(4000)
    text, omitted = collapse(body, 400)

    assert f"{omitted:,}" in text
    assert "3998 lines" in text, "the count of collapsed lines is part of the claim"


def test_history_stays_inside_its_budget_however_many_turns() -> None:
    observations = [collapse_lines(900) for _ in range(9)]
    block = render_history(observations, budget_bytes=2048)

    assert block.bytes <= 2048
    assert utf8_bytes("\n".join(observations)) > 200_000
    assert block.collapsed + block.dropped >= 8


def test_the_newest_turn_is_the_one_that_survives_whole() -> None:
    observations = [collapse_lines(700), "the answer is 42"]
    block = render_history(observations, budget_bytes=1024, verbatim=1)

    assert "the answer is 42" in block.text
    assert "--- cell 1 ---" in block.text and "middle omitted" in block.text


def test_dropping_an_older_turn_is_said_out_loud() -> None:
    observations = [collapse_lines(5000) for _ in range(6)]
    block = render_history(observations, budget_bytes=900, collapsed_bytes=300)

    assert block.dropped >= 1
    assert "dropped from this history" in block.text
    source = sum(utf8_bytes(text) for text in observations)
    assert block.omitted_bytes >= source - utf8_bytes(block.text), "nothing may vanish uncounted"


def test_the_same_turns_render_the_same_way() -> None:
    observations = [collapse_lines(1200), "x", collapse_lines(80)]
    first = render_history(observations, budget_bytes=1500)
    second = render_history(observations, budget_bytes=1500)
    assert first.text == second.text
    assert first.omitted_bytes == second.omitted_bytes


def test_no_observations_still_renders_something_the_model_can_read() -> None:
    block = render_history([])
    assert block.text == "(nothing ran)"
    assert block.bytes == 0


def test_the_default_budget_is_larger_than_one_collapsed_turn() -> None:
    assert DEFAULT_MAX_HISTORY_BYTES > 4 * 512
