"""Tests for compaction byte ceilings and conservation-checked sampling."""

from __future__ import annotations

from dataclasses import dataclass

from tau_coding.compaction.bounds import (
    Addressed,
    bounded_excerpt,
    canonicalize_text,
    clip_utf8,
    omission_line,
    sample_addressed,
    sample_addressed_from,
    utf8_bytes,
)


def test_clip_utf8_never_splits_a_multibyte_character() -> None:
    text = "héllo wörld — naïve"
    for limit in range(1, utf8_bytes(text) + 1):
        clipped = clip_utf8(text, limit)
        assert utf8_bytes(clipped) <= limit
        assert clipped.encode("utf-8").decode("utf-8") == clipped


def test_clip_utf8_returns_empty_when_the_suffix_alone_does_not_fit() -> None:
    assert clip_utf8("abc", 1) == ""
    assert clip_utf8("abc", 0) == ""


def test_clip_utf8_keeps_text_that_already_fits() -> None:
    assert clip_utf8("abc", 3) == "abc"
    assert clip_utf8("abc", 100) == "abc"


def test_bounded_excerpt_reports_the_exact_loss_in_bytes() -> None:
    source = "x" * 400
    excerpt = bounded_excerpt(source, 128)
    assert utf8_bytes(excerpt.text) <= 128
    assert excerpt.omitted_bytes == 336
    assert excerpt.text == f"{'x' * 64}\n[omitted {excerpt.omitted_bytes} UTF-8 bytes]"


def test_bounded_excerpt_passes_short_text_through_untouched() -> None:
    excerpt = bounded_excerpt("short", 128)
    assert excerpt.text == "short"
    assert excerpt.omitted_bytes == 0


def test_canonicalize_text_collapses_whitespace_and_reports_truncation() -> None:
    canonical = canonicalize_text("  a\n\n b\t\tc  ", max_bytes=64)
    assert canonical.text == "a b c"
    assert canonical.truncated is False
    assert canonical.source_bytes == 5

    long_text = canonicalize_text("word " * 200, max_bytes=32)
    assert utf8_bytes(long_text.text) <= 32
    assert long_text.truncated is True


@dataclass(frozen=True, slots=True)
class _Item:
    entry_id: str


def _items(count: int) -> list[_Item]:
    return [_Item(entry_id=f"e{index}") for index in range(count)]


def test_sample_addressed_keeps_the_head_and_the_tail() -> None:
    sampled = sample_addressed(_items(10), 6)
    assert [item.entry_id for item in sampled.values] == [
        "e0",
        "e1",
        "e2",
        "e7",
        "e8",
        "e9",
    ]
    assert sampled.omitted == 4
    assert sampled.omitted_first_entry_id == "e3"
    assert sampled.omitted_last_entry_id == "e6"
    assert sampled.split_index == 3


def test_sample_addressed_conserves_items_at_every_ceiling() -> None:
    for max_values in range(0, 12):
        for count in (0, 1, 5, 11, 40):
            items = _items(count)
            sampled = sample_addressed(items, max_values)
            assert len(sampled.values) + sampled.omitted == count
            assert len(sampled.values) <= max_values
            assert sampled.omitted == (0 if count <= max_values else count - max_values)


def test_sample_addressed_from_accepts_a_stream_without_buffering() -> None:
    sampled = sample_addressed_from(iter(_items(9)), 4)
    assert [item.entry_id for item in sampled.values] == ["e0", "e1", "e7", "e8"]
    assert sampled.omitted == 5


def test_sample_addressed_keeps_chronological_order() -> None:
    sampled = sample_addressed(_items(20), 8)
    entry_ids = [item.entry_id for item in sampled.values]
    assert entry_ids == sorted(entry_ids, key=lambda value: int(value[1:]))


def test_omission_line_addresses_the_omitted_source_range() -> None:
    line = omission_line(3, "aaa", "fff", "earlier turns")
    assert line == "… omitted 3 earlier turns; source entries aaa → fff"
    assert omission_line(1, None, None, "items").endswith("(unknown range)")


def test_addressed_protocol_is_structural() -> None:
    class _Named:
        entry_id = "xyz"

    item: Addressed = _Named()
    assert item.entry_id == "xyz"
