"""Tests for autostr.reflow – the core subtitle segmentation logic.

These tests do not require any ML models or ffmpeg; they cover only the
pure-Python reflow and SRT-writing code.
"""
from __future__ import annotations

import pytest

from autostr.reflow import (
    SubtitleEntry,
    _split_text_on_punctuation,
    _wrap_to_two_lines,
    reflow,
)
from autostr.srt_writer import _format_time, write_srt
from autostr.transcribe import TranscriptSegment


# ──────────────────────────────────────────────────────────────────────────────
# _format_time
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected", [
    (0.0,      "00:00:00,000"),
    (1.5,      "00:00:01,500"),
    (61.0,     "00:01:01,000"),
    (3661.123, "01:01:01,123"),
    (3599.999, "00:59:59,999"),
])
def test_format_time(seconds, expected):
    assert _format_time(seconds) == expected


def test_format_time_negative_clamped():
    """Negative seconds should be clamped to 00:00:00,000."""
    assert _format_time(-5.0) == "00:00:00,000"


# ──────────────────────────────────────────────────────────────────────────────
# _split_text_on_punctuation
# ──────────────────────────────────────────────────────────────────────────────

def test_split_on_strong_break():
    text = "你好。再見。"
    chunks = _split_text_on_punctuation(text, max_chars=32)
    assert len(chunks) == 2
    assert chunks[0] == "你好。"
    assert chunks[1] == "再見。"


def test_split_on_weak_break():
    text = "一二三四五六七八九十，一二三四五六七八九十"
    chunks = _split_text_on_punctuation(text, max_chars=12)
    assert all(len(c) <= 12 for c in chunks)


def test_split_no_punctuation_hard_wrap():
    text = "一" * 30
    chunks = _split_text_on_punctuation(text, max_chars=10)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks) == text


def test_split_empty_string():
    assert _split_text_on_punctuation("", max_chars=16) == []


def test_split_short_string_unchanged():
    text = "你好嗎"
    chunks = _split_text_on_punctuation(text, max_chars=16)
    assert chunks == ["你好嗎"]


# ──────────────────────────────────────────────────────────────────────────────
# _wrap_to_two_lines
# ──────────────────────────────────────────────────────────────────────────────

def test_wrap_short_text_single_line():
    lines = _wrap_to_two_lines("你好", max_chars_per_line=16)
    assert lines == ["你好"]


def test_wrap_long_text_two_lines():
    text = "一二三四五六七八九十一二三四五六七八"
    lines = _wrap_to_two_lines(text, max_chars_per_line=10)
    assert len(lines) == 2
    assert all(len(l) <= 10 for l in lines)


def test_wrap_splits_near_punctuation():
    text = "我喜歡吃蘋果，你喜歡吃什麼"
    lines = _wrap_to_two_lines(text, max_chars_per_line=16)
    # Should split on or near the comma
    assert len(lines) <= 2


# ──────────────────────────────────────────────────────────────────────────────
# reflow
# ──────────────────────────────────────────────────────────────────────────────

def _make_segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_reflow_basic():
    segments = [
        _make_segment(0.0, 3.0, "今天天氣很好。"),
        _make_segment(3.0, 6.0, "我們去爬山吧。"),
    ]
    entries = reflow(segments)
    assert len(entries) >= 2
    for entry in entries:
        assert entry.start >= 0.0
        assert entry.end > entry.start
        assert entry.lines


def test_reflow_index_sequential():
    segments = [_make_segment(float(i), float(i + 2), f"句子{i}。") for i in range(5)]
    entries = reflow(segments)
    for i, e in enumerate(entries):
        assert e.index == i + 1


def test_reflow_min_duration_enforced():
    segments = [_make_segment(0.0, 0.1, "短。")]  # only 0.1s
    entries = reflow(segments, min_duration=0.8)
    assert entries[0].end - entries[0].start >= 0.8


def test_reflow_start_delay():
    segments = [_make_segment(1.0, 3.0, "測試字幕。")]
    entries_no_delay = reflow(segments, start_delay_ms=0)
    entries_with_delay = reflow(segments, start_delay_ms=200)
    assert entries_with_delay[0].start > entries_no_delay[0].start


def test_reflow_global_shift():
    segments = [_make_segment(1.0, 3.0, "測試字幕。")]
    entries_no_shift = reflow(segments, global_shift_ms=0)
    entries_shifted = reflow(segments, global_shift_ms=500)
    assert entries_shifted[0].start == pytest.approx(entries_no_shift[0].start + 0.5)
    assert entries_shifted[0].end == pytest.approx(entries_no_shift[0].end + 0.5)


def test_reflow_start_clamped_to_zero():
    segments = [_make_segment(0.1, 1.0, "早。")]
    entries = reflow(segments, global_shift_ms=-500)  # shift backwards
    assert entries[0].start >= 0.0


def test_reflow_skips_empty_segments():
    segments = [
        _make_segment(0.0, 1.0, ""),
        _make_segment(1.0, 2.0, "  "),
        _make_segment(2.0, 3.0, "有效字幕。"),
    ]
    entries = reflow(segments)
    assert len(entries) == 1
    assert entries[0].lines[0] == "有效字幕。"


def test_reflow_line_length_limit():
    long_text = "一" * 40 + "。"
    segments = [_make_segment(0.0, 5.0, long_text)]
    entries = reflow(segments, max_chars_per_line=16)
    for entry in entries:
        for line in entry.lines:
            assert len(line) <= 16


# ──────────────────────────────────────────────────────────────────────────────
# write_srt
# ──────────────────────────────────────────────────────────────────────────────

def test_write_srt(tmp_path):
    entries = [
        SubtitleEntry(index=1, start=1.0, end=3.0, lines=["你好世界"]),
        SubtitleEntry(index=2, start=4.0, end=6.0, lines=["再見", "世界"]),
    ]
    out = tmp_path / "test.srt"
    result = write_srt(entries, out)
    assert result == out
    content = out.read_text(encoding="utf-8")
    assert "1\n" in content
    assert "00:00:01,000 --> 00:00:03,000" in content
    assert "你好世界" in content
    assert "2\n" in content
    assert "再見\n世界" in content


def test_write_srt_empty(tmp_path):
    out = tmp_path / "empty.srt"
    write_srt([], out)
    assert out.read_text() == ""


def test_write_srt_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "dir" / "output.srt"
    entries = [SubtitleEntry(index=1, start=0.0, end=1.0, lines=["測試"])]
    write_srt(entries, out)
    assert out.exists()
