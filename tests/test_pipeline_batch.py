"""Tests for batch subtitle scanning in autostr.pipeline."""
from __future__ import annotations

from autostr.pipeline import find_missing_subtitle_jobs


def test_find_missing_subtitle_jobs_respects_existing_srt(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    nested_input = input_dir / "nested"
    nested_input.mkdir(parents=True)
    output_dir.mkdir()

    existing_video = input_dir / "movie1.mp4"
    missing_video = nested_input / "movie2.mkv"
    ignored_file = input_dir / "notes.txt"

    existing_video.write_bytes(b"fake")
    missing_video.write_bytes(b"fake")
    ignored_file.write_text("ignore me", encoding="utf-8")
    (output_dir / "movie1.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")

    jobs = find_missing_subtitle_jobs(input_dir, output_dir)

    assert jobs == [(missing_video, output_dir / "nested" / "movie2.srt")]