"""Tests for the CLI entrypoint (main.py)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import build_parser, main


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["input.mp4"])
    assert args.video == "input.mp4"
    assert args.batch is False
    assert args.input_dir is None
    assert args.output_dir is None
    assert args.model_dir is None
    assert args.model_size == "medium"
    assert args.language == "zh"
    assert args.device == "cpu"
    assert args.compute_type == "int8"
    assert args.use_whisperx is True
    assert args.max_chars_per_line == 16
    assert args.start_delay_ms == 0
    assert args.global_shift_ms == 0
    assert args.min_duration == 0.8
    assert args.max_duration == 7.0
    assert args.keep_audio is False
    assert args.verbose is False


def test_parser_no_whisperx_flag():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--no-whisperx"])
    assert args.use_whisperx is False


def test_parser_start_delay():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--start-delay", "200"])
    assert args.start_delay_ms == 200


def test_parser_global_shift_negative():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--global-shift", "-300"])
    assert args.global_shift_ms == -300


def test_parser_batch_mode():
    parser = build_parser()
    args = parser.parse_args(["--batch", "--input-dir", "input", "--output-dir", "output"])
    assert args.batch is True
    assert args.video is None
    assert args.input_dir == "input"
    assert args.output_dir == "output"


def test_parser_model_dir():
    parser = build_parser()
    args = parser.parse_args(["video.mp4", "--model-dir", "D:/models"])
    assert args.model_dir == "D:/models"


def test_main_missing_file(capsys):
    ret = main(["nonexistent_file_xyz.mp4"])
    assert ret == 1


def test_main_calls_pipeline(tmp_path):
    """main() should call pipeline.run with the correct arguments."""
    fake_video = tmp_path / "test.mp4"
    fake_video.write_bytes(b"fake")
    fake_srt = tmp_path / "test.srt"

    with patch("autostr.pipeline.run", return_value=fake_srt) as mock_run:
        ret = main([str(fake_video), "--start-delay", "100"])

    assert ret == 0
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["start_delay_ms"] == 100
    assert call_kwargs["model_size"] == "medium"
    assert call_kwargs["model_dir"] is None


def test_main_batch_mode_calls_batch_pipeline(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    with patch("autostr.pipeline.run_missing_subtitles", return_value=[output_dir / "test.srt"]) as mock_run:
        ret = main(["--batch", "--input-dir", str(input_dir), "--output-dir", str(output_dir)])

    assert ret == 0
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    assert call_kwargs["input_dir"] == input_dir
    assert call_kwargs["output_dir"] == output_dir
