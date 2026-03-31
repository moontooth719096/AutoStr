"""Tests for automatic highlight detection and clip export."""
from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import patch

import autostr.highlight as highlight_module
from autostr.highlight import detect_highlights, export_highlight_clips
from autostr.transcribe import TranscriptSegment


def _make_segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_detect_highlights_returns_non_overlapping_candidates():
    segments = [
        _make_segment(0.0, 4.0, "開場介紹一下今天的主題。"),
        _make_segment(4.2, 9.0, "這一段先講結論，再補充細節。"),
        _make_segment(20.0, 26.0, "現在進入最重要的部分。"),
        _make_segment(26.3, 33.0, "這裡有明顯的轉折點。"),
        _make_segment(50.0, 58.0, "最後總結今天的重點。"),
    ]

    candidates = detect_highlights(
        segments,
        target_count=2,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
    )

    assert len(candidates) == 2
    assert candidates[0].start < candidates[0].end
    assert candidates[1].start < candidates[1].end
    assert candidates[0].end <= candidates[1].start or candidates[1].end <= candidates[0].start


def test_detect_highlights_skips_too_short_segments():
    segments = [
        _make_segment(0.0, 1.0, "短句。"),
        _make_segment(1.1, 2.0, "再短。"),
    ]

    assert detect_highlights(segments, min_clip_duration=10.0) == []


def test_export_highlight_clips_writes_manifest(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        type(
            "Candidate",
            (),
            {
                "start": 1.0,
                "end": 6.0,
                "score": 1.5,
                "reason": "句尾完整",
                "text": "測試高光",
            },
        )(),
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D libx264\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        exports = export_highlight_clips(source, candidates, output_dir, padding_seconds=1.0)

    assert len(exports) == 1
    assert exports[0].output_path.endswith("input_highlight_01.mp4")
    assert (output_dir / "input_highlights.json").exists()
    assert mock_run.call_count == 2


def test_export_highlight_clips_falls_back_when_libx264_is_missing(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        type(
            "Candidate",
            (),
            {
                "start": 1.0,
                "end": 6.0,
                "score": 1.5,
                "reason": "句尾完整",
                "text": "測試高光",
            },
        )(),
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D mpeg4\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        export_highlight_clips(source, candidates, output_dir, padding_seconds=1.0)

    export_cmd = mock_run.call_args_list[1].args[0]
    assert "mpeg4" in export_cmd
    assert "-q:v" in export_cmd


def test_export_highlight_clips_prefers_gpu_encoder_when_available(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        type(
            "Candidate",
            (),
            {
                "start": 1.0,
                "end": 6.0,
                "score": 1.5,
                "reason": "句尾完整",
                "text": "測試高光",
            },
        )(),
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D h264_nvenc\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        export_highlight_clips(source, candidates, output_dir, padding_seconds=1.0, prefer_gpu=True)

    export_cmd = mock_run.call_args_list[1].args[0]
    assert "h264_nvenc" in export_cmd
    assert "-cq:v" in export_cmd