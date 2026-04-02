"""Tests for highlight export integration in the pipeline."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from autostr.pipeline import run


def test_run_exports_highlights_into_srt_folder(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "測試高光。"})()]
    fake_highlights = [type("Candidate", (), {"start": 0.0, "end": 5.0, "score": 1.0, "reason": "句尾完整", "text": "測試高光。"})()]

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=[],
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.detect_highlights",
        return_value=fake_highlights,
    ), patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True)

    _, call_kwargs = mock_export.call_args
    assert call_kwargs["output_dir"].name == "input_highlights"
    assert call_kwargs["output_dir"].parent == Path("/output")


def test_run_prefers_gpu_for_highlight_export_when_device_is_cuda(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "測試高光。"})()]
    fake_highlights = [type("Candidate", (), {"start": 0.0, "end": 5.0, "score": 1.0, "reason": "句尾完整", "text": "測試高光。"})()]

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=[],
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.detect_highlights",
        return_value=fake_highlights,
    ), patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True, device="cuda")

    _, call_kwargs = mock_export.call_args
    assert call_kwargs["prefer_gpu"] is True


def test_run_highlight_encoder_override_wins_over_cuda(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "測試高光。"})()]
    fake_highlights = [type("Candidate", (), {"start": 0.0, "end": 5.0, "score": 1.0, "reason": "句尾完整", "text": "測試高光。"})()]

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=[],
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.detect_highlights",
        return_value=fake_highlights,
    ), patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True, device="cuda", highlight_encoder="cpu")

    _, call_kwargs = mock_export.call_args
    assert call_kwargs["prefer_gpu"] is False


def test_run_reuses_existing_subtitles_for_highlights(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    existing_srt = tmp_path / "input.srt"
    existing_srt.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n測試高光。\n\n",
        encoding="utf-8",
    )

    fake_highlights = [type("Candidate", (), {"start": 0.0, "end": 5.0, "score": 1.0, "reason": "句尾完整", "text": "測試高光。"})()]

    with patch("autostr.audio.extract_audio") as mock_extract, patch(
        "autostr.transcribe.transcribe",
    ) as mock_transcribe, patch(
        "autostr.align.align",
    ) as mock_align, patch(
        "autostr.reflow.reflow",
    ) as mock_reflow, patch(
        "autostr.srt_writer.write_srt",
    ) as mock_write_srt, patch(
        "autostr.highlight.detect_highlights",
        return_value=fake_highlights,
    ) as mock_detect, patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True)

    mock_extract.assert_not_called()
    mock_transcribe.assert_not_called()
    mock_align.assert_not_called()
    mock_reflow.assert_not_called()
    mock_write_srt.assert_called_once()
    mock_detect.assert_called_once()
    mock_export.assert_called_once()