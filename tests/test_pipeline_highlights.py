"""Tests for highlight export integration in the pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from autostr.highlight import HighlightCandidate, HighlightDetectionReport
from autostr.pipeline import run
from autostr.reflow import SubtitleEntry


def _fake_highlight() -> HighlightCandidate:
    return HighlightCandidate(
        start=0.0,
        end=5.0,
        score=1.0,
        raw_score=0.8,
        reason="句尾完整",
        text="測試高光。",
        scores={"completion": 1.0},
    )


def _fake_report() -> HighlightDetectionReport:
    highlight = _fake_highlight()
    return HighlightDetectionReport(
        strategy="balanced",
        reranker="none",
        weight_overrides={},
        target_count=3,
        min_clip_duration=15.0,
        max_clip_duration=60.0,
        min_gap_seconds=4.0,
        candidates=[highlight],
        selected=[highlight],
    )


def _fake_empty_report() -> HighlightDetectionReport:
    return HighlightDetectionReport(
        strategy="balanced",
        reranker="none",
        weight_overrides={},
        target_count=3,
        min_clip_duration=15.0,
        max_clip_duration=60.0,
        min_gap_seconds=4.0,
        candidates=[],
        selected=[],
    )


def test_run_exports_highlights_into_srt_folder(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "測試高光。"})()]
    fake_entries = [SubtitleEntry(index=1, start=0.0, end=5.0, lines=["測試高光。"])]
    fake_report = _fake_report()

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=fake_entries,
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
    ) as mock_analyze, patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True)

    analyze_args = mock_analyze.call_args.args[0]
    assert analyze_args == fake_entries
    _, call_kwargs = mock_export.call_args
    assert call_kwargs["output_dir"].name == "input_highlights"
    assert call_kwargs["output_dir"].parent == Path("/output")
    assert call_kwargs["manifest_metadata"]["selection"]["selected_count"] == 1


def test_run_prefers_gpu_for_highlight_export_when_device_is_cuda(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "測試高光。"})()]
    fake_entries = [SubtitleEntry(index=1, start=0.0, end=5.0, lines=["測試高光。"])]
    fake_report = _fake_report()

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=fake_entries,
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
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
    fake_entries = [SubtitleEntry(index=1, start=0.0, end=5.0, lines=["測試高光。"])]
    fake_report = _fake_report()

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ), patch(
        "autostr.align.align",
        return_value=fake_segments,
    ), patch(
        "autostr.reflow.reflow",
        return_value=fake_entries,
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
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
    existing_srt = tmp_path / "input_highlights" / "input.srt"
    existing_srt.parent.mkdir()
    existing_srt.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n測試高光。\n\n",
        encoding="utf-8",
    )

    fake_report = _fake_report()

    with patch("autostr.audio.extract_audio") as mock_extract, patch(
        "autostr.transcribe.transcribe",
    ) as mock_transcribe, patch(
        "autostr.align.align",
    ) as mock_align, patch(
        "autostr.reflow.reflow",
    ) as mock_reflow, patch(
        "autostr.srt_writer.write_srt",
    ) as mock_write_srt, patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
    ) as mock_detect, patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True, output_srt=existing_srt)

    mock_extract.assert_not_called()
    mock_transcribe.assert_not_called()
    mock_align.assert_not_called()
    mock_reflow.assert_not_called()
    mock_write_srt.assert_not_called()
    mock_detect.assert_called_once()
    analyze_args = mock_detect.call_args.args[0]
    assert analyze_args[0].text == "測試高光。"
    mock_export.assert_called_once()


def test_run_highlights_do_not_reuse_input_sidecar_subtitles(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    input_sidecar = tmp_path / "input.srt"
    input_sidecar.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n舊字幕。\n\n",
        encoding="utf-8",
    )

    fake_segments = [type("Seg", (), {"start": 0.0, "end": 5.0, "text": "新字幕。"})()]
    fake_entries = [SubtitleEntry(index=1, start=0.0, end=5.0, lines=["新字幕。"])]
    fake_report = _fake_report()
    highlight_srt = tmp_path / "input_highlights" / "input.srt"

    with patch("autostr.audio.extract_audio") as mock_extract, patch(
        "autostr.transcribe.transcribe",
        return_value=fake_segments,
    ) as mock_transcribe, patch(
        "autostr.align.align",
        return_value=fake_segments,
    ) as mock_align, patch(
        "autostr.reflow.reflow",
        return_value=fake_entries,
    ) as mock_reflow, patch(
        "autostr.srt_writer.write_srt",
    ) as mock_write_srt, patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
    ) as mock_detect, patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True, output_srt=highlight_srt)

    mock_extract.assert_called_once()
    mock_transcribe.assert_called_once()
    mock_align.assert_called_once()
    mock_reflow.assert_called_once()
    mock_write_srt.assert_called_once_with(fake_entries, highlight_srt)
    mock_detect.assert_called_once_with(
        fake_entries,
        target_count=3,
        min_clip_duration=15.0,
        max_clip_duration=60.0,
        min_gap_seconds=4.0,
        strategy="balanced",
        reranker="none",
        weight_overrides=None,
    )
    mock_export.assert_called_once()


def test_run_writes_no_highlights_marker_when_nothing_selected(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    existing_srt = tmp_path / "input_highlights" / "input.srt"
    existing_srt.parent.mkdir()
    existing_srt.write_text(
        "1\n00:00:00,000 --> 00:00:05,000\n測試字幕。\n\n",
        encoding="utf-8",
    )
    fake_report = _fake_empty_report()

    with patch("autostr.audio.extract_audio"), patch(
        "autostr.transcribe.transcribe",
    ), patch(
        "autostr.align.align",
    ), patch(
        "autostr.reflow.reflow",
    ), patch(
        "autostr.srt_writer.write_srt",
    ), patch(
        "autostr.highlight.analyze_highlights",
        return_value=fake_report,
    ) as mock_detect, patch(
        "autostr.highlight.export_highlight_clips",
        return_value=[],
    ) as mock_export:
        run(video, export_highlights=True, output_srt=existing_srt)

    marker_path = existing_srt.parent / "input_no_highlights.json"
    assert marker_path.exists()
    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_payload["status"] == "no_highlights"
    assert marker_payload["message"] == "No highlight clips met the selection criteria."
    assert marker_payload["selection"]["selected_count"] == 0
    mock_detect.assert_called_once()
    mock_export.assert_not_called()