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