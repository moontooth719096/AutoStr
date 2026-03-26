"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run(
    video_path: str | Path,
    output_srt: str | Path | None = None,
    model_size: str = "medium",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
    use_whisperx: bool = True,
    max_chars_per_line: int = 16,
    start_delay_ms: int = 0,
    global_shift_ms: int = 0,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    keep_audio: bool = False,
) -> Path:
    from autostr.audio import extract_audio
    from autostr.transcribe import transcribe
    from autostr.align import align
    from autostr.reflow import reflow
    from autostr.srt_writer import write_srt

    video_path = Path(video_path)

    if output_srt is None:
        output_srt = video_path.with_suffix(".srt")
    else:
        output_srt = Path(output_srt)

    logger.info("Starting pipeline for: %s", video_path)
    logger.info("Output will be written to: %s", output_srt)

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"

        logger.info("Step 1/4 – Extracting audio from %s …", video_path)
        extract_audio(video_path, audio_path)

        if keep_audio:
            kept = video_path.with_suffix(".wav")
            import shutil
            shutil.copy2(audio_path, kept)
            logger.info("Intermediate audio saved to: %s", kept)

        logger.info("Step 2/4 – Transcribing speech …")
        segments = transcribe(
            audio_path,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
        )

        if use_whisperx:
            logger.info("Step 3/4 – Fine-grained alignment with WhisperX …")
            segments = align(segments, audio_path, language=language, device=device)
        else:
            logger.info("Step 3/4 – WhisperX alignment skipped.")

        logger.info("Step 4/4 – Reflowing subtitles …")
        entries = reflow(
            segments,
            max_chars_per_line=max_chars_per_line,
            start_delay_ms=start_delay_ms,
            global_shift_ms=global_shift_ms,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        write_srt(entries, output_srt)

    logger.info("Pipeline complete. Output: %s", output_srt)
    return output_srt