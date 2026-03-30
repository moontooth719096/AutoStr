"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

MEDIA_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".mp3",
    ".m4a",
    ".aac",
    ".wav",
    ".flac",
    ".ogg",
}


def run(
    video_path: str | Path,
    output_srt: str | Path | None = None,
    model_size: str = "medium",
    model_dir: str | Path | None = None,
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
            model_dir=model_dir,
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


def find_missing_subtitle_jobs(input_dir: str | Path, output_dir: str | Path) -> list[tuple[Path, Path]]:
    """Return media files whose matching SRT file is missing from *output_dir*."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    resolved_input_dir = input_dir.resolve(strict=False)
    resolved_output_dir = output_dir.resolve(strict=False)

    jobs: list[tuple[Path, Path]] = []
    for media_path in sorted(input_dir.rglob("*")):
        if not media_path.is_file():
            continue
        if media_path.suffix.lower() not in MEDIA_SUFFIXES:
            continue

        resolved_media_path = media_path.resolve(strict=False)
        if resolved_output_dir == resolved_media_path or resolved_output_dir in resolved_media_path.parents:
            continue

        relative_path = resolved_media_path.relative_to(resolved_input_dir)
        target_srt = output_dir / relative_path.with_suffix(".srt")
        if not target_srt.exists():
            jobs.append((media_path, target_srt))

    return jobs


def run_missing_subtitles(
    input_dir: str | Path,
    output_dir: str | Path,
    model_size: str = "medium",
    model_dir: str | Path | None = None,
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
) -> list[Path]:
    """Process every media file under *input_dir* that is missing a sibling SRT in *output_dir*."""
    jobs = find_missing_subtitle_jobs(input_dir, output_dir)

    if not jobs:
        logger.info("Batch scan complete. No missing subtitles found.")
        return []

    logger.info("Batch scan complete. Processing %d missing subtitle file(s).", len(jobs))

    outputs: list[Path] = []
    for index, (video_path, output_srt) in enumerate(jobs, start=1):
        logger.info("Batch item %d/%d: %s -> %s", index, len(jobs), video_path, output_srt)
        outputs.append(
            run(
                video_path=video_path,
                output_srt=output_srt,
                model_size=model_size,
                model_dir=model_dir,
                language=language,
                device=device,
                compute_type=compute_type,
                use_whisperx=use_whisperx,
                max_chars_per_line=max_chars_per_line,
                start_delay_ms=start_delay_ms,
                global_shift_ms=global_shift_ms,
                min_duration=min_duration,
                max_duration=max_duration,
                keep_audio=keep_audio,
            )
        )

    return outputs