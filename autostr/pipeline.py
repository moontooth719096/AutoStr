"""End-to-end pipeline orchestrator."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def run(
    video_path: str | Path,
    output_srt: str | Path | None = None,
    # Transcription
    model_size: str = "medium",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
    # Alignment
    use_whisperx: bool = True,
    # Reflow
    max_chars_per_line: int = 16,
    start_delay_ms: int = 0,
    global_shift_ms: int = 0,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
    # Audio
    keep_audio: bool = False,
) -> Path:
    """Run the full AutoStr pipeline on *video_path*.

    Steps
    -----
    1. Extract mono 16 kHz audio with **ffmpeg**.
    2. Transcribe Chinese speech with **faster-whisper**.
    3. Refine word-level alignment with **WhisperX** (optional, graceful fallback).
    4. Segment and reflow subtitles for Chinese text.
    5. Write the output SRT file.

    Parameters
    ----------
    video_path:
        Input video (or audio) file.
    output_srt:
        Destination SRT path.  Defaults to the same directory and stem
        as *video_path* with a ``.srt`` extension.
    model_size:
        faster-whisper model size (``tiny`` / ``base`` / ``small`` /
        ``medium`` / ``large-v2`` / ``large-v3``).
    language:
        BCP-47 language code – ``"zh"`` for Mandarin.
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type – ``"int8"`` (CPU) or ``"float16"`` (GPU).
    use_whisperx:
        Whether to attempt WhisperX fine-grained alignment.
    max_chars_per_line:
        Maximum Chinese characters per subtitle line (14–18 recommended).
    start_delay_ms:
        Milliseconds to add to every subtitle start time.  Use a
        positive value (e.g. 100–200) if subtitles appear too early.
    global_shift_ms:
        Global shift applied to both start and end times (positive = later).
    min_duration:
        Minimum subtitle display duration in seconds.
    max_duration:
        Maximum subtitle display duration in seconds.
    keep_audio:
        If ``True``, keep the intermediate WAV file.

    Returns
    -------
    Path
        Path to the produced SRT file.
    """
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

    # ── Step 1: Extract audio ────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "audio.wav"
        logger.info("Step 1/4 – Extracting audio from %s …", video_path)
        extract_audio(video_path, audio_path)

        if keep_audio:
            kept = video_path.with_suffix(".wav")
            import shutil
            shutil.copy2(audio_path, kept)
            logger.info("Intermediate audio saved to: %s", kept)

        # ── Step 2: Transcribe ───────────────────────────────────────────────
        logger.info("Step 2/4 – Transcribing speech …")
        segments = transcribe(
            audio_path,
            model_size=model_size,
            language=language,
            device=device,
            compute_type=compute_type,
        )

        # ── Step 3: Align ────────────────────────────────────────────────────
        if use_whisperx:
            logger.info("Step 3/4 – Fine-grained alignment with WhisperX …")
            segments = align(segments, audio_path, language=language, device=device)
        else:
            logger.info("Step 3/4 – WhisperX alignment skipped.")

        # ── Step 4: Reflow & write SRT ───────────────────────────────────────
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

    logger.info("Pipeline complete.  Output: %s", output_srt)
    return output_srt
