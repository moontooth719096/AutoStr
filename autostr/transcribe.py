"""faster-whisper transcription stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from model_loader import load_whisper_model

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


def transcribe(
    audio_path: str | Path,
    model_size: str = "medium",
    model_dir: str | Path | None = None,
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
) -> list[TranscriptSegment]:
    logger.info("Loading model: %s (device=%s, compute_type=%s)", model_size, device, compute_type)

    model = load_whisper_model(
        model_size,
        model_dir=model_dir,
        device=device,
        compute_type=compute_type,
    )

    logger.info("Model loaded. Starting transcription...")

    segments_iter, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
    )

    results: list[TranscriptSegment] = []
    for idx, seg in enumerate(segments_iter, start=1):
        text = (seg.text or "").strip()
        logger.info("Transcribed segment %d: %.2f -> %.2f", idx, seg.start, seg.end)
        results.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=text,
            )
        )

    logger.info("Transcription complete. %d segments generated.", len(results))
    return results