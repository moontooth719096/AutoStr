"""WhisperX alignment stage with graceful fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AlignedWord:
    word: str
    start: float
    end: float
    score: float | None = None


@dataclass
class AlignedSegment:
    start: float
    end: float
    text: str
    words: list[AlignedWord] | None = None


def _to_aligned_segment(seg) -> AlignedSegment:
    """
    Convert a segment-like object to an AlignedSegment.
    Supports dicts and objects with start/end/text/words.
    """
    if isinstance(seg, dict):
        words = seg.get("words")
        return AlignedSegment(
            start=float(seg.get("start", 0.0)),
            end=float(seg.get("end", 0.0)),
            text=str(seg.get("text", "")).strip(),
            words=_normalize_words(words),
        )

    words = getattr(seg, "words", None)
    return AlignedSegment(
        start=float(getattr(seg, "start", 0.0)),
        end=float(getattr(seg, "end", 0.0)),
        text=str(getattr(seg, "text", "")).strip(),
        words=_normalize_words(words),
    )


def _normalize_words(words) -> list[AlignedWord] | None:
    """
    Normalize WhisperX words into a list of AlignedWord objects.
    """
    if not words:
        return None

    normalized: list[AlignedWord] = []

    for w in words:
        if isinstance(w, dict):
            word = str(w.get("word", "")).strip()
            if not word:
                continue
            normalized.append(
                AlignedWord(
                    word=word,
                    start=float(w.get("start", 0.0)),
                    end=float(w.get("end", 0.0)),
                    score=w.get("score"),
                )
            )
        else:
            word = str(getattr(w, "word", "")).strip()
            if not word:
                continue
            normalized.append(
                AlignedWord(
                    word=word,
                    start=float(getattr(w, "start", 0.0)),
                    end=float(getattr(w, "end", 0.0)),
                    score=getattr(w, "score", None),
                )
            )

    return normalized or None


def _fallback_segments(segments: Iterable) -> list[AlignedSegment]:
    """
    Convert raw transcript segments into aligned-segment-like objects
    without word-level timing.
    """
    results: list[AlignedSegment] = []
    for seg in segments:
        results.append(_to_aligned_segment(seg))
    return results


def align(
    segments: Iterable,
    audio_path: str | Path,
    language: str = "zh",
    device: str = "cpu",
):
    """
    Attempt WhisperX alignment.

    If alignment succeeds:
        return aligned segments with `.words`

    If alignment fails:
        return normalized fallback segments with no `.words`
    """
    logger.info("Loading WhisperX alignment model for language '%s'…", language)

    # Delay importing whisperx so CPU-only or optional environments still work.
    try:
        import whisperx
    except Exception as exc:
        logger.warning("WhisperX is not available (%s) – falling back to timestamps.", exc)
        return _fallback_segments(segments)

    # Normalize input early so fallback always has a consistent shape.
    raw_segments = _fallback_segments(segments)

    try:
        logger.info("Running WhisperX alignment on %s …", audio_path)

        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=device,
        )

        # WhisperX expects transcript-like structures with start/end/text
        transcript = [
            {"start": seg.start, "end": seg.end, "text": seg.text}
            for seg in raw_segments
        ]

        aligned = whisperx.align(
            transcript,
            model_a,
            metadata,
            str(audio_path),
            device=device,
            return_char_alignments=False,
        )

        aligned_segments = []
        for seg in aligned["segments"]:
            words = seg.get("words")
            aligned_segments.append(
                AlignedSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=str(seg.get("text", "")).strip(),
                    words=_normalize_words(words),
                )
            )

        logger.info("WhisperX alignment complete – %d segments aligned.", len(aligned_segments))
        return aligned_segments

    except Exception as exc:
        logger.warning("WhisperX alignment failed (%s) – falling back to faster-whisper timestamps.", exc)
        return raw_segments