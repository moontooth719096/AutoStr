"""Fine-grained alignment using WhisperX (word-level timestamps)."""

from __future__ import annotations

import logging
from pathlib import Path

from autostr.transcribe import TranscriptSegment

logger = logging.getLogger(__name__)


def align(
    segments: list[TranscriptSegment],
    audio_path: str | Path,
    language: str = "zh",
    device: str = "cpu",
) -> list[TranscriptSegment]:
    """Refine word-level timing with WhisperX.

    WhisperX performs forced-alignment using a phoneme-level acoustic
    model, which produces more accurate word timestamps than Whisper's
    built-in decoder.

    If WhisperX is not installed or alignment fails, the original
    *segments* are returned unchanged so the pipeline degrades gracefully.

    Parameters
    ----------
    segments:
        Segments produced by :func:`autostr.transcribe.transcribe`.
    audio_path:
        Path to the audio file (must match the file used for transcription).
    language:
        BCP-47 language code (``"zh"`` for Mandarin).
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    List[TranscriptSegment]
        Segments with refined (word-level) timing where available.
    """
    try:
        import whisperx  # type: ignore
    except ImportError:
        logger.warning(
            "whisperx not installed – skipping fine-grained alignment. "
            "Install with: pip install whisperx"
        )
        return segments

    audio_path = Path(audio_path)

    try:
        logger.info("Loading WhisperX alignment model for language '%s'…", language)
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )

        logger.info("Running WhisperX alignment on %s …", audio_path)
        audio = whisperx.load_audio(str(audio_path))

        # Convert our segments to the dict format WhisperX expects
        wx_segments = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": seg.words,
            }
            for seg in segments
        ]

        result = whisperx.align(
            wx_segments,
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        aligned: list[TranscriptSegment] = []
        for seg in result.get("segments", []):
            words = seg.get("words", [])
            aligned.append(
                TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=words,
                )
            )

        logger.info("WhisperX alignment complete – %d segments", len(aligned))
        return aligned

    except Exception as exc:
        logger.warning(
            "WhisperX alignment failed (%s) – falling back to faster-whisper timestamps.",
            exc,
        )
        return segments
