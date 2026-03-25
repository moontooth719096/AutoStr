"""Speech transcription using faster-whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A single transcription segment."""

    start: float          # seconds
    end: float            # seconds
    text: str
    words: list[dict] = field(default_factory=list)


def transcribe(
    audio_path: str | Path,
    model_size: str = "medium",
    language: str = "zh",
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
) -> list[TranscriptSegment]:
    """Transcribe *audio_path* with faster-whisper.

    Parameters
    ----------
    audio_path:
        Path to a WAV file (mono, 16 kHz recommended).
    model_size:
        Whisper model size: ``tiny``, ``base``, ``small``, ``medium``,
        ``large-v2``, ``large-v3``.  Default ``medium`` gives a good
        balance between speed and accuracy for Chinese.
    language:
        BCP-47 language code.  Use ``"zh"`` for Mandarin Chinese.
    device:
        ``"cpu"`` or ``"cuda"``.
    compute_type:
        Quantisation type – ``"int8"`` for CPU, ``"float16"`` for GPU.
    beam_size:
        Beam width for decoding.
    vad_filter:
        Enable faster-whisper's built-in VAD filter to remove silence
        segments and reduce hallucinations.

    Returns
    -------
    List[TranscriptSegment]
        Ordered list of transcription segments with timing.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. "
            "Run: pip install faster-whisper"
        ) from exc

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    logger.info(
        "Loading faster-whisper model '%s' on %s (%s)…",
        model_size, device, compute_type,
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("Transcribing %s …", audio_path)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    logger.info(
        "Detected language '%s' (probability %.2f)",
        info.language, info.language_probability,
    )

    segments: list[TranscriptSegment] = []
    for seg in segments_iter:
        words = [
            {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
            for w in (seg.words or [])
        ]
        segments.append(
            TranscriptSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                words=words,
            )
        )
        logger.debug("[%.2f → %.2f] %s", seg.start, seg.end, seg.text.strip())

    logger.info("Transcription complete – %d segments", len(segments))
    return segments
