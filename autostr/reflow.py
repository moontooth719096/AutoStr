"""Chinese subtitle segmentation and reflow logic."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass
class SubtitleEntry:
    """A single SRT subtitle entry."""

    index: int
    start: float
    end: float
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _split_text_on_punctuation(text: str) -> list[str]:
    """
    Split Chinese text into chunks while keeping punctuation attached.
    """
    text = text.strip()
    if not text:
        return []

    parts = re.split(r"([。！？；，、])", text)
    chunks: list[str] = []
    buffer = ""

    for i in range(0, len(parts), 2):
        left = parts[i].strip()
        right = parts[i + 1] if i + 1 < len(parts) else ""

        if left:
            buffer += left
        if right:
            buffer += right

        if buffer.strip():
            chunks.append(buffer.strip())
            buffer = ""

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]


def _wrap_to_two_lines(text: str, max_chars_per_line: int = 16) -> list[str]:
    """
    Wrap Chinese text into at most two lines.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars_per_line:
        return [text]

    lines: list[str] = []
    current = ""

    for ch in text:
        current += ch
        if len(current) >= max_chars_per_line:
            lines.append(current)
            current = ""

    if current:
        lines.append(current)

    return lines[:2]


def _extract_text(seg) -> str:
    if isinstance(seg, dict):
        return str(seg.get("text", "")).strip()
    return str(getattr(seg, "text", "")).strip()


def _extract_start(seg) -> float:
    if isinstance(seg, dict):
        return float(seg.get("start", 0.0))
    return float(getattr(seg, "start", 0.0))


def _extract_end(seg) -> float:
    if isinstance(seg, dict):
        return float(seg.get("end", 0.0))
    return float(getattr(seg, "end", 0.0))


def _extract_words(seg):
    if isinstance(seg, dict):
        return seg.get("words")
    return getattr(seg, "words", None)


def _segment_time_chunks(start: float, end: float, num_chunks: int) -> list[tuple[float, float]]:
    """
    Split a segment duration evenly across chunks.
    """
    if num_chunks <= 0:
        return []

    duration = max(end - start, 0.5)
    per_chunk = duration / num_chunks

    spans = []
    for i in range(num_chunks):
        chunk_start = start + i * per_chunk
        chunk_end = start + (i + 1) * per_chunk
        spans.append((chunk_start, chunk_end))
    return spans


def _apply_timing_adjustments(
    entry_start: float,
    entry_end: float,
    start_delay_ms: int,
    global_shift_ms: int,
    min_duration: float,
    max_duration: float,
) -> tuple[float, float]:
    entry_start += start_delay_ms / 1000.0
    entry_start += global_shift_ms / 1000.0
    entry_end += global_shift_ms / 1000.0

    if entry_end <= entry_start:
        entry_end = entry_start + min_duration

    entry_end = max(entry_end, entry_start + min_duration)
    entry_end = min(entry_end, entry_start + max_duration)

    return entry_start, entry_end


def reflow(
    segments: Iterable,
    max_chars_per_line: int = 16,
    start_delay_ms: int = 0,
    global_shift_ms: int = 0,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
) -> list[SubtitleEntry]:
    """
    Convert transcript segments into SRT subtitle entries.

    Supports:
    - AlignedSegment objects from autostr.align
    - raw dict-like transcript segments
    - fallback TranscriptSegment-like objects without `.words`
    """
    entries: list[SubtitleEntry] = []
    index = 1

    for seg in segments:
        text = _extract_text(seg)
        if not text:
            continue

        start = _extract_start(seg)
        end = _extract_end(seg)

        chunks = _split_text_on_punctuation(text)
        if not chunks:
            continue

        # If words exist, we still fall back to chunk-level timing for now,
        # but preserve compatibility with aligned and non-aligned inputs.
        _ = _extract_words(seg)

        chunk_spans = _segment_time_chunks(start, end, len(chunks))

        for chunk, (chunk_start, chunk_end) in zip(chunks, chunk_spans):
            entry_start, entry_end = _apply_timing_adjustments(
                chunk_start,
                chunk_end,
                start_delay_ms=start_delay_ms,
                global_shift_ms=global_shift_ms,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            lines = _wrap_to_two_lines(chunk, max_chars_per_line)
            if not lines:
                continue

            entries.append(
                SubtitleEntry(
                    index=index,
                    start=entry_start,
                    end=entry_end,
                    lines=lines,
                )
            )
            index += 1

    logger.info("Reflow complete – %d subtitle entries generated", len(entries))
    return entries