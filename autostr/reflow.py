"""Chinese subtitle segmentation and reflow.

This module converts raw transcription segments into well-formatted SRT
subtitle entries by:

1. Splitting long segments on Chinese punctuation and pause-aware heuristics.
2. Limiting each subtitle to ``max_chars_per_line`` characters per line.
3. Wrapping text to at most 2 lines.
4. Optionally delaying subtitle start times to avoid appearing too early.
5. Optionally applying a global or local timing shift.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from autostr.transcribe import TranscriptSegment

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SubtitleEntry:
    """A single SRT subtitle entry."""

    index: int
    start: float   # seconds
    end: float     # seconds
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


# ──────────────────────────────────────────────────────────────────────────────
# Chinese punctuation used as natural break points
# ──────────────────────────────────────────────────────────────────────────────

# Strong breaks – sentence boundaries
_STRONG_BREAKS = re.compile(r"([。！？…]+)")

# Weak breaks – clause / pause indicators
_WEAK_BREAKS = re.compile(r"([，；、：]+)")


# ──────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ──────────────────────────────────────────────────────────────────────────────

def _split_text_on_punctuation(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks no longer than *max_chars*.

    Splits are attempted first on strong breaks (sentence-ending
    punctuation), then on weak breaks (commas, semi-colons, etc.).
    If a chunk is still longer than *max_chars* after all splits, it
    is hard-wrapped at *max_chars*.
    """
    # Phase 1 – split on strong sentence boundaries
    raw_chunks: list[str] = []
    parts = _STRONG_BREAKS.split(text)
    buf = ""
    for part in parts:
        buf += part
        if _STRONG_BREAKS.match(part):
            if buf.strip():
                raw_chunks.append(buf.strip())
            buf = ""
    if buf.strip():
        raw_chunks.append(buf.strip())

    if not raw_chunks:
        raw_chunks = [text.strip()]

    # Phase 2 – further split long chunks on weak breaks
    chunks: list[str] = []
    for chunk in raw_chunks:
        if len(chunk) <= max_chars:
            chunks.append(chunk)
            continue
        sub_parts = _WEAK_BREAKS.split(chunk)
        sub_buf = ""
        for sub in sub_parts:
            sub_buf += sub
            if _WEAK_BREAKS.match(sub) and len(sub_buf) >= max_chars // 2:
                if sub_buf.strip():
                    chunks.append(sub_buf.strip())
                sub_buf = ""
        if sub_buf.strip():
            chunks.append(sub_buf.strip())

    # Phase 3 – hard-wrap anything still exceeding max_chars
    result: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_chars:
            result.append(chunk[:max_chars])
            chunk = chunk[max_chars:]
        if chunk:
            result.append(chunk)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Line wrapping
# ──────────────────────────────────────────────────────────────────────────────

def _wrap_to_two_lines(text: str, max_chars_per_line: int) -> list[str]:
    """Wrap *text* to at most 2 lines, each no longer than *max_chars_per_line*.

    The split point targets the midpoint of the string; it is adjusted to
    fall on a punctuation boundary when possible.
    """
    if len(text) <= max_chars_per_line:
        return [text]

    mid = len(text) // 2

    # Try to find a punctuation character near the midpoint to split on
    best_pos: int | None = None
    for offset in range(0, mid + 1):
        for pos in (mid + offset, mid - offset):
            if 0 < pos < len(text) and text[pos] in "，。！？；、：":
                best_pos = pos + 1  # split after the punctuation
                break
        if best_pos is not None:
            break

    if best_pos is None:
        best_pos = mid

    line1 = text[:best_pos].strip()
    line2 = text[best_pos:].strip()

    # Hard-truncate if a line is still too long
    if len(line1) > max_chars_per_line:
        line1 = line1[:max_chars_per_line]
    if len(line2) > max_chars_per_line:
        line2 = line2[:max_chars_per_line]

    return [l for l in (line1, line2) if l]


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _interpolate_times(
    parent_start: float,
    parent_end: float,
    chunk_index: int,
    total_chunks: int,
    words: list[dict],
    chunk_text: str,
) -> tuple[float, float]:
    """Estimate start/end times for a text *chunk* derived from a parent segment.

    Strategy (in order of preference):
    1. Use word-level timestamps when available.
    2. Linearly interpolate within the parent segment duration.
    """
    duration = parent_end - parent_start

    if words:
        # Try to map the chunk to word timestamps
        _punct_re = re.compile(r"[，。！？；、：]")
        chunk_clean = _punct_re.sub("", chunk_text)
        full_text = _punct_re.sub("", "".join(w.get("word", "") for w in words))
        if full_text and chunk_clean:
            start_frac = full_text.find(chunk_clean[:max(1, len(chunk_clean) // 3)])
            if start_frac >= 0:
                ratio = start_frac / max(len(full_text), 1)
                start = parent_start + ratio * duration
                end = start + (len(chunk_clean) / max(len(full_text), 1)) * duration
                end = min(end, parent_end)
                return start, end

    # Fallback: linear interpolation
    step = duration / total_chunks
    start = parent_start + chunk_index * step
    end = start + step
    return start, min(end, parent_end)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def reflow(
    segments: list[TranscriptSegment],
    max_chars_per_line: int = 16,
    start_delay_ms: int = 0,
    global_shift_ms: int = 0,
    min_duration: float = 0.8,
    max_duration: float = 7.0,
) -> list[SubtitleEntry]:
    """Convert transcription segments into reflowed subtitle entries.

    Parameters
    ----------
    segments:
        Raw segments from :func:`autostr.transcribe.transcribe` or
        :func:`autostr.align.align`.
    max_chars_per_line:
        Maximum number of Chinese characters per subtitle line.
        Recommended: 14–18 for standard screen sizes.
    start_delay_ms:
        Milliseconds to add to every subtitle start time.  Set to a
        positive value (e.g. ``100``–``200``) if subtitles appear too
        early.  Set to a negative value to move them earlier.
    global_shift_ms:
        Global timing shift applied to *both* start and end times of
        every subtitle (positive = later, negative = earlier).  Useful
        when the entire SRT track is consistently off-sync.
    min_duration:
        Minimum subtitle display duration in seconds.  Short segments
        are extended to this value.
    max_duration:
        Maximum subtitle display duration in seconds.  Over-long
        segments are clamped to this value.

    Returns
    -------
    List[SubtitleEntry]
        Ordered subtitle entries ready for SRT serialisation.
    """
    entries: List[SubtitleEntry] = []
    index = 1

    global_shift = global_shift_ms / 1000.0
    start_delay = start_delay_ms / 1000.0

    for seg in segments:
        if not seg.text:
            continue

        chunks = _split_text_on_punctuation(seg.text, max_chars=max_chars_per_line * 2)

        for i, chunk in enumerate(chunks):
            raw_start, raw_end = _interpolate_times(
                seg.start, seg.end, i, len(chunks), seg.words, chunk
            )

            # Apply optional start delay (only to the start time)
            entry_start = raw_start + start_delay + global_shift
            entry_end = raw_end + global_shift

            # Clamp start to zero
            entry_start = max(0.0, entry_start)

            # Enforce min/max duration
            entry_end = max(entry_end, entry_start + min_duration)
            entry_end = min(entry_end, entry_start + max_duration)

            lines = _wrap_to_two_lines(chunk, max_chars_per_line)

            entries.append(
                SubtitleEntry(index=index, start=entry_start, end=entry_end, lines=lines)
            )
            index += 1

    logger.info("Reflow complete – %d subtitle entries generated", len(entries))
    return entries
