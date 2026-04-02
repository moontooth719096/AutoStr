"""SRT file serialisation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from autostr.reflow import SubtitleEntry

logger = logging.getLogger(__name__)

_TIME_RANGE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


def _format_time(seconds: float) -> str:
    """Convert *seconds* (float) to SRT timestamp ``HH:MM:SS,mmm``."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _parse_timecode(timecode: str) -> float:
    hours, minutes, seconds_part = timecode.split(":")
    seconds, milliseconds = seconds_part.split(",")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(milliseconds) / 1000.0
    )


def read_srt(input_path: str | Path) -> list[SubtitleEntry]:
    """Read an SRT file and return subtitle entries.

    The parser is intentionally small and tolerant of the standard SRT layout:
    index line, timestamp line, one or more text lines, blank line.
    """
    input_path = Path(input_path)
    content = input_path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    entries: list[SubtitleEntry] = []
    blocks = re.split(r"\r?\n\s*\r?\n", content.strip())
    for block in blocks:
        block_lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]
        if len(block_lines) < 2:
            continue

        time_line_index = None
        for index, line in enumerate(block_lines[:2]):
            if _TIME_RANGE_RE.match(line.strip()):
                time_line_index = index
                break

        if time_line_index is None:
            continue

        match = _TIME_RANGE_RE.match(block_lines[time_line_index].strip())
        if match is None:
            continue

        start = _parse_timecode(match.group("start"))
        end = _parse_timecode(match.group("end"))
        text_lines = block_lines[time_line_index + 1 :]
        if not text_lines:
            text_lines = [""]

        entries.append(
            SubtitleEntry(
                index=len(entries) + 1,
                start=start,
                end=end,
                lines=text_lines,
            )
        )

    logger.info("SRT loaded from: %s (%d entries)", input_path, len(entries))
    return entries


def write_srt(entries: list[SubtitleEntry], output_path: str | Path) -> Path:
    """Serialise *entries* to an SRT file at *output_path*.

    Parameters
    ----------
    entries:
        Ordered list of :class:`~autostr.reflow.SubtitleEntry` objects.
    output_path:
        Destination ``.srt`` file path.

    Returns
    -------
    Path
        Path to the written SRT file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for entry in entries:
        lines.append(str(entry.index))
        lines.append(
            f"{_format_time(entry.start)} --> {_format_time(entry.end)}"
        )
        lines.append(entry.text)
        lines.append("")  # blank line between entries

    srt_content = "\n".join(lines)
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info("SRT written to: %s (%d entries)", output_path, len(entries))
    return output_path
