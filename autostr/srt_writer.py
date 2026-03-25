"""SRT file serialisation."""

from __future__ import annotations

import logging
from pathlib import Path

from autostr.reflow import SubtitleEntry

logger = logging.getLogger(__name__)


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
