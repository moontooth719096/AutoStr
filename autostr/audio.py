"""Audio extraction utilities using ffmpeg."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_audio(video_path: str | Path, output_path: str | Path | None = None) -> Path:
    """Extract mono 16 kHz WAV audio from *video_path* using ffmpeg.

    Parameters
    ----------
    video_path:
        Path to the source video (or audio) file.
    output_path:
        Destination WAV path.  If *None* a sibling file with ``.wav`` suffix
        is created next to *video_path*.

    Returns
    -------
    Path
        Path to the extracted WAV file.

    Raises
    ------
    FileNotFoundError
        If *ffmpeg* is not installed or *video_path* does not exist.
    subprocess.CalledProcessError
        If the ffmpeg invocation fails.
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg not found. Install it with: apt-get install ffmpeg"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        output_path = video_path.with_suffix(".wav")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                  # overwrite without prompting
        "-i", str(video_path),
        "-vn",                 # no video
        "-ac", "1",            # mono
        "-ar", "16000",        # 16 kHz
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    logger.info("Audio saved to: %s", output_path)
    return output_path
