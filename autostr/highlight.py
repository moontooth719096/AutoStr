"""Highlight detection and ffmpeg-based clip export."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
}


@dataclass(frozen=True)
class HighlightCandidate:
    start: float
    end: float
    score: float
    reason: str
    text: str


@dataclass(frozen=True)
class HighlightClip:
    index: int
    source_start: float
    source_end: float
    export_start: float
    export_end: float
    score: float
    reason: str
    text: str
    output_path: str


def _extract_text(segment) -> str:
    if isinstance(segment, dict):
        return str(segment.get("text", "")).strip()
    return str(getattr(segment, "text", "")).strip()


def _extract_start(segment) -> float:
    if isinstance(segment, dict):
        return float(segment.get("start", 0.0))
    return float(getattr(segment, "start", 0.0))


def _extract_end(segment) -> float:
    if isinstance(segment, dict):
        return float(segment.get("end", 0.0))
    return float(getattr(segment, "end", 0.0))


def _extract_words(segment) -> list | None:
    if isinstance(segment, dict):
        return segment.get("words")
    return getattr(segment, "words", None)


@lru_cache(maxsize=1)
def _ffmpeg_supports_encoder(encoder_name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == encoder_name:
            return True
    return False


def _select_video_encoder(prefer_gpu: bool) -> tuple[str, list[str]]:
    if prefer_gpu and _ffmpeg_supports_encoder("h264_nvenc"):
        return "h264_nvenc", ["-preset", "p4", "-cq:v", "23", "-b:v", "0"]

    if _ffmpeg_supports_encoder("libx264"):
        return "libx264", ["-preset", "veryfast", "-crf", "23"]

    return "mpeg4", ["-q:v", "5"]


def _segment_score(segment) -> float:
    text = _extract_text(segment)
    if not text:
        return 0.0

    duration = max(_extract_end(segment) - _extract_start(segment), 0.1)
    length_bonus = min(len(text), 120) / 12.0

    word_count = len(_extract_words(segment) or [])
    word_bonus = min(word_count, 60) / 8.0

    punctuation_bonus = 0.0
    if text.endswith(("。", "！", "？")):
        punctuation_bonus += 3.0
    elif text.endswith(("，", "、", "；")):
        punctuation_bonus += 1.2

    density_bonus = min(len(text) / duration, 40.0) / 10.0
    return length_bonus + word_bonus + punctuation_bonus + density_bonus


def _build_reason(text: str) -> str:
    reasons: list[str] = []
    stripped = text.strip()
    if stripped.endswith(("。", "！", "？")):
        reasons.append("句尾完整")
    if len(stripped) >= 24:
        reasons.append("資訊量較高")
    if len(stripped) <= 10:
        reasons.append("節奏較快")
    if not reasons:
        reasons.append("語句完整")
    return " / ".join(reasons)


def _overlaps(candidate: HighlightCandidate, accepted: list[HighlightCandidate], min_gap_seconds: float) -> bool:
    for existing in accepted:
        if candidate.end + min_gap_seconds <= existing.start:
            continue
        if candidate.start >= existing.end + min_gap_seconds:
            continue
        return True
    return False


def _normalize_segments(segments: Iterable) -> list:
    normalized = []
    for segment in segments:
        text = _extract_text(segment)
        if not text:
            continue
        start = _extract_start(segment)
        end = _extract_end(segment)
        if end <= start:
            continue
        normalized.append(segment)
    return normalized


def detect_highlights(
    segments: Iterable,
    target_count: int = 3,
    min_clip_duration: float = 15.0,
    max_clip_duration: float = 60.0,
    min_gap_seconds: float = 4.0,
) -> list[HighlightCandidate]:
    """Return the strongest non-overlapping highlight candidates from transcript segments."""
    normalized_segments = _normalize_segments(segments)
    if not normalized_segments or target_count <= 0:
        return []

    candidates: list[HighlightCandidate] = []

    for start_index in range(len(normalized_segments)):
        window_text: list[str] = []
        window_score = 0.0
        window_start = _extract_start(normalized_segments[start_index])

        for end_index in range(start_index, len(normalized_segments)):
            segment = normalized_segments[end_index]
            if end_index > start_index:
                gap = _extract_start(segment) - _extract_end(normalized_segments[end_index - 1])
                if gap > min_gap_seconds:
                    break

            window_text.append(_extract_text(segment))
            window_score += _segment_score(segment)

            window_end = _extract_end(segment)
            window_duration = window_end - window_start

            if window_duration < min_clip_duration:
                continue
            if window_duration > max_clip_duration:
                break

            combined_text = "".join(window_text).strip()
            if not combined_text:
                continue

            score = window_score / max(window_duration, 0.1)
            candidates.append(
                HighlightCandidate(
                    start=window_start,
                    end=window_end,
                    score=score,
                    reason=_build_reason(combined_text),
                    text=combined_text,
                )
            )

    selected: list[HighlightCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, item.end - item.start),
        reverse=True,
    ):
        if _overlaps(candidate, selected, min_gap_seconds=min_gap_seconds):
            continue
        selected.append(candidate)
        if len(selected) >= target_count:
            break

    return sorted(selected, key=lambda item: item.start)


def _format_seconds(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds >= 1000:
        milliseconds = 999
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _trim_clip(
    source_video: Path,
    output_path: Path,
    start_seconds: float,
    end_seconds: float,
    prefer_gpu: bool = False,
) -> None:
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("ffmpeg not found. Install it with: apt-get install ffmpeg")

    duration = max(end_seconds - start_seconds, 0.1)
    video_encoder, video_options = _select_video_encoder(prefer_gpu=prefer_gpu)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-ss",
        _format_seconds(start_seconds),
        "-t",
        _format_seconds(duration),
        "-c:v",
        video_encoder,
        *video_options,
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    logger.info("Exporting highlight clip: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            logger.error("ffmpeg failed while exporting %s: %s", output_path, stderr)
            raise RuntimeError(f"ffmpeg failed while exporting {output_path}: {stderr}") from exc
        raise RuntimeError(f"ffmpeg failed while exporting {output_path}") from exc


def export_highlight_clips(
    source_video: str | Path,
    candidates: Iterable[HighlightCandidate],
    output_dir: str | Path,
    padding_seconds: float = 1.5,
    prefer_gpu: bool = False,
) -> list[HighlightClip]:
    source_video = Path(source_video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_video.suffix.lower() not in VIDEO_SUFFIXES:
        logger.warning("Highlight clipping skipped for non-video input: %s", source_video)
        return []

    exports: list[HighlightClip] = []
    for index, candidate in enumerate(candidates, start=1):
        export_start = max(0.0, candidate.start - padding_seconds)
        export_end = max(export_start + 0.1, candidate.end + padding_seconds)
        output_path = output_dir / f"{source_video.stem}_highlight_{index:02d}.mp4"

        _trim_clip(source_video, output_path, export_start, export_end, prefer_gpu=prefer_gpu)

        exports.append(
            HighlightClip(
                index=index,
                source_start=candidate.start,
                source_end=candidate.end,
                export_start=export_start,
                export_end=export_end,
                score=candidate.score,
                reason=candidate.reason,
                text=candidate.text,
                output_path=str(output_path),
            )
        )

    manifest_path = output_dir / f"{source_video.stem}_highlights.json"
    manifest_path.write_text(
        json.dumps([asdict(item) for item in exports], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Highlight manifest written to: %s", manifest_path)

    return exports