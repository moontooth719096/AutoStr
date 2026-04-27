"""Highlight detection and ffmpeg-based clip export."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from functools import lru_cache
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

SCORE_WEIGHTS = {
    "length": 0.14,
    "density": 0.14,
    "punctuation": 0.12,
    "completion": 0.14,
    "topic_focus": 0.12,
    "cue_phrase": 0.14,
    "pause_boundary": 0.10,
    "cohesion": 0.10,
}

SCORE_PROFILES = {
    "balanced": SCORE_WEIGHTS,
    "tutorial": {
        "length": 0.14,
        "density": 0.20,
        "punctuation": 0.08,
        "completion": 0.18,
        "topic_focus": 0.12,
        "cue_phrase": 0.16,
        "pause_boundary": 0.05,
        "cohesion": 0.07,
    },
    "entertainment": {
        "length": 0.10,
        "density": 0.12,
        "punctuation": 0.20,
        "completion": 0.16,
        "topic_focus": 0.10,
        "cue_phrase": 0.14,
        "pause_boundary": 0.10,
        "cohesion": 0.08,
    },
}

CUE_PHRASES = (
    "重點",
    "關鍵",
    "注意",
    "總結",
    "結論",
    "核心",
    "重申",
    "一定要",
    "最重要",
    "先講",
    "先說",
    "最後",
)


@dataclass(frozen=True)
class HighlightCandidate:
    start: float
    end: float
    score: float
    reason: str
    text: str
    raw_score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)


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
    raw_score: float = 0.0
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HighlightDetectionReport:
    strategy: str
    reranker: str
    weight_overrides: dict[str, float]
    target_count: int
    min_clip_duration: float
    max_clip_duration: float
    min_gap_seconds: float
    candidates: list[HighlightCandidate]
    selected: list[HighlightCandidate]

    def manifest_metadata(self) -> dict[str, object]:
        selected_keys = {_candidate_key(candidate) for candidate in self.selected}
        accepted: list[HighlightCandidate] = []
        rejected: list[dict[str, object]] = []

        for candidate in sorted(
            self.candidates,
            key=lambda item: (item.score, item.raw_score, item.end - item.start),
            reverse=True,
        ):
            if _candidate_key(candidate) in selected_keys:
                accepted.append(candidate)
                continue
            if _overlaps(candidate, accepted, min_gap_seconds=self.min_gap_seconds):
                rejection_reason = "overlap_with_higher_ranked_candidate"
            elif len(accepted) >= self.target_count:
                rejection_reason = "target_count_reached"
            else:
                rejection_reason = "not_selected"
            rejected.append(_serialize_candidate(candidate, rejection_reason=rejection_reason))

        rejection_summary = _summarize_rejections(rejected)
        score_distribution = _score_distribution(self.candidates)

        selected_duration = round(sum(candidate.end - candidate.start for candidate in self.selected), 4)
        candidate_timeline_start = min((candidate.start for candidate in self.candidates), default=0.0)
        candidate_timeline_end = max((candidate.end for candidate in self.candidates), default=0.0)
        candidate_timeline_span = round(max(candidate_timeline_end - candidate_timeline_start, 0.0), 4)
        average_candidate_score = round(
            sum(candidate.score for candidate in self.candidates) / max(len(self.candidates), 1),
            4,
        )
        average_selected_score = round(
            sum(candidate.score for candidate in self.selected) / max(len(self.selected), 1),
            4,
        )
        alternates = [
            {**candidate, "alternate_rank": index}
            for index, candidate in enumerate(rejected[:3], start=1)
        ]

        return {
            "selection": {
                "strategy": self.strategy,
                "reranker": self.reranker,
                "weight_overrides": dict(self.weight_overrides),
                "target_count": self.target_count,
                "min_clip_duration": self.min_clip_duration,
                "max_clip_duration": self.max_clip_duration,
                "min_gap_seconds": self.min_gap_seconds,
                "generated_count": len(self.candidates),
                "selected_count": len(self.selected),
                "rejected_count": len(rejected),
            },
            "metrics": {
                "candidate_timeline_start": candidate_timeline_start,
                "candidate_timeline_end": candidate_timeline_end,
                "candidate_timeline_span": candidate_timeline_span,
                "selected_total_duration": selected_duration,
                "selected_coverage_ratio": round(selected_duration / max(candidate_timeline_span, 0.1), 4),
                "average_candidate_score": average_candidate_score,
                "average_selected_score": average_selected_score,
                "score_distribution": score_distribution,
            },
            "rejections": {
                "summary": rejection_summary,
            },
            "alternates": alternates,
            "candidate_pool": {
                "selected": [_serialize_candidate(candidate) for candidate in self.selected],
                "rejected": rejected,
            },
        }


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


def _length_score(text: str, duration: float) -> float:
    if not text:
        return 0.0

    char_count = len(text.strip())
    if char_count < 12:
        return max(char_count / 12.0, 0.1)
    if char_count <= 72:
        return 1.0
    return max(0.2, 1.0 - ((char_count - 72) / 96.0))


def _density_score(text: str, duration: float) -> float:
    if not text:
        return 0.0

    chars_per_second = len(text.strip()) / max(duration, 0.1)
    target_density = 4.5
    distance = abs(chars_per_second - target_density)
    return max(0.0, 1.0 - (distance / target_density))


def _punctuation_score(text: str) -> float:
    if not text:
        return 0.0

    stripped = text.strip()
    punctuation_count = sum(1 for ch in stripped if ch in "，。、！？；：,!?;:")
    score = 0.2 + min(punctuation_count, 4) * 0.15
    if stripped.endswith(("。", "！", "？", "!", "?")):
        score += 0.2
    return min(score, 1.0)


def _completion_score(text: str) -> float:
    if not text:
        return 0.0

    stripped = text.strip()
    if stripped.endswith(("。", "！", "？", "!", "?")):
        return 1.0
    if stripped.endswith(("；", ";", "，", ",", "、", ":", "：")):
        return 0.45
    return 0.7 if len(stripped) >= 16 else 0.35


def _tokenize_for_focus(text: str) -> list[str]:
    tokens: list[str] = []
    buffer = ""
    for char in text.lower():
        if char.isalnum():
            buffer += char
            continue
        if buffer:
            tokens.append(buffer)
            buffer = ""
        if "\u4e00" <= char <= "\u9fff":
            tokens.append(char)
    if buffer:
        tokens.append(buffer)
    return tokens


def _topic_focus_score(text: str) -> float:
    tokens = _tokenize_for_focus(text)
    if not tokens:
        return 0.0

    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    repeated = sum(count for count in counts.values() if count > 1)
    unique_ratio = len(counts) / max(len(tokens), 1)
    repetition_ratio = repeated / len(tokens)
    focus = 0.55 * repetition_ratio + 0.45 * (1.0 - min(unique_ratio, 1.0))
    return max(0.0, min(1.0, focus + 0.25))


def _cue_phrase_score(text: str) -> float:
    if not text:
        return 0.0

    lowered = text.lower()
    matches = sum(1 for phrase in CUE_PHRASES if phrase in lowered)
    if matches == 0:
        return 0.0
    return min(1.0, 0.35 + matches * 0.22)


def _pause_boundary_score(before_gap: float = 0.0, after_gap: float = 0.0) -> float:
    strongest_gap = max(before_gap, after_gap, 0.0)
    if strongest_gap <= 0.0:
        return 0.0
    return round(min(1.0, strongest_gap / 1.5), 4)


def _cohesion_score(internal_gaps: list[float] | None = None, max_gap_seconds: float = 4.0) -> float:
    if not internal_gaps:
        return 1.0

    average_gap = sum(internal_gaps) / len(internal_gaps)
    normalized_gap = min(average_gap / max(max_gap_seconds, 0.1), 1.0)
    return round(max(0.0, 1.0 - normalized_gap), 4)


def _score_breakdown(
    text: str,
    duration: float,
    signal_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    signal_overrides = signal_overrides or {}
    return {
        "length": round(_length_score(text, duration), 4),
        "density": round(_density_score(text, duration), 4),
        "punctuation": round(_punctuation_score(text), 4),
        "completion": round(_completion_score(text), 4),
        "topic_focus": round(_topic_focus_score(text), 4),
        "cue_phrase": round(signal_overrides.get("cue_phrase", _cue_phrase_score(text)), 4),
        "pause_boundary": round(signal_overrides.get("pause_boundary", 0.0), 4),
        "cohesion": round(signal_overrides.get("cohesion", 1.0), 4),
    }


def _normalize_score_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in weights.values())
    if total <= 0.0:
        return dict(SCORE_WEIGHTS)
    return {
        name: round(max(value, 0.0) / total, 6)
        for name, value in weights.items()
    }


def _resolve_score_weights(
    strategy: str = "balanced",
    weight_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    weights = dict(SCORE_PROFILES.get(strategy, SCORE_WEIGHTS))
    if not weight_overrides:
        return weights

    for name, value in weight_overrides.items():
        if name not in weights:
            continue
        weights[name] = max(float(value), 0.0)
    return _normalize_score_weights(weights)


def _composite_score(
    scores: dict[str, float],
    strategy: str = "balanced",
    weight_overrides: dict[str, float] | None = None,
) -> float:
    weights = _resolve_score_weights(strategy, weight_overrides=weight_overrides)
    return round(
        sum(scores[name] * weights[name] for name in weights),
        4,
    )


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
    word_count = len(_extract_words(segment) or [])
    word_bonus = min(word_count, 60) / 60.0
    scores = _score_breakdown(text, duration)
    return _composite_score(scores) + word_bonus * 0.2


def _build_reason(text: str, scores: dict[str, float] | None = None) -> str:
    if scores is None:
        scores = _score_breakdown(text, max(len(text) / 4.5, 0.1))

    labels = {
        "length": "長度自然",
        "density": "資訊密度佳",
        "punctuation": "標點節奏明確",
        "completion": "句尾完整",
        "topic_focus": "主題集中",
        "cue_phrase": "重點提示明確",
        "pause_boundary": "停頓邊界清楚",
        "cohesion": "句群連貫",
    }
    reasons: list[str] = []
    for name, value in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        if value >= 0.7 and name in labels:
            reasons.append(labels[name])
    if not reasons:
        reasons.append("語句完整")
    return " / ".join(reasons[:3])


def _candidate_key(candidate: HighlightCandidate) -> tuple[float, float, float, float]:
    return (candidate.start, candidate.end, candidate.score, candidate.raw_score)


def _serialize_candidate(
    candidate: HighlightCandidate,
    rejection_reason: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "start": candidate.start,
        "end": candidate.end,
        "duration": round(max(candidate.end - candidate.start, 0.0), 4),
        "score": candidate.score,
        "raw_score": candidate.raw_score,
        "reason": candidate.reason,
        "text": candidate.text,
        "scores": dict(candidate.scores),
    }
    if rejection_reason is not None:
        payload["rejection_reason"] = rejection_reason
    return payload


@dataclass(frozen=True)
class HighlightRerankContext:
    reranker: str
    strategy: str
    target_count: int
    min_gap_seconds: float
    weight_overrides: dict[str, float]


def _summarize_rejections(rejected: list[dict[str, object]]) -> dict[str, int]:
    summary = {
        "overlap_with_higher_ranked_candidate": 0,
        "target_count_reached": 0,
        "not_selected": 0,
    }
    for candidate in rejected:
        rejection_reason = candidate.get("rejection_reason")
        if rejection_reason in summary:
            summary[rejection_reason] += 1
    return summary


def _score_distribution(candidates: list[HighlightCandidate]) -> dict[str, object]:
    buckets = {
        "0.0-0.2": 0,
        "0.2-0.4": 0,
        "0.4-0.6": 0,
        "0.6-0.8": 0,
        "0.8-1.0": 0,
        "1.0+": 0,
    }
    if not candidates:
        return {
            "min": 0.0,
            "max": 0.0,
            "buckets": buckets,
        }

    scores = [candidate.score for candidate in candidates]
    for score in scores:
        if score < 0.2:
            buckets["0.0-0.2"] += 1
        elif score < 0.4:
            buckets["0.2-0.4"] += 1
        elif score < 0.6:
            buckets["0.4-0.6"] += 1
        elif score < 0.8:
            buckets["0.6-0.8"] += 1
        elif score < 1.0:
            buckets["0.8-1.0"] += 1
        else:
            buckets["1.0+"] += 1
    return {
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "buckets": buckets,
    }


def _narrative_rerank_bonus(candidate: HighlightCandidate, context: HighlightRerankContext) -> float:
    scores = candidate.scores
    return round(
        scores.get("cue_phrase", 0.0) * 0.08
        + scores.get("completion", 0.0) * 0.05
        + scores.get("pause_boundary", 0.0) * 0.05
        + scores.get("cohesion", 0.0) * 0.03,
        4,
    )


def rerank_highlight_candidates(
    candidates: Iterable[HighlightCandidate],
    reranker: str = "none",
    strategy: str = "balanced",
    target_count: int = 3,
    min_gap_seconds: float = 4.0,
    weight_overrides: dict[str, float] | None = None,
) -> list[HighlightCandidate]:
    """Apply an optional reranking pass to scored candidates before final selection."""
    candidate_list = list(candidates)
    context = HighlightRerankContext(
        reranker=reranker,
        strategy=strategy,
        target_count=target_count,
        min_gap_seconds=min_gap_seconds,
        weight_overrides=dict(weight_overrides or {}),
    )

    if reranker == "none":
        return [
            HighlightCandidate(
                start=candidate.start,
                end=candidate.end,
                score=candidate.score,
                raw_score=candidate.raw_score,
                reason=candidate.reason,
                text=candidate.text,
                scores={**candidate.scores, "reranker": reranker, "rerank_bonus": 0.0},
            )
            for candidate in candidate_list
        ]

    if reranker != "narrative":
        raise ValueError(f"Unsupported reranker: {reranker}")

    reranked: list[HighlightCandidate] = []
    for candidate in candidate_list:
        rerank_bonus = _narrative_rerank_bonus(candidate, context)
        reason = candidate.reason
        if rerank_bonus > 0.0 and "敘事節點" not in reason:
            reason = f"{reason} / 敘事節點"
        reranked.append(
            HighlightCandidate(
                start=candidate.start,
                end=candidate.end,
                score=round(candidate.score + rerank_bonus, 4),
                raw_score=candidate.raw_score,
                reason=reason,
                text=candidate.text,
                scores={**candidate.scores, "reranker": reranker, "rerank_bonus": rerank_bonus},
            )
        )
    return reranked


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


def generate_highlight_candidates(
    segments: Iterable,
    min_clip_duration: float = 15.0,
    max_clip_duration: float = 60.0,
    max_segment_gap_seconds: float = 4.0,
) -> list[HighlightCandidate]:
    """Build a broad candidate pool before scoring and selection."""
    normalized_segments = _normalize_segments(segments)
    if not normalized_segments:
        return []

    candidates: list[HighlightCandidate] = []

    for start_index in range(len(normalized_segments)):
        window_text: list[str] = []
        window_score = 0.0
        internal_gaps: list[float] = []
        window_start = _extract_start(normalized_segments[start_index])
        before_gap = 0.0
        if start_index > 0:
            before_gap = max(
                0.0,
                window_start - _extract_end(normalized_segments[start_index - 1]),
            )

        for end_index in range(start_index, len(normalized_segments)):
            segment = normalized_segments[end_index]
            if end_index > start_index:
                gap = _extract_start(segment) - _extract_end(normalized_segments[end_index - 1])
                if gap > max_segment_gap_seconds:
                    break
                internal_gaps.append(max(gap, 0.0))

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

            after_gap = 0.0
            if end_index + 1 < len(normalized_segments):
                after_gap = max(
                    0.0,
                    _extract_start(normalized_segments[end_index + 1]) - window_end,
                )

            signal_scores = {
                "pause_boundary": _pause_boundary_score(before_gap=before_gap, after_gap=after_gap),
                "cohesion": _cohesion_score(internal_gaps, max_gap_seconds=max_segment_gap_seconds),
            }

            raw_score = round(window_score / max(window_duration, 0.1), 4)
            candidates.append(
                HighlightCandidate(
                    start=window_start,
                    end=window_end,
                    score=raw_score,
                    raw_score=raw_score,
                    reason="待評分",
                    text=combined_text,
                    scores=signal_scores,
                )
            )

    return candidates


def score_highlight_candidates(candidates: Iterable[HighlightCandidate]) -> list[HighlightCandidate]:
    """Assign explainable sub-scores and a composite score to each candidate."""
    scored: list[HighlightCandidate] = []
    for candidate in candidates:
        duration = max(candidate.end - candidate.start, 0.1)
        scores = _score_breakdown(candidate.text, duration, signal_overrides=candidate.scores)
        score = _composite_score(scores)
        scored.append(
            HighlightCandidate(
                start=candidate.start,
                end=candidate.end,
                score=score,
                raw_score=candidate.raw_score,
                reason=_build_reason(candidate.text, scores),
                text=candidate.text,
                scores=scores,
            )
        )
    return scored


def score_highlight_candidates_with_strategy(
    candidates: Iterable[HighlightCandidate],
    strategy: str = "balanced",
    weight_overrides: dict[str, float] | None = None,
) -> list[HighlightCandidate]:
    """Assign explainable sub-scores and a composite score using a named strategy."""
    scored: list[HighlightCandidate] = []
    resolved_weights = _resolve_score_weights(strategy, weight_overrides=weight_overrides)
    for candidate in candidates:
        duration = max(candidate.end - candidate.start, 0.1)
        scores = _score_breakdown(candidate.text, duration, signal_overrides=candidate.scores)
        score = _composite_score(scores, strategy=strategy, weight_overrides=weight_overrides)
        scored.append(
            HighlightCandidate(
                start=candidate.start,
                end=candidate.end,
                score=score,
                raw_score=candidate.raw_score,
                reason=_build_reason(candidate.text, scores),
                text=candidate.text,
                scores={**scores, "strategy": strategy, "weights": resolved_weights},
            )
        )
    return scored


def select_highlights(
    candidates: Iterable[HighlightCandidate],
    target_count: int = 3,
    min_gap_seconds: float = 4.0,
) -> list[HighlightCandidate]:
    """Return the strongest non-overlapping candidates from a scored pool."""
    if target_count <= 0:
        return []

    selected: list[HighlightCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (item.score, item.raw_score, item.end - item.start),
        reverse=True,
    ):
        if _overlaps(candidate, selected, min_gap_seconds=min_gap_seconds):
            continue
        selected.append(candidate)
        if len(selected) >= target_count:
            break

    return sorted(selected, key=lambda item: item.start)


def detect_highlights(
    segments: Iterable,
    target_count: int = 3,
    min_clip_duration: float = 15.0,
    max_clip_duration: float = 60.0,
    min_gap_seconds: float = 4.0,
    strategy: str = "balanced",
    reranker: str = "none",
    weight_overrides: dict[str, float] | None = None,
) -> list[HighlightCandidate]:
    """Return the strongest non-overlapping highlight candidates from transcript segments."""
    return analyze_highlights(
        segments,
        target_count=target_count,
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
        min_gap_seconds=min_gap_seconds,
        strategy=strategy,
        reranker=reranker,
        weight_overrides=weight_overrides,
    ).selected


def analyze_highlights(
    segments: Iterable,
    target_count: int = 3,
    min_clip_duration: float = 15.0,
    max_clip_duration: float = 60.0,
    min_gap_seconds: float = 4.0,
    strategy: str = "balanced",
    reranker: str = "none",
    weight_overrides: dict[str, float] | None = None,
) -> HighlightDetectionReport:
    """Return selected highlights plus the scored candidate pool used to choose them."""
    candidates = generate_highlight_candidates(
        segments,
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
        max_segment_gap_seconds=min_gap_seconds,
    )
    if not candidates:
        return HighlightDetectionReport(
            strategy=strategy,
            reranker=reranker,
            weight_overrides=dict(weight_overrides or {}),
            target_count=target_count,
            min_clip_duration=min_clip_duration,
            max_clip_duration=max_clip_duration,
            min_gap_seconds=min_gap_seconds,
            candidates=[],
            selected=[],
        )

    scored_candidates = score_highlight_candidates_with_strategy(
        candidates,
        strategy=strategy,
        weight_overrides=weight_overrides,
    )
    reranked_candidates = rerank_highlight_candidates(
        scored_candidates,
        reranker=reranker,
        strategy=strategy,
        target_count=target_count,
        min_gap_seconds=min_gap_seconds,
        weight_overrides=weight_overrides,
    )
    selected = select_highlights(
        reranked_candidates,
        target_count=target_count,
        min_gap_seconds=min_gap_seconds,
    )
    return HighlightDetectionReport(
        strategy=strategy,
        reranker=reranker,
        weight_overrides=dict(weight_overrides or {}),
        target_count=target_count,
        min_clip_duration=min_clip_duration,
        max_clip_duration=max_clip_duration,
        min_gap_seconds=min_gap_seconds,
        candidates=reranked_candidates,
        selected=selected,
    )


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
    manifest_metadata: dict[str, object] | None = None,
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
                raw_score=getattr(candidate, "raw_score", candidate.score),
                scores=dict(getattr(candidate, "scores", {})),
            )
        )

    manifest_path = output_dir / f"{source_video.stem}_highlights.json"
    manifest_payload: dict[str, object] = {
        "source_video": str(source_video),
        "padding_seconds": padding_seconds,
        "prefer_gpu": prefer_gpu,
        "exported_count": len(exports),
        "clips": [asdict(item) for item in exports],
    }
    if manifest_metadata:
        manifest_payload.update(manifest_metadata)
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Highlight manifest written to: %s", manifest_path)

    return exports