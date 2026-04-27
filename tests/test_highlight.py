"""Tests for automatic highlight detection and clip export."""
from __future__ import annotations

import json
from subprocess import CompletedProcess
from unittest.mock import patch

import autostr.highlight as highlight_module
from autostr.highlight import (
    HighlightDetectionReport,
    HighlightCandidate,
    analyze_highlights,
    detect_highlights,
    export_highlight_clips,
    generate_highlight_candidates,
    rerank_highlight_candidates,
    score_highlight_candidates,
    score_highlight_candidates_with_strategy,
    select_highlights,
)
from autostr.transcribe import TranscriptSegment


def _make_segment(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


def test_generate_highlight_candidates_builds_candidate_pool():
    segments = [
        _make_segment(0.0, 4.0, "開場先講今天的核心結論。"),
        _make_segment(4.1, 9.0, "接著補上判斷依據與操作細節。"),
        _make_segment(9.2, 15.8, "這一段把最重要的風險說清楚。"),
    ]

    candidates = generate_highlight_candidates(
        segments,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
    )

    assert candidates
    assert candidates[0].start == 0.0
    assert candidates[0].end >= 9.0
    assert candidates[0].raw_score > 0.0
    assert "pause_boundary" in candidates[0].scores
    assert "cohesion" in candidates[0].scores


def test_score_highlight_candidates_exposes_subscores():
    candidates = [
        HighlightCandidate(
            start=0.0,
            end=12.0,
            score=0.6,
            raw_score=0.6,
            reason="待評分",
            text="這一段先講明確結論。接著補上原因。最後收在完整句尾，這就是今天的重點。",
            scores={"pause_boundary": 0.8, "cohesion": 0.95},
        )
    ]

    scored = score_highlight_candidates(candidates)

    assert len(scored) == 1
    assert scored[0].scores.keys() == {
        "length",
        "density",
        "punctuation",
        "completion",
        "topic_focus",
        "cue_phrase",
        "pause_boundary",
        "cohesion",
    }
    assert scored[0].scores["completion"] > 0.7
    assert scored[0].scores["cue_phrase"] > 0.3
    assert scored[0].scores["pause_boundary"] == 0.8
    assert scored[0].scores["cohesion"] == 0.95
    assert scored[0].score > 0.0
    assert "句尾完整" in scored[0].reason or "重點提示明確" in scored[0].reason or "語句完整" in scored[0].reason


def test_score_highlight_candidates_with_strategy_marks_profile():
    candidates = [
        HighlightCandidate(
            start=0.0,
            end=12.0,
            score=0.6,
            raw_score=0.6,
            reason="待評分",
            text="這一段先講明確結論。接著補上原因。最後收在完整句尾。",
            scores={"pause_boundary": 0.7, "cohesion": 0.9},
        )
    ]

    scored = score_highlight_candidates_with_strategy(candidates, strategy="tutorial")

    assert scored[0].scores["strategy"] == "tutorial"
    assert scored[0].scores["pause_boundary"] == 0.7
    assert scored[0].scores["cohesion"] == 0.9


def test_score_highlight_candidates_with_strategy_applies_weight_overrides():
    candidates = [
        HighlightCandidate(
            start=0.0,
            end=12.0,
            score=0.6,
            raw_score=0.6,
            reason="待評分",
            text="先講結論，這裡是最重要的重點。請注意後面的關鍵。",
            scores={"pause_boundary": 0.9, "cohesion": 0.8},
        )
    ]

    scored = score_highlight_candidates_with_strategy(
        candidates,
        strategy="balanced",
        weight_overrides={"cue_phrase": 0.3, "pause_boundary": 0.2},
    )

    assert scored[0].scores["weights"]["cue_phrase"] > scored[0].scores["weights"]["length"]
    assert scored[0].scores["weights"]["pause_boundary"] > scored[0].scores["weights"]["cohesion"]


def test_rerank_highlight_candidates_adds_narrative_bonus():
    candidates = [
        HighlightCandidate(
            start=0.0,
            end=12.0,
            score=0.7,
            raw_score=0.6,
            reason="句尾完整",
            text="先講結論，這裡就是今天最重要的重點。請注意後面的關鍵。",
            scores={
                "cue_phrase": 0.9,
                "completion": 1.0,
                "pause_boundary": 0.8,
                "cohesion": 0.9,
            },
        )
    ]

    reranked = rerank_highlight_candidates(candidates, reranker="narrative")

    assert reranked[0].score > candidates[0].score
    assert reranked[0].scores["reranker"] == "narrative"
    assert reranked[0].scores["rerank_bonus"] > 0.0
    assert "敘事節點" in reranked[0].reason


def test_select_highlights_prefers_higher_scores_without_overlap():
    candidates = [
        HighlightCandidate(0.0, 10.0, 0.9, "句尾完整", "第一段。", raw_score=0.7),
        HighlightCandidate(8.0, 18.0, 0.8, "句尾完整", "第二段。", raw_score=0.9),
        HighlightCandidate(25.0, 35.0, 0.85, "主題集中", "第三段。", raw_score=0.6),
    ]

    selected = select_highlights(candidates, target_count=2, min_gap_seconds=4.0)

    assert len(selected) == 2
    assert selected[0].start == 0.0
    assert selected[1].start == 25.0


def test_detect_highlights_returns_non_overlapping_candidates():
    segments = [
        _make_segment(0.0, 4.0, "開場介紹一下今天的主題。"),
        _make_segment(4.2, 9.0, "這一段先講結論，再補充細節。"),
        _make_segment(20.0, 26.0, "現在進入最重要的部分。"),
        _make_segment(26.3, 33.0, "這裡有明顯的轉折點。"),
        _make_segment(50.0, 58.0, "最後總結今天的重點。"),
    ]

    candidates = detect_highlights(
        segments,
        target_count=2,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
        strategy="entertainment",
    )

    assert len(candidates) == 2
    assert candidates[0].start < candidates[0].end
    assert candidates[1].start < candidates[1].end
    assert candidates[0].scores
    assert candidates[0].scores["strategy"] == "entertainment"
    assert "cue_phrase" in candidates[0].scores
    assert "pause_boundary" in candidates[0].scores
    assert "cohesion" in candidates[0].scores
    assert candidates[0].end <= candidates[1].start or candidates[1].end <= candidates[0].start


def test_detect_highlights_prefers_pause_and_cue_signals():
    segments = [
        _make_segment(0.0, 4.0, "今天先暖場一下。"),
        _make_segment(5.6, 11.5, "先講結論，這一段是最重要的重點。"),
        _make_segment(11.7, 17.6, "請注意，後面這個做法才是關鍵。"),
        _make_segment(25.0, 31.0, "一般說明補充。"),
    ]

    candidates = detect_highlights(
        segments,
        target_count=1,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
        strategy="tutorial",
    )

    assert len(candidates) == 1
    assert candidates[0].scores["cue_phrase"] > 0.5
    assert candidates[0].scores["pause_boundary"] > 0.3


def test_analyze_highlights_exposes_candidate_pool_metadata():
    segments = [
        _make_segment(0.0, 4.0, "開場先講今天的主題。"),
        _make_segment(4.2, 9.0, "這一段先講結論，再補充細節。"),
        _make_segment(9.4, 15.5, "這裡再把風險與重點完整說清楚。"),
    ]

    report = analyze_highlights(
        segments,
        target_count=1,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
        min_gap_seconds=4.0,
        strategy="tutorial",
    )

    metadata = report.manifest_metadata()

    assert metadata["selection"]["strategy"] == "tutorial"
    assert metadata["selection"]["reranker"] == "none"
    assert metadata["selection"]["weight_overrides"] == {}
    assert metadata["selection"]["generated_count"] >= 2
    assert metadata["selection"]["selected_count"] == 1
    assert metadata["metrics"]["candidate_timeline_span"] > 0.0
    assert metadata["metrics"]["selected_total_duration"] > 0.0
    assert metadata["metrics"]["selected_coverage_ratio"] > 0.0
    assert metadata["metrics"]["score_distribution"]["max"] >= metadata["metrics"]["score_distribution"]["min"]
    assert metadata["candidate_pool"]["selected"]
    assert metadata["candidate_pool"]["rejected"]
    assert metadata["rejections"]["summary"]["target_count_reached"] >= 1
    assert metadata["alternates"]
    assert metadata["alternates"][0]["alternate_rank"] == 1


def test_analyze_highlights_reports_weight_overrides():
    segments = [
        _make_segment(0.0, 4.0, "前面先鋪陳。"),
        _make_segment(4.3, 10.5, "先講結論，這裡就是今天最重要的重點。"),
        _make_segment(10.8, 17.0, "請注意，接著說明真正的關鍵做法。"),
    ]

    report = analyze_highlights(
        segments,
        target_count=1,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
        strategy="balanced",
        reranker="narrative",
        weight_overrides={"cue_phrase": 0.28, "pause_boundary": 0.16},
    )

    metadata = report.manifest_metadata()

    assert metadata["selection"]["weight_overrides"] == {
        "cue_phrase": 0.28,
        "pause_boundary": 0.16,
    }
    assert metadata["selection"]["reranker"] == "narrative"
    assert report.selected[0].scores["weights"]["cue_phrase"] > report.selected[0].scores["weights"]["density"]
    assert metadata["metrics"]["score_distribution"]["buckets"]


def test_manifest_metadata_summarizes_overlap_rejections():
    selected = HighlightCandidate(
        start=0.0,
        end=10.0,
        score=0.92,
        raw_score=0.8,
        reason="句尾完整",
        text="第一段重點。",
    )
    overlapping = HighlightCandidate(
        start=8.0,
        end=18.0,
        score=0.9,
        raw_score=0.7,
        reason="主題集中",
        text="第二段重點。",
    )
    distant = HighlightCandidate(
        start=28.0,
        end=36.0,
        score=0.78,
        raw_score=0.6,
        reason="資訊密度佳",
        text="第三段補充。",
    )

    report = HighlightDetectionReport(
        strategy="balanced",
        weight_overrides={},
        target_count=1,
        min_clip_duration=6.0,
        max_clip_duration=20.0,
        min_gap_seconds=4.0,
        candidates=[selected, overlapping, distant],
        selected=[selected],
    )

    metadata = report.manifest_metadata()

    assert metadata["rejections"]["summary"]["overlap_with_higher_ranked_candidate"] == 1
    assert metadata["rejections"]["summary"]["target_count_reached"] == 1
    assert metadata["metrics"]["score_distribution"]["buckets"]["0.8-1.0"] == 2


def test_detect_highlights_skips_too_short_segments():
    segments = [
        _make_segment(0.0, 1.0, "短句。"),
        _make_segment(1.1, 2.0, "再短。"),
    ]

    assert detect_highlights(segments, min_clip_duration=10.0) == []


def test_export_highlight_clips_writes_manifest(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        HighlightCandidate(
            start=1.0,
            end=6.0,
            score=1.5,
            raw_score=1.2,
            reason="句尾完整",
            text="測試高光",
            scores={"completion": 1.0},
        )
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D libx264\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        exports = export_highlight_clips(
            source,
            candidates,
            output_dir,
            padding_seconds=1.0,
            manifest_metadata={
                "selection": {"strategy": "balanced", "selected_count": 1},
                "metrics": {"selected_total_duration": 5.0, "score_distribution": {"min": 1.5, "max": 1.5, "buckets": {}}},
                "alternates": [{"alternate_rank": 1}],
            },
        )

    assert len(exports) == 1
    assert exports[0].output_path.endswith("input_highlight_01.mp4")
    assert exports[0].scores == {"completion": 1.0}
    manifest = json.loads((output_dir / "input_highlights.json").read_text(encoding="utf-8"))
    assert manifest["exported_count"] == 1
    assert manifest["selection"]["strategy"] == "balanced"
    assert manifest["metrics"]["selected_total_duration"] == 5.0
    assert manifest["alternates"][0]["alternate_rank"] == 1
    assert manifest["clips"][0]["scores"] == {"completion": 1.0}
    assert mock_run.call_count == 2


def test_export_highlight_clips_falls_back_when_libx264_is_missing(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        HighlightCandidate(
            start=1.0,
            end=6.0,
            score=1.5,
            reason="句尾完整",
            text="測試高光",
        )
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D mpeg4\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        export_highlight_clips(source, candidates, output_dir, padding_seconds=1.0)

    export_cmd = mock_run.call_args_list[1].args[0]
    assert "mpeg4" in export_cmd
    assert "-q:v" in export_cmd


def test_export_highlight_clips_prefers_gpu_encoder_when_available(tmp_path):
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    output_dir = tmp_path / "clips"

    candidates = [
        HighlightCandidate(
            start=1.0,
            end=6.0,
            score=1.5,
            reason="句尾完整",
            text="測試高光",
        )
    ]

    def fake_run(cmd, check=True, capture_output=True, text=True):
        if cmd[:3] == ["ffmpeg", "-hide_banner", "-encoders"]:
            return CompletedProcess(cmd, 0, stdout=" V....D h264_nvenc\n", stderr="")
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    highlight_module._ffmpeg_supports_encoder.cache_clear()
    with patch("autostr.highlight.shutil.which", return_value="ffmpeg"), patch(
        "autostr.highlight.subprocess.run",
        side_effect=fake_run,
    ) as mock_run:
        export_highlight_clips(source, candidates, output_dir, padding_seconds=1.0, prefer_gpu=True)

    export_cmd = mock_run.call_args_list[1].args[0]
    assert "h264_nvenc" in export_cmd
    assert "-cq:v" in export_cmd