#!/usr/bin/env python3
"""AutoStr – CLI entrypoint.

Usage examples
--------------
Basic (CPU, medium model)::

    python main.py input.mp4

With GPU and a larger model::

    python main.py input.mp4 --device cuda --model large-v2

Fix subtitles that appear too early::

    python main.py input.mp4 --start-delay 150

Apply a global timing shift (shift everything 500 ms later)::

    python main.py input.mp4 --global-shift 500

Specify output path and control line length::

    python main.py input.mp4 -o my_subtitles.srt --max-chars 18

Scan an input folder for missing subtitles::

    python main.py --batch

Use a custom local Whisper model cache::

    python main.py input.mp4 --model-dir D:/models
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _build_highlight_weight_overrides(args: argparse.Namespace) -> dict[str, float] | None:
    overrides: dict[str, float] = {}
    if args.highlight_cue_weight is not None:
        overrides["cue_phrase"] = args.highlight_cue_weight
    if args.highlight_pause_weight is not None:
        overrides["pause_boundary"] = args.highlight_pause_weight
    return overrides or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autostr",
        description=(
            "AutoStr – Dockerised Chinese video subtitle alignment and reflow.\n"
            "Accepts either one video file or a batch input/output folder pair."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("video", nargs="?", help="Input video (or audio) file path.")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output SRT file path. Defaults to /output/<video>.srt inside the container.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Scan an input folder and process files whose matching SRT is missing.",
    )
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the intermediate WAV audio file.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Whisper model cache directory inside the container or on the host-mounted path.",
    )

    # ── Highlights ──────────────────────────────────────────────────────────
    hl = p.add_argument_group("highlights / clipping")
    hl.add_argument(
        "--highlights",
        action="store_true",
        help="Detect and export automatic highlight clips.",
    )
    hl.add_argument(
        "--highlight-output-dir",
        default=None,
        help="Directory for highlight clips. Defaults to the same folder as the SRT output; in highlight mode that is /output/<video>_highlights.",
    )
    hl.add_argument(
        "--highlight-count",
        type=int,
        default=3,
        help="Maximum number of highlight clips to export.",
    )
    hl.add_argument(
        "--highlight-min-duration",
        type=float,
        default=15.0,
        help="Minimum length of a highlight clip in seconds.",
    )
    hl.add_argument(
        "--highlight-max-duration",
        type=float,
        default=60.0,
        help="Maximum length of a highlight clip in seconds.",
    )
    hl.add_argument(
        "--highlight-min-gap",
        type=float,
        default=4.0,
        dest="highlight_min_gap_seconds",
        help="Minimum gap in seconds required between selected highlights.",
    )
    hl.add_argument(
        "--highlight-padding",
        type=float,
        default=1.5,
        dest="highlight_padding_seconds",
        help="Seconds to add before and after each highlight clip.",
    )
    hl.add_argument(
        "--highlight-strategy",
        default="balanced",
        choices=["balanced", "tutorial", "entertainment"],
        help="Named highlight scoring profile.",
    )
    hl.add_argument(
        "--highlight-reranker",
        default="none",
        choices=["none", "narrative"],
        help="Optional reranker hook to reorder scored highlight candidates.",
    )
    hl.add_argument(
        "--highlight-cue-weight",
        type=float,
        default=None,
        help="Override the cue_phrase scoring weight for highlights.",
    )
    hl.add_argument(
        "--highlight-pause-weight",
        type=float,
        default=None,
        help="Override the pause_boundary scoring weight for highlights.",
    )
    hl.add_argument(
        "--highlight-encoder",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Highlight export encoder preference. Use auto to follow --device.",
    )

    # ── ASR ──────────────────────────────────────────────────────────────────
    asr = p.add_argument_group("ASR / transcription")
    asr.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        dest="model_size",
        help="faster-whisper model size (default: %(default)s).",
    )
    asr.add_argument(
        "--language",
        default="zh",
        help="BCP-47 language code (default: %(default)s for Mandarin).",
    )
    asr.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device (default: %(default)s).",
    )
    asr.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        dest="compute_type",
        help="Quantisation type. Use int8 for CPU, float16 for GPU.",
    )

    # ── Alignment ────────────────────────────────────────────────────────────
    aln = p.add_argument_group("alignment")
    aln.add_argument(
        "--no-whisperx",
        action="store_false",
        dest="use_whisperx",
        help="Disable WhisperX fine-grained alignment.",
    )

    # ── Reflow ───────────────────────────────────────────────────────────────
    rf = p.add_argument_group("reflow / formatting")
    rf.add_argument(
        "--max-chars",
        type=int,
        default=16,
        dest="max_chars_per_line",
        help="Max Chinese characters per subtitle line.",
    )
    rf.add_argument(
        "--start-delay",
        type=int,
        default=0,
        dest="start_delay_ms",
        metavar="MS",
        help="Milliseconds to add to every subtitle start time.",
    )
    rf.add_argument(
        "--global-shift",
        type=int,
        default=0,
        dest="global_shift_ms",
        metavar="MS",
        help="Shift all subtitle times by N ms.",
    )
    rf.add_argument(
        "--min-duration",
        type=float,
        default=0.8,
        dest="min_duration",
        help="Minimum subtitle display duration.",
    )
    rf.add_argument(
        "--max-duration",
        type=float,
        default=7.0,
        dest="max_duration",
        help="Maximum subtitle display duration.",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG logging.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from autostr.pipeline import run, run_missing_subtitles
    highlight_weight_overrides = _build_highlight_weight_overrides(args)

    try:
        if args.batch:
            if args.video is not None or args.output is not None:
                print("ERROR: --batch cannot be combined with a single video input or --output.", file=sys.stderr)
                return 1

            input_dir = Path("/input")
            output_dir = Path("/output")
            if not input_dir.exists() or not input_dir.is_dir():
                print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
                return 1
            if not output_dir.exists() or not output_dir.is_dir():
                print(f"ERROR: Output directory not found: {output_dir}", file=sys.stderr)
                return 1

            print(f"Starting AutoStr batch scan... input={input_dir} output={output_dir}", flush=True)
            pending_jobs = run_missing_subtitles(
                input_dir=input_dir,
                output_dir=output_dir,
                model_size=args.model_size,
                model_dir=args.model_dir,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                use_whisperx=args.use_whisperx,
                max_chars_per_line=args.max_chars_per_line,
                start_delay_ms=args.start_delay_ms,
                global_shift_ms=args.global_shift_ms,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
                keep_audio=args.keep_audio,
                export_highlights=args.highlights,
                highlight_output_dir=args.highlight_output_dir,
                highlight_count=args.highlight_count,
                highlight_min_duration=args.highlight_min_duration,
                highlight_max_duration=args.highlight_max_duration,
                highlight_min_gap_seconds=args.highlight_min_gap_seconds,
                highlight_padding_seconds=args.highlight_padding_seconds,
                highlight_strategy=args.highlight_strategy,
                highlight_reranker=args.highlight_reranker,
                highlight_weight_overrides=highlight_weight_overrides,
                highlight_encoder=args.highlight_encoder,
            )

            print(f"✔ Batch processing complete. Generated {len(pending_jobs)} subtitle file(s).")
            return 0

        if args.video is None:
            print("ERROR: a video file path is required unless --batch is used.", file=sys.stderr)
            return 1

        video_path = Path(args.video)
        if not video_path.exists():
            print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
            return 1

        print("Starting AutoStr pipeline...", flush=True)
        output_srt = run(
            video_path=video_path,
            output_srt=args.output,
            model_size=args.model_size,
            model_dir=args.model_dir,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            use_whisperx=args.use_whisperx,
            max_chars_per_line=args.max_chars_per_line,
            start_delay_ms=args.start_delay_ms,
            global_shift_ms=args.global_shift_ms,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            keep_audio=args.keep_audio,
            export_highlights=args.highlights,
            highlight_output_dir=args.highlight_output_dir,
            highlight_count=args.highlight_count,
            highlight_min_duration=args.highlight_min_duration,
            highlight_max_duration=args.highlight_max_duration,
            highlight_min_gap_seconds=args.highlight_min_gap_seconds,
            highlight_padding_seconds=args.highlight_padding_seconds,
            highlight_strategy=args.highlight_strategy,
            highlight_reranker=args.highlight_reranker,
            highlight_weight_overrides=highlight_weight_overrides,
            highlight_encoder=args.highlight_encoder,
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        return 1

    print(f"✔ Subtitles written to: {output_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())