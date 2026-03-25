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
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autostr",
        description=(
            "AutoStr – Dockerised Chinese video subtitle alignment and reflow.\n"
            "Accepts a video file and produces an SRT subtitle file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    p.add_argument("video", help="Input video (or audio) file path.")
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output SRT file path.  Defaults to <video>.srt.",
    )
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep the intermediate WAV audio file.",
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
        help="Quantisation type.  Use int8 for CPU, float16 for GPU (default: %(default)s).",
    )

    # ── Alignment ────────────────────────────────────────────────────────────
    aln = p.add_argument_group("alignment")
    aln.add_argument(
        "--no-whisperx",
        action="store_false",
        dest="use_whisperx",
        help="Disable WhisperX fine-grained alignment (use faster-whisper timestamps only).",
    )

    # ── Reflow ───────────────────────────────────────────────────────────────
    rf = p.add_argument_group("reflow / formatting")
    rf.add_argument(
        "--max-chars",
        type=int,
        default=16,
        dest="max_chars_per_line",
        help="Max Chinese characters per subtitle line (default: %(default)s).",
    )
    rf.add_argument(
        "--start-delay",
        type=int,
        default=0,
        dest="start_delay_ms",
        metavar="MS",
        help=(
            "Milliseconds to add to every subtitle start time. "
            "Set to 100–200 if subtitles appear too early (default: %(default)s)."
        ),
    )
    rf.add_argument(
        "--global-shift",
        type=int,
        default=0,
        dest="global_shift_ms",
        metavar="MS",
        help=(
            "Global timing shift in ms applied to all start and end times. "
            "Positive = later, negative = earlier (default: %(default)s)."
        ),
    )
    rf.add_argument(
        "--min-duration",
        type=float,
        default=0.8,
        dest="min_duration",
        metavar="SEC",
        help="Minimum subtitle display duration in seconds (default: %(default)s).",
    )
    rf.add_argument(
        "--max-duration",
        type=float,
        default=7.0,
        dest="max_duration",
        metavar="SEC",
        help="Maximum subtitle display duration in seconds (default: %(default)s).",
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
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

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video file not found: {video_path}", file=sys.stderr)
        return 1

    from autostr.pipeline import run

    try:
        output_srt = run(
            video_path=video_path,
            output_srt=args.output,
            model_size=args.model_size,
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
        )
    except Exception as exc:
        logging.getLogger(__name__).exception("Pipeline failed: %s", exc)
        return 1

    print(f"✔ Subtitles written to: {output_srt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
