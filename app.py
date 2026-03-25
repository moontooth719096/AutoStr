import argparse
import os
import subprocess
from pathlib import Path

from model_loader import load_whisper_model
from subtitle_utils import (
    split_sentence_chunks,
    wrap_chinese_text,
    srt_timestamp,
    build_srt_entries,
)

def extract_audio(input_video: str, output_wav: str):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        output_wav,
    ]
    subprocess.run(cmd, check=True)

def transcribe_audio(audio_path: str, model_name: str):
    model = load_whisper_model(model_name)
    segments, _ = model.transcribe(audio_path, language="zh", beam_size=5)

    results = []
    for seg in segments:
        results.append({
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        })
    return results

def postprocess_segments(segments, start_delay_ms=150):
    entries = []
    for seg in segments:
        text_chunks = split_sentence_chunks(seg["text"])
        if not text_chunks:
            continue

        duration = max(seg["end"] - seg["start"], 0.5)
        per_chunk = duration / len(text_chunks)

        for i, chunk in enumerate(text_chunks):
            start = seg["start"] + (i * per_chunk)
            end = seg["start"] + ((i + 1) * per_chunk)

            start += start_delay_ms / 1000.0
            if end <= start:
                end = start + 0.5

            wrapped = wrap_chinese_text(chunk)
            entries.append({
                "start": start,
                "end": end,
                "text": wrapped,
            })

    return entries

def main():
    parser = argparse.ArgumentParser(description="Dockerized Chinese subtitle alignment pipeline")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output SRT path")
    parser.add_argument("--model", default=os.getenv("WHISPER_MODEL", "small"), help="Whisper model name")
    parser.add_argument("--start-delay-ms", type=int, default=int(os.getenv("START_DELAY_MS", "150")))
    args = parser.parse_args()

    input_video = args.input
    output_srt = args.output

    workdir = Path("/tmp/autostr")
    workdir.mkdir(parents=True, exist_ok=True)
    audio_path = str(workdir / "audio.wav")

    extract_audio(input_video, audio_path)
    raw_segments = transcribe_audio(audio_path, args.model)
    processed = postprocess_segments(raw_segments, start_delay_ms=args.start_delay_ms)
    build_srt_entries(processed, output_srt)

if __name__ == "__main__":
    main()