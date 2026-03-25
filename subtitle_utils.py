import re
from pathlib import Path

def split_sentence_chunks(text: str):
    text = text.strip()
    if not text:
        return []

    # Split by Chinese punctuation while keeping semantic readability.
    parts = re.split(r'([。！？；，])', text)
    chunks = []
    buffer = ""

    for i in range(0, len(parts), 2):
        left = parts[i].strip()
        right = parts[i + 1] if i + 1 < len(parts) else ""
        if left:
            buffer += left
        if right:
            buffer += right
        if buffer:
            chunks.append(buffer.strip())
            buffer = ""

    if buffer.strip():
        chunks.append(buffer.strip())

    return [c for c in chunks if c]

def wrap_chinese_text(text: str, max_chars: int = 16):
    text = text.strip()
    if len(text) <= max_chars:
        return text

    lines = []
    current = ""
    for ch in text:
        current += ch
        if len(current) >= max_chars:
            lines.append(current)
            current = ""

    if current:
        lines.append(current)

    return "\n".join(lines[:2])

def srt_timestamp(seconds: float):
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def build_srt_entries(entries, output_path: str):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for idx, entry in enumerate(entries, start=1):
        start = srt_timestamp(entry["start"])
        end = srt_timestamp(entry["end"])
        text = entry["text"].strip()
        if not text:
            continue
        lines.append(f"{idx}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")