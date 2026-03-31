# AutoStr

**Dockerised end-to-end Chinese video subtitle alignment and reflow.**

AutoStr takes a video file and produces a clean, well-formatted SRT subtitle
file in Chinese.  The pipeline:

1. **Extracts audio** (mono 16 kHz WAV) with `ffmpeg`.
2. **Transcribes speech** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
3. **Refines word-level timing** with [WhisperX](https://github.com/m-bain/whisperX) (optional, gracefully skipped if not installed).
4. **Segments and reflows** text using Chinese punctuation, pause-aware heuristics, configurable line-length limits, automatic 1–2-line wrapping, and optional timing adjustments.
5. **Outputs an SRT file** ready for video players, editors, and further post-processing.

---

## Quick start with Docker

### 1 – Build the image

```bash
docker build -t autostr .
```

The `medium` faster-whisper model is pre-downloaded during the build so the
first run is faster.  To pre-download a different model:

```bash
docker build --build-arg WHISPER_MODEL=large-v2 -t autostr:large .
```

Model files are cached under `/models` in the container by default.  Keep the
container path fixed as `/models`; only the host-side folder changes.

If you already have a local model cache, mount it to `/models` and you do not
need to pass `--model-dir`.

### CPU / GPU layouts

Use the default compose file for CPU runs:

```bash
docker compose up --build
```

Use the GPU override compose file when you want CUDA:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The same Python codebase is reused in both cases. Only the container base image,
device, and compute type change.

### 2 – Run the pipeline

```bash
# Basic usage – subtitles written to /output/input.srt
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4

# Custom output path
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 -o /output/subtitles.srt

# Use a local model cache instead of downloading again
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4

# Fix subtitles that appear too early (add 150 ms delay)
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --start-delay 150

# Shift everything 500 ms later (e.g. entire track is off-sync)
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --global-shift 500

# Use a larger model for better accuracy
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --model large-v2

# Verbose output
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 -v

# Batch scan: process every video in /input that does not yet have a matching
# .srt file in /output
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/input:/input \
  -v /path/to/your/output:/output \
  autostr --batch

# Batch scan and export highlight clips alongside subtitles
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/input:/input \
  -v /path/to/your/output:/output \
  autostr --batch --highlights
```

### 3 – Using docker compose

```bash
# Create input/output directories and put your files there
mkdir -p input output
cp /path/to/input.mp4 input/test.mp4

# Run
docker compose run --rm autostr

# With options
docker compose run --rm autostr \
  /input/test.mp4 -o /output/test.srt --start-delay 150 --max-chars 18 --model large-v2

# Batch scan the mounted input folder and fill in missing subtitles only
docker compose run --rm autostr \
  --batch
```

The compose service mounts `./input` to `/input`, `./output` to `/output`, and
`./models` to `/models`, so the container-side paths stay fixed.
If you want different host folders, change only the left-hand side of each
mount.

---

## GPU support

### CUDA build

```bash
docker build \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime \
  -t autostr:cuda .
```

### CUDA run

```bash
docker run --rm --gpus all \
  -v /path/to/models:/models \
  -v /path/to/videos:/input \
  -v /path/to/output:/output \
  autostr:cuda /input/input.mp4 --device cuda --compute-type float16
```

### Compose files

```bash
# CPU default
docker compose up --build

# GPU override
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

---

## All CLI options

```
usage: autostr [-h] [-o OUTPUT] [--batch] [--keep-audio]
               [--model {tiny,base,small,medium,large-v2,large-v3}]
               [--language LANGUAGE] [--device {cpu,cuda}]
               [--compute-type {int8,float16,float32}]
               [--no-whisperx]
               [--highlights] [--highlight-output-dir HIGHLIGHT_OUTPUT_DIR]
               [--highlight-count HIGHLIGHT_COUNT]
               [--highlight-min-duration HIGHLIGHT_MIN_DURATION]
               [--highlight-max-duration HIGHLIGHT_MAX_DURATION]
               [--highlight-padding HIGHLIGHT_PADDING_SECONDS]
               [--max-chars MAX_CHARS_PER_LINE]
               [--start-delay MS] [--global-shift MS]
               [--min-duration SEC] [--max-duration SEC]
               [-v]
               [video]

positional arguments:
  video                Input video (or audio) file path.

I/O:
  -o, --output         Output SRT file path.  Defaults to /output/<video>.srt.
  --batch              Scan an input folder and process files whose matching
                       SRT is missing.
  --keep-audio         Keep the intermediate WAV audio file.
  --model-dir          Whisper model cache directory.

ASR / transcription:
  --model              faster-whisper model size (default: medium).
  --language           BCP-47 language code (default: zh).
  --device             Inference device: cpu or cuda (default: cpu).
  --compute-type       Quantisation type: int8 (CPU) or float16 (GPU).

alignment:
  --no-whisperx        Disable WhisperX and use faster-whisper timestamps only.

highlights / clipping:
  --highlights              Detect and export automatic highlight clips.
  --highlight-output-dir     Directory for clip output. Defaults to the same
                            folder as the SRT output; in highlight mode that
                            becomes /output/<video>_highlights.
  --highlight-count         Maximum number of clips to export.
  --highlight-min-duration  Minimum length of a clip in seconds.
  --highlight-max-duration  Maximum length of a clip in seconds.
  --highlight-padding       Seconds to add before and after each clip.

reflow / formatting:
  --max-chars          Max Chinese characters per subtitle line (default: 16).
  --start-delay MS     Add N ms to every subtitle start time (default: 0).
  --global-shift MS    Shift all start+end times by N ms (default: 0).
  --min-duration SEC   Minimum display duration in seconds (default: 0.8).
  --max-duration SEC   Maximum display duration in seconds (default: 7.0).

other:
  -v, --verbose        Enable DEBUG logging.
```

---

## Tuning for Chinese content

### Subtitles appear too early

This is the most common problem.  Causes include:
- Whisper detects the *start of breathing or silence* before speech begins.
- The ASR model cuts segments slightly early.

**Fix:**

```bash
# Add 100–200 ms start delay
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --start-delay 150
```

If only the first few minutes are off, use `--start-delay`.  
If the *entire* file is off by a consistent amount, prefer `--global-shift`.

### Subtitles appear too late

```bash
# Negative global shift moves subtitles earlier
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --global-shift -200
```

### Subtitles are too long / hard to read

```bash
# Reduce max characters per line (default is 16)
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --max-chars 14
```

### Subtitles flicker (too short on screen)

```bash
# Increase minimum display duration
docker run --rm \
  -v /path/to/your/models:/models \
  -v /path/to/your/videos:/input \
  -v /path/to/your/output:/output \
  autostr /input/input.mp4 --min-duration 1.2
```

### Improving accuracy

| Scenario | Recommendation |
|---|---|
| Good CPU, not in a hurry | `--model large-v2` |
| Fast, acceptable accuracy | `--model medium` (default) |
| Very fast, lower accuracy | `--model small` |
| GPU available | `--device cuda --compute-type float16 --model large-v3` |
| Fine-grained word alignment | Keep WhisperX enabled (default) |
| Skip WhisperX (faster) | `--no-whisperx` |

---

## Troubleshooting SRT uploads

If YouTube or another platform reports errors for specific subtitle lines, the
problem is usually SRT formatting rather than the transcript text itself.

Check that every subtitle block has this exact structure:

1. A numeric subtitle index.
2. A time range line in the form `00:00:00,000 --> 00:00:00,000`.
3. One or more subtitle text lines.
4. A blank line before the next block.

Common causes of import errors:

- Orphan text lines without an index or time range.
- Missing blank lines between subtitle blocks.
- Broken numbering after manual edits.
- Non-SRT timecode format.

AutoStr writes standard SRT output, but manual post-editing can break the
structure. If a platform flags a few line numbers, inspect the surrounding
blocks first.

---

## Recommended parameters for Chinese content

| Use case | Suggested command |
|---|---|
| Standard Mandarin lecture/talk | `--model medium --start-delay 100` |
| Fast speech / accent | `--model large-v2 --start-delay 150` |
| Long documentary (GPU) | `--device cuda --compute-type float16 --model large-v3` |
| Quick draft subtitles | `--model small --no-whisperx` |
| Tight reading pace | `--max-chars 14 --min-duration 1.0` |

---

## Project structure

```
AutoStr/
├── autostr/                  # Python package
│   ├── __init__.py
│   ├── audio.py              # ffmpeg audio extraction
│   ├── highlight.py          # highlight scoring and clip export
│   ├── transcribe.py         # faster-whisper ASR
│   ├── align.py              # WhisperX fine-grained alignment
│   ├── reflow.py             # Chinese segmentation & reflow
│   ├── srt_writer.py         # SRT serialisation
│   └── pipeline.py           # End-to-end orchestrator
├── tests/                    # Unit tests (pytest)
│   ├── test_reflow_and_srt.py
│   └── test_cli.py
├── main.py                   # CLI entrypoint
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── docker-compose.gpu.yml
└── .dockerignore
```

---

## Running without Docker

```bash
# Install system dependencies
sudo apt-get install ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Optional: WhisperX for fine-grained alignment
pip install git+https://github.com/m-bain/whisperX.git

# Run
python main.py input.mp4
```

---

## Running the tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Extending the pipeline

The pipeline is designed to be extended.  Suggested extension points:

- **VAD / silence detection** – add a `autostr/vad.py` module that uses
  [pyannote-audio](https://github.com/pyannote/pyannote-audio) or
  [silero-vad](https://github.com/snakers4/silero-vad) to further refine
  segment boundaries.
- **Speaker diarisation** – integrate WhisperX's diarisation support to
  label subtitles per speaker.
- **Custom vocabulary** – pass a `initial_prompt` to faster-whisper with
  domain-specific terms, names, or technical vocabulary.
- **Batch processing** – wrap `pipeline.run()` in a loop or use
  `multiprocessing` to process many files in parallel.

---

## Licence

MIT
