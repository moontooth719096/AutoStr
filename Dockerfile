# ─────────────────────────────────────────────────────────────────────────────
# AutoStr – Dockerfile
#
# Build (CPU):
#   docker build -t autostr .
#
# Build (GPU/CUDA):
#   docker build --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime \
#                -t autostr:cuda .
#
# Run:
#   docker run --rm -v /path/to/videos:/data autostr /data/input.mp4
# ─────────────────────────────────────────────────────────────────────────────

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

LABEL maintainer="AutoStr" \
      description="Chinese video subtitle alignment and reflow pipeline"

# ── System dependencies ────────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ── Python environment ─────────────────────────────────────────────────────────
WORKDIR /app

# Copy dependency manifest first (improves layer caching)
COPY requirements.txt .

# Install PyTorch CPU-only wheel first to avoid downloading the full CUDA
# bundle when a GPU is not needed.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Optionally install WhisperX (commented out by default because it pulls in
# additional large dependencies; enable for fine-grained alignment).
# RUN pip install --no-cache-dir git+https://github.com/m-bain/whisperX.git

# ── Application code ───────────────────────────────────────────────────────────
COPY autostr/ autostr/
COPY main.py .

# ── Default model pre-download (optional) ─────────────────────────────────────
# Pre-downloading the model into the image makes the first run faster.
# Comment out to skip (model will be downloaded on first use instead).
ARG WHISPER_MODEL=medium
RUN python - <<'EOF'
import os, sys
model = os.environ.get("WHISPER_MODEL", "medium")
# Pre-download via faster-whisper's internal hub
try:
    from faster_whisper import WhisperModel
    print(f"Pre-downloading faster-whisper model '{model}' …", flush=True)
    WhisperModel(model, device="cpu", compute_type="int8")
    print("Done.", flush=True)
except Exception as e:
    print(f"Warning: could not pre-download model: {e}", file=sys.stderr, flush=True)
EOF

# ── Runtime defaults ───────────────────────────────────────────────────────────
# Mount a volume at /data and pass the filename as the first argument.
VOLUME ["/data"]
WORKDIR /data

ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]
