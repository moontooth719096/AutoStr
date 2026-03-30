import os
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

def load_whisper_model(
    model_name: str,
    model_dir: str | Path | None = None,
    device: str | None = None,
    compute_type: str | None = None,
):
    model_root = Path(model_dir) if model_dir is not None else MODEL_DIR
    model_root.mkdir(parents=True, exist_ok=True)

    return WhisperModel(
        model_name,
        device=device or os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=compute_type or os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        download_root=str(model_root),
    )