import os
from pathlib import Path

from faster_whisper import WhisperModel

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

def load_whisper_model(model_name: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    return WhisperModel(
        model_name,
        device=os.getenv("WHISPER_DEVICE", "cpu"),
        compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        download_root=str(MODEL_DIR),
    )