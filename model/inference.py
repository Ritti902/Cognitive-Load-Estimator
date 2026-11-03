"""ONNX inference with the preprocessing saved by the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ["LOW", "MEDIUM", "OVERLOADED"]
MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "fusion_model.onnx"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.json"
FEATURE_COLUMNS = [
    "blink_rate", "gaze_score", "expression", "wpm", "backspace_rate",
    "scroll_reversals", "mouse_idle", "response_time", "confidence", "f10", "f11",
]

session: Any = None
preprocessing: dict[str, Any] | None = None
model_error: str | None = None


class ModelUnavailableError(RuntimeError):
    """Raised when no verified ONNX model and preprocessing file are available."""


def load_model() -> bool:
    global session, preprocessing, model_error
    session = None
    preprocessing = None
    model_error = None
    if not MODEL_PATH.is_file():
        model_error = f"ONNX model not found: {MODEL_PATH}"
        return False
    if not PREPROCESSING_PATH.is_file():
        model_error = f"Training preprocessing configuration not found: {PREPROCESSING_PATH}"
        return False
    try:
        import onnxruntime as ort

        config = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
        if config.get("feature_columns") != FEATURE_COLUMNS or config.get("classes") != CLASSES:
            raise ValueError("Preprocessing feature order or class mapping does not match inference")
        mean = np.asarray(config.get("mean"), dtype=np.float32)
        scale = np.asarray(config.get("scale"), dtype=np.float32)
        if mean.shape != (11,) or scale.shape != (11,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Preprocessing mean/scale must contain 11 finite values and positive scales")
        candidate = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        inputs = candidate.get_inputs()
        outputs = candidate.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape[-1] != 11 or outputs[0].shape[-1] != 3:
            raise ValueError("ONNX model must accept 11 features and return 3 class logits")
        session = candidate
        preprocessing = config
    except Exception as exc:
        model_error = f"Could not load ONNX model: {exc}"
        session = None
        preprocessing = None
        return False
    return True


def model_status() -> dict[str, Any]:
    return {"ready": session is not None, "error": model_error, "classes": CLASSES, "feature_count": len(FEATURE_COLUMNS)}


def classify_features(feature_vector: list[float] | np.ndarray) -> tuple[str, float]:
    if session is None or preprocessing is None:
        raise ModelUnavailableError(model_error or "ONNX model is not loaded")
    values = np.asarray(feature_vector, dtype=np.float32)
    if values.shape != (11,) or not np.isfinite(values).all():
        raise ValueError("Inference requires exactly 11 finite numeric features")

    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    scale = np.asarray(preprocessing["scale"], dtype=np.float32)
    normalized = ((values - mean) / scale).reshape(1, 11).astype(np.float32)
    inputs = session.get_inputs()
    output_name = session.get_outputs()[0].name
    logits = np.asarray(session.run([output_name], {inputs[0].name: normalized})[0], dtype=np.float64)
    if logits.shape != (1, 3) or not np.isfinite(logits).all():
        raise ValueError("ONNX output must be one finite set of three class logits")
    shifted = logits[0] - logits[0].max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    class_index = int(np.argmax(probabilities))
    return CLASSES[class_index], round(float(probabilities[class_index]), 3)


load_model()
