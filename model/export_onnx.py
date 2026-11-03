"""Export the trained fusion model and compare ONNX logits with PyTorch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset_builder import FEATURE_COLUMNS, OUTPUT_CSV
from .fusion_model import FusionModel

MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = MODEL_DIR / "fusion_model.pth"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.json"
MODEL_PATH = MODEL_DIR / "fusion_model.onnx"
CANDIDATE_PATH = MODEL_DIR / "fusion_model.candidate.onnx"


def export_onnx() -> dict:
    if not WEIGHTS_PATH.is_file() or not PREPROCESSING_PATH.is_file():
        raise FileNotFoundError("Train a model first; fusion_model.pth and preprocessing.json are both required")
    config = json.loads(PREPROCESSING_PATH.read_text(encoding="utf-8"))
    if config.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Preprocessing feature order does not match the model input schema")
    frame = pd.read_csv(OUTPUT_CSV)
    if frame.empty or not set(FEATURE_COLUMNS).issubset(frame.columns):
        raise ValueError("A nonempty, validated training dataset is required to verify the ONNX export")
    raw = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("Dataset contains non-finite features")
    mean = np.asarray(config["mean"], dtype=np.float32)
    scale = np.asarray(config["scale"], dtype=np.float32)
    if mean.shape != (11,) or scale.shape != (11,):
        raise ValueError("Preprocessing config must have 11 feature means and scales")
    normalized = ((raw[: min(32, len(raw))] - mean) / scale).astype(np.float32)

    model = FusionModel()
    state = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    dummy = torch.from_numpy(normalized)
    torch.onnx.export(
        model,
        dummy,
        str(CANDIDATE_PATH),
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    try:
        import onnxruntime as ort

        runtime = ort.InferenceSession(str(CANDIDATE_PATH), providers=["CPUExecutionProvider"])
        input_info, output_info = runtime.get_inputs(), runtime.get_outputs()
        if len(input_info) != 1 or len(output_info) != 1:
            raise ValueError("ONNX graph must have exactly one input and one output")
        if input_info[0].shape[-1] != 11 or output_info[0].shape[-1] != 3:
            raise ValueError(f"Unexpected ONNX dimensions: {input_info[0].shape} -> {output_info[0].shape}")
        with torch.no_grad():
            torch_logits = model(dummy).cpu().numpy()
        onnx_logits = runtime.run([output_info[0].name], {input_info[0].name: normalized})[0]
        max_abs_diff = float(np.max(np.abs(torch_logits - onnx_logits)))
        if not np.allclose(torch_logits, onnx_logits, rtol=1e-3, atol=1e-4):
            raise ValueError(f"PyTorch and ONNX logits differ beyond tolerance (max abs diff {max_abs_diff})")
        torch_predictions = torch_logits.argmax(axis=1)
        onnx_predictions = onnx_logits.argmax(axis=1)
        if not np.array_equal(torch_predictions, onnx_predictions):
            raise ValueError("PyTorch and ONNX class predictions disagree")
        CANDIDATE_PATH.replace(MODEL_PATH)
    except Exception:
        if CANDIDATE_PATH.exists():
            CANDIDATE_PATH.unlink()
        raise

    result = {
        "model_path": str(MODEL_PATH),
        "input_shape": input_info[0].shape,
        "output_shape": output_info[0].shape,
        "classes": config["classes"],
        "verified_rows": len(normalized),
        "max_absolute_logit_difference": max_abs_diff,
        "predictions_agree": True,
        "tolerance": {"rtol": 1e-3, "atol": 1e-4},
    }
    (MODEL_DIR / "onnx_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    export_onnx()
