"""Train and evaluate the existing fusion MLP without participant leakage."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_builder import CLASS_NAMES, FEATURE_COLUMNS, OUTPUT_CSV, REPORT_JSON

MODEL_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = MODEL_DIR / "fusion_model.pth"
PREPROCESSING_PATH = MODEL_DIR / "preprocessing.json"
EVALUATION_PATH = MODEL_DIR / "evaluation_report.json"
LABEL_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def _split_by_participant(df: pd.DataFrame, labels: np.ndarray, groups: np.ndarray):
    """Find participant-disjoint train, validation and test partitions with all classes."""
    from sklearn.model_selection import GroupShuffleSplit

    all_classes = set(range(len(CLASS_NAMES)))
    for outer_seed in range(1000):
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=outer_seed + 42)
        train_validation, test = next(outer.split(df, labels, groups))
        if set(np.unique(labels[test])) != all_classes or min(np.bincount(labels[test], minlength=3)) < 2:
            continue
        remaining_labels = labels[train_validation]
        remaining_groups = groups[train_validation]
        for inner_seed in range(100):
            inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=outer_seed * 100 + inner_seed + 1)
            train_local, validation_local = next(inner.split(train_validation, remaining_labels, remaining_groups))
            train = train_validation[train_local]
            validation = train_validation[validation_local]
            if set(np.unique(labels[train])) != all_classes or set(np.unique(labels[validation])) != all_classes:
                continue
            if min(np.bincount(labels[train], minlength=3)) < 3:
                continue
            return train, validation, test
    raise ValueError(
        "Cannot create participant-disjoint train/validation/test sets with all three classes. "
        "Collect more independently labeled sessions from more participants in every class."
    )


def _predict(model, values: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return model(torch.tensor(values, dtype=torch.float32)).argmax(dim=1).cpu().numpy()


def train() -> dict:
    if not OUTPUT_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {OUTPUT_CSV}. Run python dataset_builder.py first.")
    df = pd.read_csv(OUTPUT_CSV)
    if df.empty:
        report = json.loads(REPORT_JSON.read_text(encoding="utf-8")) if REPORT_JSON.exists() else {}
        raise ValueError(
            "No usable NASA-TLX-labeled sessions are available; refusing to train. "
            f"Dataset report: {report.get('usable_sessions', 0)} usable sessions, "
            f"{report.get('class_distribution', {})} class distribution."
        )
    required = {"session_id", "student_id", "label_source", "nasa_tlx_score", "load_label", *FEATURE_COLUMNS}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Training CSV is missing required columns: {missing_columns}")
    if df["session_id"].duplicated().any():
        raise ValueError("Expected one row per session; duplicate session_id values would distort evaluation")
    if not (df["label_source"] == "NASA-TLX").all():
        raise ValueError("Only post-study NASA-TLX labels may be used for training")
    if not set(df["load_label"].unique()).issubset(LABEL_TO_INDEX):
        raise ValueError("Dataset contains a class outside LOW, MEDIUM, OVERLOADED")
    nasa_scores = df["nasa_tlx_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(nasa_scores).all() or ((nasa_scores < 0) | (nasa_scores > 100)).any():
        raise ValueError("NASA-TLX labels must be finite scores from 0 to 100")
    nasa_classes = np.where(nasa_scores <= 33, "LOW", np.where(nasa_scores <= 66, "MEDIUM", "OVERLOADED"))
    if not np.array_equal(nasa_classes, df["load_label"].to_numpy()):
        raise ValueError("load_label does not match the documented NASA-TLX thresholds")
    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).all():
        raise ValueError("Feature matrix contains missing or non-finite values")
    if not np.allclose(df[["f10", "f11"]].to_numpy(dtype=np.float64), 0.0):
        raise ValueError("f10 and f11 are reserved compatibility inputs and must remain zero")
    # Keep participant/session trace columns in the CSV, never in the model matrix.
    labels = df["load_label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
    if min(np.bincount(labels, minlength=3)) < 5:
        raise ValueError("At least five labeled sessions per class are required before grouped evaluation")
    groups = df["student_id"].fillna("").astype(str).to_numpy()
    groups = np.asarray([group if group.strip() else df.iloc[index]["session_id"] for index, group in enumerate(groups)])
    train_index, validation_index, test_index = _split_by_participant(df, labels, groups)

    try:
        import torch
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from .fusion_model import FusionModel
    except ImportError as exc:
        raise RuntimeError(f"Training dependencies are unavailable: {exc}") from exc

    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_index]).astype(np.float32)
    x_validation = scaler.transform(x[validation_index]).astype(np.float32)
    x_test = scaler.transform(x[test_index]).astype(np.float32)
    y_train, y_validation, y_test = labels[train_index], labels[validation_index], labels[test_index]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    model = FusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=min(16, len(y_train)),
        shuffle=True,
    )
    best_loss = float("inf")
    best_state = None
    patience = 30
    epochs_without_improvement = 0
    for _epoch in range(300):
        model.train()
        for features, target in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(torch.tensor(x_validation)), torch.tensor(y_validation)).item())
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), WEIGHTS_PATH)

    test_predictions = _predict(model, x_test)
    train_predictions = _predict(model, x_train)
    validation_predictions = _predict(model, x_validation)
    test_counts = {name: int(sum(labels[index] == label for index in test_index)) for label, name in enumerate(CLASS_NAMES)}
    participants_per_class = {name: int(df.loc[df.load_label == name, "student_id"].nunique()) for name in CLASS_NAMES}
    evaluation = {
        "sessions": len(df),
        "participants": int(df.student_id.nunique()),
        "class_distribution": {name: int((df.load_label == name).sum()) for name in CLASS_NAMES},
        "participants_per_class": participants_per_class,
        "split_sessions": {"train": len(train_index), "validation": len(validation_index), "test": len(test_index)},
        "split_participants": {
            "train": int(pd.Series(groups[train_index]).nunique()),
            "validation": int(pd.Series(groups[validation_index]).nunique()),
            "test": int(pd.Series(groups[test_index]).nunique()),
        },
        "test_class_support": test_counts,
        "train_accuracy": float(accuracy_score(y_train, train_predictions)),
        "validation_accuracy": float(accuracy_score(y_validation, validation_predictions)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "per_class": classification_report(y_test, test_predictions, labels=[0, 1, 2], target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1, 2]).tolist(),
        "metrics_supported": all(test_counts[name] >= 2 for name in CLASS_NAMES),
        "meaningful_evaluation_support": all(participants_per_class[name] >= 10 for name in CLASS_NAMES),
        "overfitting_check": "Compare train, validation, and held-out test accuracy; held-out test was not used for training or early stopping.",
        "feature_columns": FEATURE_COLUMNS,
        "classes": CLASS_NAMES,
        "label_source": "NASA-TLX",
    }
    config = {
        "feature_columns": FEATURE_COLUMNS,
        "classes": CLASS_NAMES,
        "mean": scaler.mean_.astype(float).tolist(),
        "scale": scaler.scale_.astype(float).tolist(),
        "scaler": "StandardScaler fitted on participant-disjoint training data only",
        "reserved_features": {"f10": 0.0, "f11": 0.0},
    }
    PREPROCESSING_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    print(f"Weights: {WEIGHTS_PATH}\nPreprocessing: {PREPROCESSING_PATH}\nEvaluation: {EVALUATION_PATH}")
    return evaluation


if __name__ == "__main__":
    try:
        train()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Training not run: {exc}") from exc
