"""Build one traceable model row per complete session with a verified NASA-TLX label."""

from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from database import DB_PATH, init_db

FEATURE_COLUMNS = [
    "blink_rate", "gaze_score", "expression", "wpm", "backspace_rate",
    "scroll_reversals", "mouse_idle", "response_time", "confidence", "f10", "f11",
]
CLASS_NAMES = ["LOW", "MEDIUM", "OVERLOADED"]
EXPRESSION_MAP = {"NEUTRAL": 0.0, "CONFUSED": 1.0, "FOCUSED": 2.0}
OUTPUT_CSV = Path(__file__).resolve().parent / "dataset" / "training_dataset.csv"
REPORT_JSON = Path(__file__).resolve().parent / "dataset" / "dataset_validation_report.json"


def _mean_valid(rows: list[sqlite3.Row], fields: tuple[str, ...], *, ranges: dict[str, tuple[float, float]] | None = None) -> tuple[dict[str, float], int]:
    accepted: list[dict[str, float]] = []
    dropped = 0
    ranges = ranges or {}
    for row in rows:
        item: dict[str, float] = {}
        valid = True
        for field in fields:
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    valid = False
                    break
                if field in ranges and not ranges[field][0] <= value <= ranges[field][1]:
                    valid = False
                    break
                item[field] = value
            except (TypeError, ValueError):
                valid = False
                break
        if valid:
            accepted.append(item)
        else:
            dropped += 1
    if not accepted:
        return {}, dropped
    return {field: sum(item[field] for item in accepted) / len(accepted) for field in fields}, dropped


def _valid_cv_mean(rows: list[sqlite3.Row]) -> tuple[dict[str, float], int]:
    accepted: list[dict[str, float]] = []
    dropped = 0
    for row in rows:
        try:
            blink_rate = float(row["blink_rate"])
            gaze_score = float(row["gaze_score"])
            expression = EXPRESSION_MAP[str(row["expression"] or "").upper().strip()]
            if not all(math.isfinite(value) and value >= 0 for value in (blink_rate, gaze_score)):
                raise ValueError("invalid CV number")
        except (KeyError, TypeError, ValueError):
            dropped += 1
            continue
        accepted.append({"blink_rate": blink_rate, "gaze_score": gaze_score, "expression": expression})
    if not accepted:
        return {}, dropped
    return {field: sum(item[field] for item in accepted) / len(accepted) for field in accepted[0]}, dropped


def _invalid_value_counts(conn: sqlite3.Connection) -> dict[str, int]:
    counts: Counter[str] = Counter()
    definitions = {
        "cv_signals": {
            "blink_rate": (0, math.inf), "gaze_score": (0, math.inf),
        },
        "behaviour_signals": {
            "wpm": (0, 1), "backspace_rate": (0, 1), "scroll_reversals": (0, 1), "mouse_idle": (0, 1),
        },
        "quiz_responses": {
            "response_time": (0, math.inf), "confidence": (1, 5), "is_correct": (0, 1),
        },
    }
    for table, fields in definitions.items():
        rows = conn.execute(f"SELECT {', '.join(fields)} FROM {table}").fetchall()
        for field, (minimum, maximum) in fields.items():
            for row in rows:
                try:
                    value = float(row[field])
                    if not math.isfinite(value) or not minimum <= value <= maximum:
                        counts[f"{table}.{field}"] += 1
                except (TypeError, ValueError):
                    counts[f"{table}.{field}"] += 1
    rows = conn.execute("SELECT expression FROM cv_signals").fetchall()
    counts["cv_signals.expression"] = sum(
        str(row["expression"] or "").upper().strip() not in EXPRESSION_MAP for row in rows
    )
    rows = conn.execute("SELECT gaze FROM cv_signals").fetchall()
    counts["cv_signals.gaze"] = sum(
        str(row["gaze"] or "").upper().strip() not in {"CENTER", "LEFT", "RIGHT"} for row in rows
    )
    return dict(counts)


def build_dataset() -> pd.DataFrame:
    init_db()
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    sessions = conn.execute("SELECT * FROM sessions ORDER BY id").fetchall()
    raw_signal_counts = {
        table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("cv_signals", "behaviour_signals", "quiz_responses")
    }
    prediction_rows_ignored = int(conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
    invalid_value_counts = _invalid_value_counts(conn)
    output_rows: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    dropped_rows: Counter[str] = Counter()

    try:
        for session in sessions:
            sid = session["id"]
            if session["assessment_status"] != "submitted":
                status = session["assessment_status"] or "unverified"
                reason = {
                    "pending": "NASA-TLX assessment is pending",
                    "missing": "NASA-TLX assessment is missing",
                    "unverified": "Stored score or label has no verified assessment submission",
                }.get(status, f"Assessment status is not verified: {status}")
                excluded.append({"session_id": sid, "reason": reason})
                continue
            if session["label_source"] != "NASA-TLX" or session["nasa_tlx_score"] is None:
                excluded.append({"session_id": sid, "reason": "Completed assessment source or score is missing"})
                continue
            try:
                score = float(session["nasa_tlx_score"])
            except (TypeError, ValueError):
                excluded.append({"session_id": sid, "reason": "NASA-TLX score is not numeric"})
                continue
            if not math.isfinite(score) or not 0 <= score <= 100:
                excluded.append({"session_id": sid, "reason": "NASA-TLX score is outside 0-100"})
                continue
            label = "LOW" if score <= 33 else "MEDIUM" if score <= 66 else "OVERLOADED"
            if session["load_class"] != label:
                excluded.append({"session_id": sid, "reason": "Stored class does not match NASA-TLX score mapping"})
                continue

            cv_rows = conn.execute(
                "SELECT blink_rate, gaze_score, expression FROM cv_signals WHERE session_id = ?", (sid,)
            ).fetchall()
            behavior_rows = conn.execute(
                "SELECT wpm, backspace_rate, scroll_reversals, mouse_idle FROM behaviour_signals WHERE session_id = ?", (sid,)
            ).fetchall()
            quiz_rows = conn.execute(
                "SELECT response_time, confidence, is_correct FROM quiz_responses WHERE session_id = ?", (sid,)
            ).fetchall()

            cv_values, dropped_cv = _valid_cv_mean(cv_rows)
            dropped_rows["cv"] += dropped_cv

            behavior_values, dropped_behavior = _mean_valid(
                behavior_rows,
                ("wpm", "backspace_rate", "scroll_reversals", "mouse_idle"),
                ranges={field: (0, 1) for field in ("wpm", "backspace_rate", "scroll_reversals", "mouse_idle")},
            )
            dropped_rows["behaviour"] += dropped_behavior

            quiz_values, dropped_quiz = _mean_valid(
                quiz_rows,
                ("response_time", "confidence", "is_correct"),
                ranges={"response_time": (0, math.inf), "confidence": (1, 5), "is_correct": (0, 1)},
            )
            dropped_rows["quiz"] += dropped_quiz

            if not cv_values or not behavior_values or not quiz_values:
                missing = []
                if not cv_values:
                    missing.append("valid CV data")
                if not behavior_values:
                    missing.append("valid normalized behaviour data")
                if not quiz_values:
                    missing.append("valid quiz data")
                excluded.append({"session_id": sid, "reason": "Missing " + ", ".join(missing)})
                continue

            row = {
                "session_id": sid,
                "student_id": session["student_id"] or sid,
                "assessment_status": session["assessment_status"],
                "blink_rate": cv_values["blink_rate"],
                "gaze_score": cv_values["gaze_score"],
                "expression": cv_values["expression"],
                "wpm": behavior_values["wpm"],
                "backspace_rate": behavior_values["backspace_rate"],
                "scroll_reversals": behavior_values["scroll_reversals"],
                "mouse_idle": behavior_values["mouse_idle"],
                "response_time": quiz_values["response_time"],
                "confidence": quiz_values["confidence"],
                # Reserved compatibility inputs in the existing model. No collected
                # signal has a validated meaning for either slot.
                "f10": 0.0,
                "f11": 0.0,
                "load_label": label,
                "nasa_tlx_score": score,
                "label_source": "NASA-TLX",
            }
            output_rows.append(row)
    finally:
        conn.close()

    columns = ["session_id", "student_id", "assessment_status", *FEATURE_COLUMNS,
               "load_label", "nasa_tlx_score", "label_source"]
    frame = pd.DataFrame(output_rows, columns=columns)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)

    class_counts = {name: int((frame["load_label"] == name).sum()) for name in CLASS_NAMES}
    participants_per_class = {
        name: int(frame.loc[frame["load_label"] == name, "student_id"].nunique()) for name in CLASS_NAMES
    }
    enough_for_evaluation = all(participants_per_class[name] >= 10 for name in CLASS_NAMES)
    report = {
        "database": str(DB_PATH),
        "dataset_csv": str(OUTPUT_CSV),
        "total_sessions": len(sessions),
        "raw_signal_rows": raw_signal_counts,
        "prediction_rows_ignored": prediction_rows_ignored,
        "assessment_status_counts": dict(Counter((session["assessment_status"] or "unverified") for session in sessions)),
        "label_source_counts": dict(Counter((session["label_source"] or "missing") for session in sessions)),
        "sessions_with_verified_nasa_tlx": sum(
            session["assessment_status"] == "submitted"
            and session["label_source"] == "NASA-TLX"
            and session["nasa_tlx_score"] is not None
            for session in sessions
        ),
        "usable_sessions": len(frame),
        "usable_rows": len(frame),
        "excluded_sessions": excluded,
        "dropped_signal_rows": dict(dropped_rows),
        "missing_or_invalid_value_counts": invalid_value_counts,
        "missing_or_invalid_values": {
            "gaze_score": "NULL in historical CV rows when the old collector did not persist it; such rows are excluded.",
            "expression": "Unknown or missing labels are excluded from session expression means.",
            "behaviour": "Must be finite and normalized to 0-1.",
            "quiz": "Response time must be nonnegative, confidence 1-5, correctness 0 or 1.",
        },
        "class_distribution": class_counts,
        "participants_per_class": participants_per_class,
        "feature_columns": FEATURE_COLUMNS,
        "feature_dtypes": {column: "float64" for column in FEATURE_COLUMNS},
        "trace_columns": ["session_id", "student_id", "assessment_status", "label_source", "nasa_tlx_score"],
        "f10_f11": "Reserved model-compatibility inputs, fixed to zero; no meaning is imputed.",
        "sufficient_for_meaningful_three_class_evaluation": enough_for_evaluation,
        "evaluation_note": "Meaningful held-out evaluation requires at least 10 distinct participants per class; no class is balanced with fabricated data.",
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Dataset written to {OUTPUT_CSV}")
    print(f"Validation report written to {REPORT_JSON}")
    return frame


if __name__ == "__main__":
    build_dataset()
