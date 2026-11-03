"""Normalize behaviour windows to the 0-1 scale stored in SQLite and trained on."""

from __future__ import annotations

import math
import time
from typing import Any


def _finite_number(data: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key} must be a finite number")
    return number


def validate_and_normalise(data: dict[str, Any]) -> dict[str, Any]:
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id is required")
    timestamp = int(_finite_number(data, "timestamp", time.time()))
    wpm = min(max(_finite_number(data, "wpm"), 0.0), 120.0)
    backspace_rate = min(max(_finite_number(data, "backspace_rate"), 0.0), 1.0)
    reversals = min(max(_finite_number(data, "scroll_reversals"), 0.0), 20.0)
    mouse_idle = min(max(_finite_number(data, "mouse_idle_sec"), 0.0), 60.0)
    return {
        "session_id": session_id.strip(),
        "timestamp": timestamp,
        "wpm": round(wpm / 120.0, 5),
        "backspace_rate": round(backspace_rate, 5),
        "scroll_reversals": round(reversals / 20.0, 5),
        "mouse_idle": round(mouse_idle / 60.0, 5),
    }
