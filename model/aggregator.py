"""Session-scoped feature aggregation and fusion inference."""

from __future__ import annotations

import math
import time
from typing import Any

from .inference import ModelUnavailableError, classify_features, model_status

EXPRESSION_MAP = {"NEUTRAL": 0.0, "CONFUSED": 1.0, "FOCUSED": 2.0}
LIVE_FEATURE_MAX_AGE_SECONDS = 15


class Aggregator:
    def __init__(self) -> None:
        self.active_session_id: str | None = None
        self.reset_for_session(None)

    def reset_for_session(self, session_id: str | None) -> None:
        self.active_session_id = session_id
        self.cv_buffer: list[dict[str, Any]] = []
        self.behaviour_buffer: list[dict[str, Any]] = []
        self.quiz_buffer: list[dict[str, Any]] = []
        self.latest_result: dict[str, Any] = {
            "timestamp": int(time.time()),
            "status": "waiting_for_session" if session_id is None else "waiting_for_features",
            "load_class": None,
            "load_score": None,
            "session_id": session_id,
            "model": model_status(),
            "features_available": [],
        }

    def _add(self, buffer: list[dict[str, Any]], data: dict[str, Any], limit: int) -> None:
        if not self.active_session_id or data.get("session_id") != self.active_session_id:
            raise ValueError("Signal session_id does not match the active session")
        sample = dict(data)
        sample.setdefault("timestamp", int(time.time()))
        buffer.append(sample)
        del buffer[:-limit]

    def add_cv(self, data: dict[str, Any]) -> None:
        self._add(self.cv_buffer, data, 10)

    def add_behaviour(self, data: dict[str, Any]) -> None:
        self._add(self.behaviour_buffer, data, 10)

    def add_quiz(self, data: dict[str, Any]) -> None:
        self._add(self.quiz_buffer, data, 5)

    @staticmethod
    def _average(buffer: list[dict[str, Any]], key: str) -> float:
        values = [float(item[key]) for item in buffer]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError(f"No valid values available for {key}")
        return sum(values) / len(values)

    def classify(self) -> dict[str, Any]:
        if not self.active_session_id:
            return self._set_status("waiting_for_session")

        available = [
            name for name, buffer in (
                ("cv", self.cv_buffer),
                ("behaviour", self.behaviour_buffer),
                ("quiz", self.quiz_buffer),
            ) if buffer
        ]
        if len(available) != 3:
            return self._set_status("waiting_for_features", features_available=available)

        now = time.time()
        stale = []
        for name, buffer in (("cv", self.cv_buffer), ("behaviour", self.behaviour_buffer)):
            try:
                age = now - float(buffer[-1]["timestamp"])
            except (KeyError, TypeError, ValueError):
                age = float("inf")
            if age < 0 or age > LIVE_FEATURE_MAX_AGE_SECONDS:
                stale.append(name)
        if stale:
            return self._set_status(
                "waiting_for_fresh_features",
                features_available=available,
                stale_features=stale,
            )

        features = {
            "blink_rate": self._average(self.cv_buffer, "blink_rate"),
            "gaze_score": self._average(self.cv_buffer, "gaze_score"),
            "expression": sum(EXPRESSION_MAP[item["expression"].upper()] for item in self.cv_buffer) / len(self.cv_buffer),
            "wpm": self._average(self.behaviour_buffer, "wpm"),
            "backspace_rate": self._average(self.behaviour_buffer, "backspace_rate"),
            "scroll_reversals": self._average(self.behaviour_buffer, "scroll_reversals"),
            "mouse_idle": self._average(self.behaviour_buffer, "mouse_idle"),
            "response_time": self._average(self.quiz_buffer, "response_time"),
            "confidence": self._average(self.quiz_buffer, "confidence"),
        }
        feature_vector = [
            features["blink_rate"], features["gaze_score"], features["expression"],
            features["wpm"], features["backspace_rate"], features["scroll_reversals"],
            features["mouse_idle"], features["response_time"], features["confidence"],
            0.0,  # f10 is a reserved architecture slot; no collected feature maps to it.
            0.0,  # f11 is a reserved architecture slot; no collected feature maps to it.
        ]

        common = {
            "timestamp": int(time.time()),
            "session_id": self.active_session_id,
            "features_available": available,
            "features": {key: round(value, 5) for key, value in features.items()},
            "feature_count": len(feature_vector),
            "model": model_status(),
        }
        if not common["model"]["ready"]:
            self.latest_result = {
                **common,
                "status": "model_unavailable",
                "error": common["model"]["error"],
                "load_class": None,
                "load_score": None,
            }
            return self.latest_result
        try:
            load_class, load_score = classify_features(feature_vector)
        except ModelUnavailableError as exc:
            self.latest_result = {**common, "status": "model_unavailable", "error": str(exc), "load_class": None, "load_score": None}
        except Exception as exc:
            self.latest_result = {**common, "status": "inference_error", "error": str(exc), "load_class": None, "load_score": None}
        else:
            self.latest_result = {
                **common,
                "status": "ready",
                "load_class": load_class,
                "load_score": load_score,
                "load_score_definition": "maximum softmax class probability; not calibrated",
            }
        return self.latest_result

    def _set_status(self, status: str, **extra: Any) -> dict[str, Any]:
        self.latest_result = {
            "timestamp": int(time.time()),
            "status": status,
            "load_class": None,
            "load_score": None,
            "session_id": self.active_session_id,
            "model": model_status(),
            **extra,
        }
        return self.latest_result
