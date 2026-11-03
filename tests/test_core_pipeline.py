"""Focused regression checks for session safety, normalization, and traceability."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import database
import dataset_builder
import collect_sessions
from behaviour_pipeline.features import validate_and_normalise
from model import aggregator as aggregator_module


class DatabaseMigrationTests(unittest.TestCase):
    def test_migration_preserves_existing_records_and_labels_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            test_db = Path(directory) / "existing.db"
            connection = sqlite3.connect(test_db)
            connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, student_id TEXT, date TEXT, difficulty TEXT, nasa_tlx_score REAL, load_class TEXT)")
            connection.execute("INSERT INTO sessions VALUES ('old-session', 'participant-1', '2025-01-01', 'easy', 20, 'LOW')")
            connection.execute("INSERT INTO sessions VALUES ('empty-session', 'participant-2', '2025-01-02', 'easy', NULL, NULL)")
            connection.commit()
            connection.close()

            with mock.patch.object(database, "DB_PATH", test_db):
                database.init_db()
                self.assertEqual(database.get_session("old-session")["load_class"], "LOW")
                self.assertIsNone(database.get_session("old-session")["label_source"])
                self.assertEqual(database.get_session("old-session")["assessment_status"], "unverified")
                self.assertEqual(database.get_session("empty-session")["assessment_status"], "missing")
                database.save_session("new-session", "participant-2", "medium")
                self.assertEqual(database.get_session("new-session")["assessment_status"], "pending")
                score, label = database.save_nasa_tlx("new-session", {
                    "mental": 10, "physical": 10, "temporal": 10,
                    "performance": 10, "effort": 10, "frustration": 10,
                })
                self.assertEqual((score, label), (50.0, "MEDIUM"))
                self.assertEqual(database.get_session("new-session")["label_source"], "NASA-TLX")
                self.assertEqual(database.get_session("new-session")["assessment_status"], "submitted")
                self.assertFalse(database.save_score({
                    "timestamp": 100, "session_id": "new-session", "status": "model_unavailable",
                    "load_class": "LOW", "load_score": 0.1,
                }))
                self.assertTrue(database.save_score({
                    "timestamp": 101, "session_id": "new-session", "status": "ready",
                    "load_class": "MEDIUM", "load_score": 0.7,
                }))
                with database.connect() as conn:
                    row = conn.execute("SELECT session_id, load_class FROM scores").fetchone()
                    self.assertEqual(tuple(row), ("new-session", "MEDIUM"))
                with self.assertRaises(ValueError):
                    database.save_nasa_tlx("new-session", {
                        "mental": 10, "physical": 10, "temporal": 10,
                        "performance": 10, "effort": 10, "frustration": 10,
                    })

    def test_collector_creates_unique_unlabeled_sessions_and_activates_each(self):
        with tempfile.TemporaryDirectory() as directory:
            test_db = Path(directory) / "sessions.db"
            posted = []

            def activate(url, json, timeout):
                posted.append((url, json["session_id"]))
                response = mock.Mock()
                response.json.return_value = {"status": "ok", "session_id": json["session_id"]}
                response.raise_for_status.return_value = None
                return response

            with mock.patch.object(database, "DB_PATH", test_db), mock.patch.object(collect_sessions.requests, "post", side_effect=activate):
                first = collect_sessions.create_session("p-1", "easy", "http://fastapi.test")
                second = collect_sessions.create_session("p-1", "hard", "http://fastapi.test")
                self.assertNotEqual(first, second)
                self.assertEqual([item[1] for item in posted], [first, second])
                self.assertEqual(database.get_session(first)["label_source"], None)
                self.assertIsNone(database.get_session(second)["nasa_tlx_score"])

    def test_collector_reports_activation_failure_without_erasing_session(self):
        with tempfile.TemporaryDirectory() as directory:
            test_db = Path(directory) / "pending.db"
            with mock.patch.object(database, "DB_PATH", test_db), mock.patch.object(
                collect_sessions.requests, "post", side_effect=collect_sessions.requests.ConnectionError("offline")
            ):
                with self.assertRaisesRegex(RuntimeError, "was saved, but FastAPI activation failed"):
                    collect_sessions.create_session("p-1", "easy", "http://fastapi.test")
                with database.connect() as connection:
                    rows = connection.execute("SELECT COUNT(*), MAX(label_source) FROM sessions").fetchone()
                self.assertEqual(rows[0], 1)
                self.assertIsNone(rows[1])


class BehaviourNormalizationTests(unittest.TestCase):
    def test_normalizes_to_training_scale_and_never_records_text(self):
        output = validate_and_normalise({
            "session_id": "session-a", "timestamp": 100, "wpm": 60,
            "backspace_rate": 0.25, "scroll_reversals": 10, "mouse_idle_sec": 30,
        })
        self.assertEqual(output, {
            "session_id": "session-a", "timestamp": 100, "wpm": 0.5,
            "backspace_rate": 0.25, "scroll_reversals": 0.5, "mouse_idle": 0.5,
        })
        self.assertNotIn("text", output)
        with self.assertRaises(ValueError):
            validate_and_normalise({"wpm": float("nan")})


class DatasetTraceabilityTests(unittest.TestCase):
    def test_builder_only_includes_complete_verified_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            test_db = Path(directory) / "fixture.db"
            output_csv = Path(directory) / "training_dataset.csv"
            report_path = Path(directory) / "dataset_validation_report.json"
            with mock.patch.object(database, "DB_PATH", test_db), \
                 mock.patch.object(dataset_builder, "DB_PATH", test_db), \
                 mock.patch.object(dataset_builder, "OUTPUT_CSV", output_csv), \
                 mock.patch.object(dataset_builder, "REPORT_JSON", report_path):
                database.init_db()
                database.save_session("complete", "p-1", "easy")
                database.save_nasa_tlx("complete", {
                    "mental": 5, "physical": 5, "temporal": 5,
                    "performance": 5, "effort": 5, "frustration": 5,
                })
                database.save_session("unverified", "p-2", "easy")
                # A legacy class and score without a label source remain excluded.
                with database.connect() as conn:
                    conn.execute("UPDATE sessions SET nasa_tlx_score=20, load_class='LOW', assessment_status='unverified' WHERE id='unverified'")
                database.save_cv_signal("complete", {
                    "timestamp": 100, "blink_rate": 12, "gaze": "CENTER", "gaze_score": 0.1, "expression": "NEUTRAL",
                })
                database.save_behaviour_signal("complete", {
                    "timestamp": 100, "wpm": 0.5, "backspace_rate": 0.2,
                    "scroll_reversals": 0.1, "mouse_idle": 0.4,
                })
                database.save_quiz_response("complete", 1, 8.0, True, 4, "attempt-1")

                database.save_session("invalid-features", "p-3", "hard")
                database.save_nasa_tlx("invalid-features", {
                    "mental": 15, "physical": 15, "temporal": 15,
                    "performance": 15, "effort": 15, "frustration": 15,
                })
                database.save_cv_signal("invalid-features", {
                    "timestamp": 100, "blink_rate": 12, "gaze": "CENTER", "gaze_score": 0.1, "expression": "NEUTRAL",
                })
                database.save_behaviour_signal("invalid-features", {
                    "timestamp": 100, "wpm": 0.5, "backspace_rate": 0.2,
                    "scroll_reversals": 0.1, "mouse_idle": 0.4,
                })
                database.save_quiz_response("invalid-features", 1, 8.0, True, 4, "attempt-2")
                with database.connect() as conn:
                    conn.execute("UPDATE cv_signals SET gaze_score=NULL WHERE session_id='invalid-features'")

                frame = dataset_builder.build_dataset()
                self.assertEqual(len(frame), 1)
                self.assertEqual(frame.iloc[0]["session_id"], "complete")
                self.assertEqual(frame.iloc[0]["f10"], 0.0)
                self.assertEqual(frame.iloc[0]["f11"], 0.0)
                self.assertEqual(frame.iloc[0]["assessment_status"], "submitted")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(report["usable_sessions"], 1)
                self.assertEqual(report["sessions_with_verified_nasa_tlx"], 2)
                self.assertEqual(report["assessment_status_counts"]["unverified"], 1)
                self.assertEqual(report["label_source_counts"]["NASA-TLX"], 2)
                self.assertEqual(report["label_source_counts"]["missing"], 1)
                self.assertEqual(report["dropped_signal_rows"]["cv"], 1)
                self.assertEqual(report["missing_or_invalid_value_counts"]["cv_signals.gaze_score"], 1)
                invalid = next(item for item in report["excluded_sessions"] if item["session_id"] == "invalid-features")
                self.assertEqual(invalid["reason"], "Missing valid CV data")
                self.assertFalse(report["sufficient_for_meaningful_three_class_evaluation"])
                self.assertTrue(any(item["session_id"] == "unverified" for item in report["excluded_sessions"]))


class AggregatorSessionTests(unittest.TestCase):
    def test_switching_sessions_clears_buffers_and_requires_all_modalities(self):
        aggregator = aggregator_module.Aggregator()
        aggregator.reset_for_session("session-a")
        with self.assertRaises(ValueError):
            aggregator.add_cv({"session_id": "session-old"})
        waiting = aggregator.classify()
        self.assertEqual(waiting["status"], "waiting_for_features")
        self.assertEqual(waiting["load_class"], None)

        aggregator.add_cv({"session_id": "session-a", "blink_rate": 12, "gaze_score": 0.1, "expression": "NEUTRAL"})
        aggregator.add_behaviour({"session_id": "session-a", "wpm": 0.2, "backspace_rate": 0.1, "scroll_reversals": 0.1, "mouse_idle": 0.5})
        aggregator.add_quiz({"session_id": "session-a", "response_time": 4, "confidence": 3, "is_correct": True})

        with mock.patch.object(aggregator_module, "model_status", return_value={"ready": True, "error": None}), \
             mock.patch.object(aggregator_module, "classify_features", return_value=("MEDIUM", 0.75)):
            result = aggregator.classify()
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["session_id"], "session-a")
            self.assertEqual(result["feature_count"], 11)
            aggregator.reset_for_session("session-b")
            self.assertEqual(aggregator.cv_buffer, [])
            self.assertEqual(aggregator.behaviour_buffer, [])
            self.assertEqual(aggregator.quiz_buffer, [])
            self.assertEqual(aggregator.classify()["status"], "waiting_for_features")

    def test_disconnected_live_modalities_do_not_reuse_stale_features(self):
        aggregator = aggregator_module.Aggregator()
        aggregator.reset_for_session("session-a")
        timestamp = int(time.time()) - aggregator_module.LIVE_FEATURE_MAX_AGE_SECONDS - 1
        aggregator.add_cv({"session_id": "session-a", "timestamp": timestamp, "blink_rate": 12, "gaze_score": 0.1, "expression": "NEUTRAL"})
        aggregator.add_behaviour({"session_id": "session-a", "timestamp": timestamp, "wpm": 0.2, "backspace_rate": 0.1, "scroll_reversals": 0.1, "mouse_idle": 0.5})
        aggregator.add_quiz({"session_id": "session-a", "timestamp": timestamp, "response_time": 4, "confidence": 3})
        result = aggregator.classify()
        self.assertEqual(result["status"], "waiting_for_fresh_features")
        self.assertEqual(result["stale_features"], ["cv", "behaviour"])
        self.assertIsNone(result["load_class"])


if __name__ == "__main__":
    unittest.main()
