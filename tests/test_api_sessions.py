"""FastAPI integration checks for activation and per-session persistence."""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import database
from fastapi.testclient import TestClient
from model import server


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "api.db"
        self.db_patch = mock.patch.object(database, "DB_PATH", self.db_path)
        self.db_patch.start()
        database.init_db()
        database.save_session("session-a", "participant-a", "easy")
        database.save_session("session-b", "participant-b", "medium")
        server.active_session_id = None
        server.aggregator.reset_for_session(None)
        self.client_context = TestClient(server.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        server.active_session_id = None
        server.aggregator.reset_for_session(None)
        self.db_patch.stop()
        self.temp_dir.cleanup()

    def test_start_rejects_missing_and_unknown_ids(self):
        self.assertEqual(self.client.post("/session/start", json={}).status_code, 422)
        self.assertEqual(self.client.post("/session/start", json={"session_id": "unknown"}).status_code, 404)
        self.assertIsNone(server.active_session_id)

    def test_signal_persistence_and_switching_sessions(self):
        cv = {"session_id": "session-a", "timestamp": int(time.time()), "blink_rate": 12, "gaze": "CENTER", "gaze_score": 0.1, "expression": "NEUTRAL"}
        self.assertEqual(self.client.post("/cv", json=cv).status_code, 409)
        activated = self.client.post("/session/start", json={"session_id": "session-a"})
        self.assertEqual(activated.status_code, 200)
        self.assertEqual(activated.json()["assessment_status"], "pending")
        self.assertEqual(self.client.get("/session/active").json()["session_id"], "session-a")
        self.assertEqual(self.client.post("/cv", json={**cv, "session_id": "session-b"}).status_code, 409)
        self.assertEqual(self.client.post("/cv", json=cv).status_code, 200)

        behavior = {
            "session_id": "session-a", "timestamp": int(time.time()), "wpm": 0.5,
            "backspace_rate": 0.25, "scroll_reversals": 0.1, "mouse_idle": 0.4,
        }
        self.assertEqual(self.client.post("/behaviour", json=behavior).status_code, 200)
        self.assertEqual(self.client.post("/behaviour", json={**behavior, "wpm": 12}).status_code, 422)
        quiz = {"session_id": "session-a", "response_time": 5, "confidence": 4, "is_correct": True}
        self.assertEqual(self.client.post("/quiz", json=quiz).status_code, 200)
        self.assertEqual(self.client.post("/quiz", json={**quiz, "confidence": 8}).status_code, 422)

        server.aggregator.classify()
        self.assertEqual(server.aggregator.latest_result["status"], "model_unavailable")
        self.assertIsNone(server.aggregator.latest_result["load_class"])

        self.assertEqual(self.client.post("/session/start", json={"session_id": "session-b"}).status_code, 200)
        self.assertEqual(server.aggregator.cv_buffer, [])
        self.assertEqual(self.client.post("/cv", json=cv).status_code, 409)
        self.assertEqual(self.client.post("/cv", json={**cv, "session_id": "session-b"}).status_code, 200)

        connection = sqlite3.connect(self.db_path)
        cv_rows = connection.execute("SELECT session_id, COUNT(*) FROM cv_signals GROUP BY session_id ORDER BY session_id").fetchall()
        behavior_rows = connection.execute("SELECT session_id, COUNT(*) FROM behaviour_signals GROUP BY session_id").fetchall()
        connection.close()
        self.assertEqual(cv_rows, [("session-a", 1), ("session-b", 1)])
        self.assertEqual(behavior_rows, [("session-a", 1)])


if __name__ == "__main__":
    unittest.main()
