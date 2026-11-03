"""Quiz and NASA-TLX request validation with isolated SQLite state."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import database
import collect_sessions
import dataset_builder
import nasa_tlx
import quiz
from behaviour_pipeline import bridge
from fastapi.testclient import TestClient
from model import server


class FlaskStudyFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "study.db"
        self.db_patch = mock.patch.object(database, "DB_PATH", self.db_path)
        self.db_patch.start()
        database.init_db()
        database.save_session("session-a", "participant-a", "easy")
        database.save_session("session-b", "participant-b", "easy")
        quiz.question_attempts.clear()
        self.quiz_client = quiz.app.test_client()
        self.tlx_client = nasa_tlx.app.test_client()

    def tearDown(self):
        quiz.question_attempts.clear()
        self.db_patch.stop()
        self.temp_dir.cleanup()

    @staticmethod
    def _active_response(session_id):
        response = mock.Mock()
        response.json.return_value = {"session_id": session_id}
        response.raise_for_status.return_value = None
        return response

    def test_question_attempts_are_isolated_and_single_use(self):
        with mock.patch.object(quiz.requests, "get", return_value=self._active_response("session-a")):
            question_a = self.quiz_client.get("/question/easy/0?session_id=session-a")
        with mock.patch.object(quiz.requests, "get", return_value=self._active_response("session-b")):
            question_b = self.quiz_client.get("/question/easy/0?session_id=session-b")
        self.assertEqual(question_a.status_code, 200)
        self.assertEqual(question_b.status_code, 200)
        self.assertNotIn("answer", question_a.json)
        self.assertNotEqual(question_a.json["attempt_id"], question_b.json["attempt_id"])

        body = {
            "session_id": "session-b", "difficulty": "easy", "index": 0,
            "chosen_option": 0, "confidence": 4, "attempt_id": question_b.json["attempt_id"],
        }
        with mock.patch.object(quiz.requests, "get", return_value=self._active_response("session-b")), \
             mock.patch.object(quiz.requests, "post", return_value=self._active_response("session-b")):
            submitted = self.quiz_client.post("/answer", json=body)
            repeated = self.quiz_client.post("/answer", json=body)
            stale_session = self.quiz_client.post("/answer", json={**body, "session_id": "session-a"})
        self.assertEqual(submitted.status_code, 200)
        self.assertTrue(submitted.json["is_correct"])
        self.assertGreaterEqual(submitted.json["response_time"], 0)
        self.assertEqual(repeated.status_code, 409)
        self.assertEqual(stale_session.status_code, 409)

        connection = sqlite3.connect(self.db_path)
        try:
            rows = connection.execute("SELECT session_id, COUNT(*) FROM quiz_responses GROUP BY session_id").fetchall()
        finally:
            connection.close()
        self.assertEqual(rows, [("session-b", 1)])

    def test_quiz_rejects_missing_session_and_invalid_confidence(self):
        self.assertEqual(self.quiz_client.get("/question/easy/0").status_code, 400)
        with mock.patch.object(quiz.requests, "get", return_value=self._active_response("session-a")):
            question = self.quiz_client.get("/question/easy/0?session_id=session-a").json
        invalid = self.quiz_client.post("/answer", json={
            "session_id": "session-a", "difficulty": "easy", "index": 0,
            "chosen_option": 0, "confidence": 9, "attempt_id": question["attempt_id"],
        })
        self.assertEqual(invalid.status_code, 400)

    def test_socketio_bridge_normalizes_and_forwards_session_bound_features(self):
        client = bridge.socketio.test_client(bridge.app)
        response = mock.Mock(status_code=200, text="ok")
        with mock.patch.object(bridge.requests, "post", return_value=response) as post:
            acknowledgement = client.emit("behaviour_features", {
                "session_id": "session-a", "timestamp": 100, "wpm": 240,
                "backspace_rate": 0.25, "scroll_reversals": 30, "mouse_idle_sec": 120,
            }, callback=True)
        self.assertEqual(acknowledgement, {"status": "ok", "session_id": "session-a"})
        self.assertEqual(post.call_args.args[0], "http://127.0.0.1:8000/behaviour")
        forwarded = post.call_args.kwargs["json"]
        self.assertEqual(forwarded["session_id"], "session-a")
        self.assertEqual(forwarded["wpm"], 1.0)
        self.assertEqual(forwarded["backspace_rate"], 0.25)
        self.assertEqual(forwarded["scroll_reversals"], 1.0)
        self.assertEqual(forwarded["mouse_idle"], 1.0)
        client.disconnect()

    def test_nasa_tlx_is_validated_and_cannot_be_submitted_twice(self):
        ratings = {key: 10 for key in database.NASA_FIELDS}
        with self.tlx_client as client:
            form = client.get("/tlx/session-a")
            self.assertEqual(form.status_code, 200)
            self.assertIn(b"session-a", form.data)
            self.assertIn(b"participant-a", form.data)
            self.assertNotIn(b'value="10"', form.data)
            first = client.post("/submit_tlx", json={"session_id": "session-a", **ratings})
            completed_form = client.get("/tlx/session-a")
            second = client.post("/submit_tlx", json={"session_id": "session-a", **ratings})
            invalid = client.post("/submit_tlx", json={"session_id": "session-b", **{**ratings, "mental": 21}})
            unknown = client.post("/submit_tlx", json={"session_id": "missing", **ratings})
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json["nasa_score"], 50)
        self.assertEqual(first.json["load_class"], "MEDIUM")
        self.assertEqual(first.json["session_id"], "session-a")
        self.assertEqual(first.json["assessment_status"], "submitted")
        self.assertEqual(first.json["label_source"], "NASA-TLX")
        self.assertIn(b"already submitted", completed_form.data)
        self.assertNotIn(b"id=\"tlxForm\"", completed_form.data)
        saved = database.get_session("session-a")
        self.assertEqual(saved["student_id"], "participant-a")
        self.assertEqual(saved["assessment_status"], "submitted")
        self.assertEqual(saved["label_source"], "NASA-TLX")
        self.assertEqual(second.status_code, 409)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(unknown.status_code, 404)
        self.assertEqual(database.get_session("session-b")["assessment_status"], "pending")

    def test_session_activation_assessment_and_dataset_trace_as_one_flow(self):
        server.active_session_id = None
        server.aggregator.reset_for_session(None)
        output_csv = Path(self.temp_dir.name) / "training_dataset.csv"
        report_path = Path(self.temp_dir.name) / "dataset_validation_report.json"
        try:
            with TestClient(server.app) as api:
                def activate(url, json, timeout):
                    response = api.post("/session/start", json=json)
                    wrapped = mock.Mock()
                    wrapped.json.return_value = response.json()
                    wrapped.raise_for_status.side_effect = None if response.status_code < 400 else RuntimeError("activation failed")
                    return wrapped

                with mock.patch.object(collect_sessions.requests, "post", side_effect=activate):
                    session_id = collect_sessions.create_session("participant-c", "medium", "http://fastapi.test")

                active = api.get("/session/active").json()
                self.assertEqual(active["session_id"], session_id)
                self.assertEqual(active["session"]["student_id"], "participant-c")
                form = self.tlx_client.get(f"/tlx/{session_id}")
                self.assertIn(session_id.encode(), form.data)
                self.assertIn(b"participant-c", form.data)

                ratings = {key: 10 for key in database.NASA_FIELDS}
                submitted = self.tlx_client.post("/submit_tlx", json={"session_id": session_id, **ratings})
                self.assertEqual(submitted.status_code, 200)
                saved = database.get_session(session_id)
                self.assertEqual(saved["student_id"], "participant-c")
                self.assertEqual(saved["assessment_status"], "submitted")
                self.assertEqual(saved["label_source"], "NASA-TLX")
                self.assertEqual(saved["nasa_tlx_score"], 50.0)
                self.assertEqual(saved["load_class"], "MEDIUM")

                database.save_cv_signal(session_id, {
                    "timestamp": 100, "blink_rate": 12, "gaze": "CENTER",
                    "gaze_score": 0.1, "expression": "NEUTRAL",
                })
                database.save_behaviour_signal(session_id, {
                    "timestamp": 100, "wpm": 0.5, "backspace_rate": 0.2,
                    "scroll_reversals": 0.1, "mouse_idle": 0.4,
                })
                database.save_quiz_response(session_id, 1, 8.0, True, 4, "attempt-c")

                with mock.patch.object(dataset_builder, "DB_PATH", self.db_path), \
                     mock.patch.object(dataset_builder, "OUTPUT_CSV", output_csv), \
                     mock.patch.object(dataset_builder, "REPORT_JSON", report_path):
                    frame = dataset_builder.build_dataset()
                row = frame.loc[frame["session_id"] == session_id].iloc[0]
                self.assertEqual(row["student_id"], "participant-c")
                self.assertEqual(row["assessment_status"], "submitted")
                self.assertEqual(row["load_label"], "MEDIUM")
                self.assertEqual(row["nasa_tlx_score"], 50.0)
                self.assertNotIn("session-a", frame["session_id"].tolist())
        finally:
            server.active_session_id = None
            server.aggregator.reset_for_session(None)


if __name__ == "__main__":
    unittest.main()
