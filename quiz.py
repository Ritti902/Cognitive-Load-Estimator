"""Flask quiz service with per-attempt timing and session-bound submissions."""

from __future__ import annotations

import os
import threading
import time
import uuid
from typing import Any

import requests
from flask import Flask, jsonify, request

from database import delete_quiz_response, save_quiz_response, session_exists

app = Flask(__name__)
API_URL = os.environ.get("COGNITIVE_LOAD_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Keep the existing question bank and answer key. Only the public question omits its answer.
QUESTIONS = {
    "easy": [
        {"id": 1, "text": "What does AI stand for?", "options": ["Artificial Intelligence", "Auto Input", "Automated Interface", "None"], "answer": 0},
        {"id": 2, "text": "Which language is used for ML?", "options": ["HTML", "Python", "CSS", "Excel"], "answer": 1},
    ],
    "medium": [
        {"id": 3, "text": "What is overfitting in ML?", "options": ["Model too complex", "Model too simple", "Low accuracy", "Fast training"], "answer": 0},
    ],
    "hard": [
        {"id": 4, "text": "What does gradient descent minimise?", "options": ["Loss function", "Accuracy", "Epoch count", "Layer depth"], "answer": 0},
    ],
}

question_attempts: dict[str, dict[str, Any]] = {}
attempts_lock = threading.Lock()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def _active_session(session_id: str) -> bool:
    if not session_exists(session_id):
        return False
    try:
        response = requests.get(f"{API_URL}/session/active", timeout=3)
        response.raise_for_status()
        return response.json().get("session_id") == session_id
    except requests.RequestException as exc:
        raise RuntimeError(f"FastAPI session status is unavailable: {exc}") from exc


def _json_body() -> dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


@app.get("/question/<difficulty>/<int:index>")
def get_question(difficulty: str, index: int):
    if difficulty not in QUESTIONS:
        return jsonify({"error": "difficulty must be easy, medium, or hard"}), 400
    session_id = request.args.get("session_id", "").strip()
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not session_exists(session_id):
        return jsonify({"error": "Unknown session_id"}), 404
    try:
        if not _active_session(session_id):
            return jsonify({"error": "This session is not active"}), 409
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    questions = QUESTIONS[difficulty]
    if index < 0 or index >= len(questions):
        return jsonify({"done": True})
    q = questions[index]
    attempt_id = uuid.uuid4().hex
    with attempts_lock:
        # Expire abandoned attempts after two hours to bound in-memory state.
        now = time.monotonic()
        for key, value in list(question_attempts.items()):
            if now - value["started"] > 7200:
                question_attempts.pop(key, None)
        question_attempts[attempt_id] = {
            "session_id": session_id,
            "difficulty": difficulty,
            "index": index,
            "started": now,
        }
    return jsonify({"id": q["id"], "text": q["text"], "options": q["options"], "attempt_id": attempt_id})


@app.post("/answer")
def submit_answer():
    data = _json_body()
    session_id = data.get("session_id")
    difficulty = data.get("difficulty")
    index = data.get("index")
    chosen = data.get("chosen_option")
    confidence = data.get("confidence")
    attempt_id = data.get("attempt_id")
    if not isinstance(session_id, str) or not session_id.strip():
        return jsonify({"error": "session_id is required"}), 400
    if difficulty not in QUESTIONS:
        return jsonify({"error": "difficulty must be easy, medium, or hard"}), 400
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(QUESTIONS[difficulty]):
        return jsonify({"error": "index is outside the selected question set"}), 400
    if isinstance(chosen, bool) or not isinstance(chosen, int) or not 0 <= chosen < len(QUESTIONS[difficulty][index]["options"]):
        return jsonify({"error": "chosen_option must be a valid option index"}), 400
    if isinstance(confidence, bool) or not isinstance(confidence, int) or not 1 <= confidence <= 5:
        return jsonify({"error": "confidence must be an integer from 1 to 5"}), 400
    if not isinstance(attempt_id, str) or not attempt_id:
        return jsonify({"error": "attempt_id is required; retrieve the question first"}), 400

    with attempts_lock:
        attempt = question_attempts.get(attempt_id)
        if attempt is None:
            return jsonify({"error": "Question attempt is unknown, expired, or already submitted"}), 409
        if (attempt["session_id"], attempt["difficulty"], attempt["index"]) != (session_id, difficulty, index):
            return jsonify({"error": "Answer does not match the issued question attempt"}), 409
        # Consume before persistence so concurrent duplicate submissions cannot both pass.
        question_attempts.pop(attempt_id)
        response_time = round(max(0.0, time.monotonic() - attempt["started"]), 3)

    try:
        if not _active_session(session_id):
            return jsonify({"error": "This session is not active"}), 409
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

    question = QUESTIONS[difficulty][index]
    is_correct = chosen == question["answer"]
    inserted = False
    try:
        save_quiz_response(session_id, question["id"], response_time, is_correct, confidence, attempt_id)
        inserted = True
        response = requests.post(
            f"{API_URL}/quiz",
            json={"session_id": session_id, "response_time": response_time, "confidence": confidence, "is_correct": is_correct},
            timeout=3,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        if inserted:
            delete_quiz_response(attempt_id)
        return jsonify({"error": f"FastAPI did not accept quiz features: {exc}"}), 503
    except Exception as exc:
        # Integrity errors include repeated attempt IDs; no answer is overwritten.
        if inserted:
            delete_quiz_response(attempt_id)
        return jsonify({"error": str(exc)}), 409

    return jsonify({"response_time": response_time, "is_correct": is_correct, "correct_answer": question["answer"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
