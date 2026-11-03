"""Create a participant session and activate it on the FastAPI service."""

from __future__ import annotations

import os
import uuid
from urllib.parse import urljoin

import requests

from database import init_db, save_session

API_URL = os.environ.get("COGNITIVE_LOAD_API_URL", "http://127.0.0.1:8000").rstrip("/")
DIFFICULTIES = {"1": "easy", "2": "medium", "3": "hard"}


def _prompt_difficulty() -> str:
    print("\nDifficulty:\n1. Easy\n2. Medium\n3. Hard")
    while True:
        choice = input("Select difficulty [1/2/3]: ").strip()
        if choice in DIFFICULTIES:
            return DIFFICULTIES[choice]
        print("Choose 1, 2, or 3.")


def create_session(student_id: str | None = None, difficulty: str | None = None, api_url: str = API_URL) -> str:
    """Persist a uniquely identified unlabeled session, then activate it via FastAPI.

    NASA-TLX is intentionally collected after the participant finishes the study;
    entering a label before collection would contaminate the ground truth.
    """
    if student_id is None:
        print("\n==============================\n   NEW COGNITIVE LOAD SESSION\n==============================")
        student_id = input("Participant ID: ").strip()
    student_id = (student_id or "").strip()
    if not student_id:
        raise ValueError("Participant ID cannot be empty")

    if difficulty is None:
        difficulty = _prompt_difficulty()
    difficulty = difficulty.strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("Difficulty must be easy, medium, or hard")

    init_db()
    session_id = f"session_{uuid.uuid4().hex}"
    save_session(session_id, student_id, difficulty)

    endpoint = urljoin(api_url.rstrip("/") + "/", "session/start")
    try:
        response = requests.post(endpoint, json={"session_id": session_id}, timeout=5)
        response.raise_for_status()
        result = response.json()
        if result.get("status") != "ok" or result.get("session_id") != session_id:
            raise RuntimeError(result.get("detail") or result.get("message") or "FastAPI did not activate the requested session")
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Session {session_id} was saved, but FastAPI activation failed at {endpoint}: {exc}. "
            f"Start FastAPI and activate it with Invoke-RestMethod -Method Post -Uri '{endpoint}' "
            f"-ContentType 'application/json' -Body '{{\"session_id\":\"{session_id}\"}}'."
        ) from exc

    print("\nSession created and activated.")
    print("Session ID :", session_id)
    print("Participant:", student_id)
    print("Difficulty :", difficulty)
    print("NASA-TLX   : submit after the study at http://127.0.0.1:5003/tlx/" + session_id)
    return session_id


if __name__ == "__main__":
    create_session()
