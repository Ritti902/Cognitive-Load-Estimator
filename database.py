"""SQLite persistence for the Cognitive Load Estimator.

The database path is anchored to this project, so starting a service from a
different working directory cannot silently create a second database.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("COGNITIVE_LOAD_DB", PROJECT_DIR / "dataset" / "cognitive_load.db"))
NASA_FIELDS = ("mental", "physical", "temporal", "performance", "effort", "frustration")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db() -> None:
    """Create the existing schema and apply non-destructive additive migrations."""
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            student_id TEXT,
            date TEXT,
            difficulty TEXT,
            nasa_tlx_score REAL,
            load_class TEXT,
            assessment_status TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS quiz_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question_id INTEGER,
            response_time REAL,
            is_correct INTEGER,
            confidence INTEGER,
            timestamp INTEGER
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS cv_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp INTEGER,
            blink_rate REAL,
            gaze TEXT,
            expression TEXT
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS behaviour_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp INTEGER,
            wpm REAL,
            backspace_rate REAL,
            scroll_reversals REAL,
            mouse_idle REAL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS scores (
            timestamp INTEGER,
            session_id TEXT,
            load_class TEXT,
            load_score REAL
        )""")

        # Additive migrations preserve existing session and signal records.
        migrations = {
            "sessions": {"label_source": "TEXT", "assessment_status": "TEXT"},
            "cv_signals": {"gaze_score": "REAL"},
            "quiz_responses": {"attempt_id": "TEXT"},
        }
        added_assessment_status = False
        for table, columns in migrations.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, sql_type in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
                    if table == "sessions" and column == "assessment_status":
                        added_assessment_status = True
        if added_assessment_status:
            # Existing scores and labels have no auditable submission record.
            # Keep them unverified, while marking genuinely empty legacy rows missing.
            conn.execute("""UPDATE sessions
                SET assessment_status = CASE
                    WHEN nasa_tlx_score IS NULL AND load_class IS NULL AND label_source IS NULL
                    THEN 'missing' ELSE 'unverified' END
                WHERE assessment_status IS NULL""")
        conn.execute("""CREATE UNIQUE INDEX IF NOT EXISTS ux_quiz_attempt_id
            ON quiz_responses(attempt_id) WHERE attempt_id IS NOT NULL""")


def save_session(session_id: str, student_id: str, difficulty: str) -> None:
    if not session_id or not student_id or difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("A session ID, participant ID, and valid difficulty are required")
    init_db()
    with connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, student_id, date, difficulty, nasa_tlx_score, load_class, label_source, assessment_status) "
            "VALUES (?, ?, ?, ?, NULL, NULL, NULL, 'pending')",
            (session_id, student_id, time.strftime("%Y-%m-%dT%H:%M:%S%z"), difficulty),
        )


def get_session(session_id: str) -> dict[str, Any] | None:
    if not session_id:
        return None
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row) if row else None


def session_exists(session_id: str) -> bool:
    return get_session(session_id) is not None


def save_quiz_response(
    session_id: str,
    question_id: int,
    response_time: float,
    is_correct: bool,
    confidence: int,
    attempt_id: str,
) -> None:
    if not session_exists(session_id):
        raise ValueError("Unknown session ID")
    with connect() as conn:
        conn.execute(
            "INSERT INTO quiz_responses "
            "(session_id, question_id, response_time, is_correct, confidence, timestamp, attempt_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, question_id, response_time, int(is_correct), confidence, int(time.time()), attempt_id),
        )


def delete_quiz_response(attempt_id: str) -> None:
    """Roll back a just-recorded answer if FastAPI cannot accept its features."""
    with connect() as conn:
        conn.execute("DELETE FROM quiz_responses WHERE attempt_id = ?", (attempt_id,))


def save_nasa_tlx(session_id: str, scores: dict[str, Any]) -> tuple[float, str]:
    if set(scores) != set(NASA_FIELDS):
        raise ValueError("All six NASA-TLX ratings are required")
    if any(isinstance(scores[key], bool) for key in NASA_FIELDS):
        raise ValueError("NASA-TLX ratings must be numbers from 0 to 20")
    try:
        values = {key: float(scores[key]) for key in NASA_FIELDS}
    except (TypeError, ValueError) as exc:
        raise ValueError("NASA-TLX ratings must be numbers from 0 to 20") from exc
    if any(not 0 <= value <= 20 for value in values.values()):
        raise ValueError("NASA-TLX ratings must be numbers from 0 to 20")

    nasa_score = sum(values.values()) / len(NASA_FIELDS) * 5
    load_class = "LOW" if nasa_score <= 33 else "MEDIUM" if nasa_score <= 66 else "OVERLOADED"
    with connect() as conn:
        row = conn.execute(
            "SELECT nasa_tlx_score, assessment_status FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not row:
            raise ValueError("Unknown session ID")
        if row["assessment_status"] == "submitted" or row["nasa_tlx_score"] is not None:
            raise ValueError("NASA-TLX has already been submitted for this session")
        conn.execute(
            "UPDATE sessions SET nasa_tlx_score = ?, load_class = ?, label_source = 'NASA-TLX', "
            "assessment_status = 'submitted' WHERE id = ?",
            (nasa_score, load_class, session_id),
        )
    return nasa_score, load_class


def save_cv_signal(session_id: str, data: dict[str, Any]) -> None:
    if not session_exists(session_id):
        raise ValueError("Unknown session ID")
    with connect() as conn:
        conn.execute(
            "INSERT INTO cv_signals "
            "(session_id, timestamp, blink_rate, gaze, gaze_score, expression) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, int(data["timestamp"]), float(data["blink_rate"]), data["gaze"],
             float(data["gaze_score"]), data["expression"]),
        )


def save_behaviour_signal(session_id: str, data: dict[str, Any]) -> None:
    if not session_exists(session_id):
        raise ValueError("Unknown session ID")
    with connect() as conn:
        conn.execute(
            "INSERT INTO behaviour_signals "
            "(session_id, timestamp, wpm, backspace_rate, scroll_reversals, mouse_idle) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, int(data["timestamp"]), float(data["wpm"]), float(data["backspace_rate"]),
             float(data["scroll_reversals"]), float(data["mouse_idle"])),
        )


def save_score(result: dict[str, Any]) -> bool:
    """Persist only actual model predictions with a valid session association."""
    if result.get("status") != "ready" or result.get("load_class") not in {"LOW", "MEDIUM", "OVERLOADED"}:
        return False
    session_id = result.get("session_id")
    if not session_exists(session_id):
        return False
    with connect() as conn:
        conn.execute(
            "INSERT INTO scores (timestamp, session_id, load_class, load_score) VALUES (?, ?, ?, ?)",
            (int(result["timestamp"]), session_id, result["load_class"], float(result["load_score"])),
        )
    return True
