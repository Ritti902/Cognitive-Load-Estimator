"""FastAPI ingestion, session management, prediction, and dashboard updates."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from database import (
    get_session,
    init_db,
    save_behaviour_signal,
    save_cv_signal,
    save_score,
    session_exists,
)
from .aggregator import Aggregator
from .inference import model_status

app = FastAPI(title="Cognitive Load Estimator API")
logger = logging.getLogger(__name__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

aggregator = Aggregator()
dashboard_clients: list[WebSocket] = []
active_session_id: str | None = None
classification_task: asyncio.Task | None = None


def _required_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=422, detail=f"{key} is required")
    return value.strip()


def _number(data: dict[str, Any], key: str, default: float | None = None) -> float:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise HTTPException(status_code=422, detail=f"{key} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"{key} must be a finite number") from None
    if not math.isfinite(result):
        raise HTTPException(status_code=422, detail=f"{key} must be a finite number")
    return result


def _active_session(data: dict[str, Any]) -> str:
    global active_session_id
    requested = _required_text(data, "session_id")
    if not active_session_id:
        raise HTTPException(status_code=409, detail="No active session. Create and activate a session first.")
    if requested != active_session_id:
        raise HTTPException(status_code=409, detail="Signal session_id does not match the active session")
    if not session_exists(requested):
        active_session_id = None
        aggregator.reset_for_session(None)
        raise HTTPException(status_code=409, detail="Active session no longer exists in SQLite")
    return requested


@app.on_event("startup")
async def startup() -> None:
    global classification_task
    init_db()
    if classification_task is None or classification_task.done():
        classification_task = asyncio.create_task(classification_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    global classification_task
    if classification_task and not classification_task.done():
        classification_task.cancel()
        try:
            await classification_task
        except asyncio.CancelledError:
            pass
    classification_task = None


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "active_session_id": active_session_id, "model": model_status()}


@app.post("/session/start")
async def start_session(data: dict[str, Any]) -> dict[str, Any]:
    global active_session_id
    session_id = _required_text(data, "session_id")
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    if active_session_id != session_id:
        active_session_id = session_id
        aggregator.reset_for_session(session_id)
    return {
        "status": "ok",
        "session_id": session_id,
        "student_id": session["student_id"],
        "difficulty": session["difficulty"],
        "label_source": session.get("label_source"),
        "assessment_status": session.get("assessment_status"),
    }


@app.get("/session/active")
async def get_active_session() -> dict[str, Any]:
    session = get_session(active_session_id) if active_session_id else None
    return {"active": session is not None, "session_id": active_session_id if session else None, "session": session}


@app.post("/cv")
async def receive_cv(data: dict[str, Any]) -> dict[str, str]:
    session_id = _active_session(data)
    gaze = str(data.get("gaze", "")).upper().strip()
    expression = str(data.get("expression", "")).upper().strip()
    if gaze not in {"CENTER", "LEFT", "RIGHT"}:
        raise HTTPException(status_code=422, detail="gaze must be CENTER, LEFT, or RIGHT")
    if expression not in {"NEUTRAL", "CONFUSED", "FOCUSED"}:
        raise HTTPException(status_code=422, detail="expression must be NEUTRAL, CONFUSED, or FOCUSED")
    cleaned = {
        "session_id": session_id,
        "timestamp": int(_number(data, "timestamp", time.time())),
        "blink_rate": _number(data, "blink_rate"),
        "gaze": gaze,
        "gaze_score": _number(data, "gaze_score"),
        "expression": expression,
    }
    if cleaned["blink_rate"] < 0 or cleaned["gaze_score"] < 0:
        raise HTTPException(status_code=422, detail="CV rates and scores cannot be negative")
    save_cv_signal(session_id, cleaned)
    aggregator.add_cv(cleaned)
    return {"status": "ok", "session_id": session_id}


@app.post("/behaviour")
async def receive_behaviour(data: dict[str, Any]) -> dict[str, str]:
    session_id = _active_session(data)
    cleaned = {
        "session_id": session_id,
        "timestamp": int(_number(data, "timestamp", time.time())),
        "wpm": _number(data, "wpm"),
        "backspace_rate": _number(data, "backspace_rate"),
        "scroll_reversals": _number(data, "scroll_reversals"),
        "mouse_idle": _number(data, "mouse_idle"),
    }
    if any(not 0 <= cleaned[key] <= 1 for key in ("wpm", "backspace_rate", "scroll_reversals", "mouse_idle")):
        raise HTTPException(status_code=422, detail="Behaviour inputs must use the normalized 0-1 scale")
    save_behaviour_signal(session_id, cleaned)
    aggregator.add_behaviour(cleaned)
    return {"status": "ok", "session_id": session_id}


@app.post("/quiz")
async def receive_quiz(data: dict[str, Any]) -> dict[str, str]:
    session_id = _active_session(data)
    response_time = _number(data, "response_time")
    confidence = _number(data, "confidence")
    if response_time < 0:
        raise HTTPException(status_code=422, detail="response_time cannot be negative")
    if confidence != int(confidence) or not 1 <= confidence <= 5:
        raise HTTPException(status_code=422, detail="confidence must be an integer from 1 to 5")
    is_correct = data.get("is_correct")
    if not isinstance(is_correct, bool):
        raise HTTPException(status_code=422, detail="is_correct must be a boolean")
    aggregator.add_quiz({
        "session_id": session_id,
        "response_time": response_time,
        "confidence": confidence,
        "is_correct": is_correct,
    })
    return {"status": "ok", "session_id": session_id}


@app.get("/score")
async def get_score() -> dict[str, Any]:
    return aggregator.latest_result


@app.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket) -> None:
    await ws.accept()
    dashboard_clients.append(ws)
    try:
        await ws.send_text(json.dumps(aggregator.latest_result))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in dashboard_clients:
            dashboard_clients.remove(ws)


async def classification_loop() -> None:
    while True:
        await asyncio.sleep(5)
        try:
            result = aggregator.classify()
        except Exception as exc:
            logger.exception("Classification failed")
            result = {
                "timestamp": int(time.time()),
                "status": "inference_error",
                "error": str(exc),
                "session_id": active_session_id,
                "load_class": None,
                "load_score": None,
                "model": model_status(),
            }
            aggregator.latest_result = result
        if result.get("status") == "ready":
            try:
                if not save_score(result):
                    result["persistence_error"] = "Prediction was not saved to SQLite"
            except Exception as exc:
                logger.exception("Prediction persistence failed")
                result["persistence_error"] = str(exc)
        message = json.dumps(result)
        for client in dashboard_clients.copy():
            try:
                await client.send_text(message)
            except Exception as exc:
                logger.debug("Removing disconnected dashboard client: %s", exc)
                if client in dashboard_clients:
                    dashboard_clients.remove(client)
