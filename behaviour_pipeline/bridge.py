"""Flask-SocketIO bridge from anonymous browser metrics to FastAPI."""

from __future__ import annotations

import os

import requests
from flask import Flask
from flask_socketio import SocketIO

try:
    from .features import validate_and_normalise
except ImportError:  # Direct execution: python behaviour_pipeline\bridge.py
    from features import validate_and_normalise

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
FASTAPI_URL = os.environ.get("COGNITIVE_LOAD_API_URL", "http://127.0.0.1:8000").rstrip("/")


@socketio.on("behaviour_features")
def handle_features(data):
    if not isinstance(data, dict):
        return {"status": "error", "message": "Expected a feature object"}
    try:
        clean_data = validate_and_normalise(data)
    except (TypeError, ValueError) as exc:
        return {"status": "error", "message": str(exc)}
    try:
        response = requests.post(f"{FASTAPI_URL}/behaviour", json=clean_data, timeout=3)
        if response.status_code >= 400:
            return {"status": "error", "message": f"FastAPI returned {response.status_code}: {response.text[:200]}"}
    except requests.RequestException as exc:
        return {"status": "error", "message": f"FastAPI is unavailable: {exc}"}
    return {"status": "ok", "session_id": clean_data["session_id"]}


if __name__ == "__main__":
    print("Behaviour bridge running on http://127.0.0.1:5001")
    socketio.run(app, host="127.0.0.1", port=5001, debug=False)
