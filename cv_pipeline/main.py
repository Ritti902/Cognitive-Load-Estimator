"""Webcam feature extraction and session-aware forwarding to FastAPI."""

from __future__ import annotations

import asyncio
import os
import threading
import time

import cv2
import mediapipe as mp
import requests

try:
    from .blink_detector import BlinkCounter, calculate_ear
    from .expression_classifier import extract_expression_features, load_or_default_classifier, predict_expression
    from .gaze_detector import get_gaze_direction
except ImportError:  # Direct execution: python cv_pipeline\main.py
    from blink_detector import BlinkCounter, calculate_ear
    from expression_classifier import extract_expression_features, load_or_default_classifier, predict_expression
    from gaze_detector import get_gaze_direction

FASTAPI_URL = os.environ.get("COGNITIVE_LOAD_API_URL", "http://127.0.0.1:8000").rstrip("/")
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

latest_features: dict = {}
_feature_generation = 0


def run_cv_pipeline() -> None:
    global latest_features, _feature_generation
    print("Initializing MediaPipe Face Mesh...")
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    blink_counter = BlinkCounter()
    classifier = load_or_default_classifier()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check camera permissions and close other camera apps.")
        face_mesh.close()
        return

    print("Webcam opened. Waiting for a face; no frames are stored or sent.")
    start_time = time.monotonic()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("WARNING: Webcam frame read failed; retrying.")
                time.sleep(0.5)
                continue
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                time.sleep(0.1)
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            blink_counter.update(left_ear, right_ear)
            blink_rate = blink_counter.get_blinks_per_minute(time.monotonic() - start_time)
            gaze, gaze_score = get_gaze_direction(landmarks)
            expression = predict_expression(extract_expression_features(landmarks), classifier)

            _feature_generation += 1
            latest_features = {
                "timestamp": int(time.time()),
                "blink_rate": round(float(blink_rate), 3),
                "gaze": gaze,
                "gaze_score": round(float(gaze_score), 5),
                "expression": expression,
                "left_ear": round(float(left_ear), 5),
                "right_ear": round(float(right_ear), 5),
                "_generation": _feature_generation,
            }
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping webcam capture.")
    finally:
        cap.release()
        face_mesh.close()


async def send_features() -> None:
    print("CV sender started; frames are sent only while a session is active.")
    last_session_id: str | None = None
    last_generation = 0
    while True:
        try:
            response = requests.get(f"{FASTAPI_URL}/session/active", timeout=3)
            response.raise_for_status()
            session_id = response.json().get("session_id")
            if not session_id:
                last_session_id = None
                await asyncio.sleep(1)
                continue
            if session_id != last_session_id:
                # A frame captured before a session switch must not become that
                # session's first observation. Wait for the next fresh frame.
                last_session_id = session_id
                last_generation = _feature_generation
                await asyncio.sleep(1)
                continue
            current = dict(latest_features)
            generation = current.pop("_generation", 0)
            if current and generation > last_generation:
                current["session_id"] = session_id
                sent = requests.post(f"{FASTAPI_URL}/cv", json=current, timeout=3)
                sent.raise_for_status()
                last_generation = generation
        except requests.RequestException as exc:
            print(f"FastAPI CV connection problem: {exc}")
        except Exception as exc:
            print(f"CV sender error: {exc}")
        await asyncio.sleep(1)


if __name__ == "__main__":
    print("COGNITIVE LOAD ESTIMATOR - CV")
    threading.Thread(target=run_cv_pipeline, daemon=True).start()
    asyncio.run(send_features())
