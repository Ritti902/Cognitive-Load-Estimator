# Cognitive Load Estimator

An existing multimodal application that combines webcam-derived CV features, browser behaviour features, and quiz responses in a FastAPI backend, SQLite database, and React dashboard. The current MLP and LOW/MEDIUM/OVERLOADED mapping are retained.

## Environment

Use Python 3.11 and Node.js 18 or newer. From this project directory, install the existing Python and frontend dependencies:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
& "C:\Users\ritik\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Set-Location "Web App"
npm ci
Set-Location ..
```

The application database is `dataset\cognitive_load.db`, anchored to this project regardless of the service working directory. The similarly named files in the parent folder are not read by the app.

## Run the application

Start each Python command in its own PowerShell window. In each window, activate the environment before launching the service. Start FastAPI first:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python -m uvicorn model.server:app --host 127.0.0.1 --port 8000
```

Then start the behaviour bridge, quiz, and post-study NASA-TLX service:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python behaviour_pipeline\bridge.py
```

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python quiz.py
```

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python nasa_tlx.py
```

Create and activate a new session. This prompts for participant ID and difficulty, stores the new unlabeled session, then activates it through FastAPI:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python collect_sessions.py
```

After activating a session, run the webcam pipeline in a separate window (camera access is required):

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python cv_pipeline\main.py
```

Run the existing React dashboard:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main\Web App"
npm start
```

The dashboard reads the active session and participant from FastAPI, starts behaviour capture for that ID, displays quiz questions, and shows a prediction only when all three modalities and a valid ONNX model are available. Complete the quiz during the session. Afterward use the dashboard's session-bound NASA-TLX link (or the URL printed by `collect_sessions.py`), check the displayed participant and session, and submit the participant's actual ratings. A successful submission stores `assessment_status='submitted'` and `label_source='NASA-TLX'`; duplicate submissions are rejected. The dashboard shows pending, missing, submitted, or unverified assessment state.

## Dataset and model

Build the session-level CSV and its validation report:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python dataset_builder.py
```

The output files are `dataset\training_dataset.csv` and `dataset\dataset_validation_report.json`. The builder admits only complete sessions with `assessment_status='submitted'`, source `NASA-TLX`, and a score and class consistent with the existing score mapping. It ignores model predictions in `scores`, missing or unverified legacy labels, and the unrelated parent-folder test CSV. The report includes assessment-status and label-source counts plus per-session exclusion reasons. `f10` and `f11` remain documented zero-valued reserved inputs because no measured features are assigned to those model slots.

| Model input | Definition used by collection/training |
| --- | --- |
| `blink_rate` | Blinks per minute from the webcam stream |
| `gaze_score` | Absolute normalized iris offset from eye center; the categorical `gaze` direction is persisted alongside it |
| `expression` | `NEUTRAL=0`, `CONFUSED=1`, `FOCUSED=2` |
| `wpm` | Browser letters-to-words estimate, normalized by 120 WPM |
| `backspace_rate` | Backspaces divided by key events, 0–1 |
| `scroll_reversals` | Reversals in the current capture window, normalized by 20 |
| `mouse_idle` | Idle seconds, normalized by 60 |
| `response_time` | Seconds between displaying a question and submitting its answer |
| `confidence` | Participant rating from 1 to 5 |
| `f10`, `f11` | Reserved model compatibility slots; fixed at 0 because no validated observations map to them |

Train with participant-disjoint train, validation, and test splits, then export and check ONNX against PyTorch:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python -m model.train
python -m model.export_onnx
```

Training writes `model\fusion_model.pth`, `model\preprocessing.json`, and `model\evaluation_report.json`. ONNX export writes `model\fusion_model.onnx` and `model\onnx_validation.json`. Restart FastAPI after a successful export so it loads the model. Training and export intentionally stop if the dataset is incomplete, missing classes, or lacks enough independent participant groups for a held-out evaluation.

## Verification

Run backend regression checks and the existing React test/build:

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main"
.\venv\Scripts\Activate.ps1
python -m compileall -q database.py collect_sessions.py dataset_builder.py quiz.py nasa_tlx.py cv_pipeline model behaviour_pipeline tests
python -m unittest discover -s tests -v
```

```powershell
Set-Location "C:\Users\ritik\OneDrive\Desktop\Cognitive-Load_Estimator\Cognitive-Load-Estimator--main\Web App"
npm test -- --watchAll=false
npm run build
```

Inspect service and session state:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/session/active
Invoke-RestMethod http://127.0.0.1:8000/score
```

## Data and evaluation status

The current project database contains 4 historical sessions, 1,558 CV rows, 724 behaviour rows, 5 quiz responses, and 2,264 stored prediction rows. The attached report cited 1,372 CV rows; the current database has 186 more. Three historical sessions have stored scores or labels without submission provenance, and one has no assessment. None has a verified post-study NASA-TLX submission, so the dataset contains 0 usable sessions and model training is not justified. The report lists missing and unverified assessments separately, along with per-session feature exclusions and raw invalid-value counts. Existing records were preserved; additive migrations do not promote historical scores to verified labels.

For a meaningful three-class evaluation, collect complete sessions with genuine post-study NASA-TLX forms from multiple distinct participants in each class (the report uses 10 participants per class as its minimum evaluation-support threshold). The webcam hardware and live service flow must be checked on the participant machine; no webcam or real study session was fabricated for this verification.
