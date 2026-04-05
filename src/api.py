# src/api.py
"""
FastAPI inference server for SMS spam detection.

Run (from project root):
    uvicorn src.api:app --reload --port 8000

Endpoints:
    POST /predict   — classify a single SMS message
    GET  /health    — liveness check
    GET  /model     — metadata about the loaded model

CORS is configured to allow the local Vite dev server (localhost:5173)
and common localhost ports for development.  For production, replace
allow_origins with your actual domain or load it from an environment variable.
"""

import json
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BEST_PIPELINE_PATH, METRICS_PATH
from src.predict import predict as _predict, load_pipeline


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SMS Spam Detection API",
    description=(
        "Classical NLP spam classifier using a scikit-learn pipeline "
        "(TextPreprocessor → Vectorizer → Classifier). "
        "Best pipeline selected by F1 on a stratified 5-fold CV."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy pipeline loading — loaded once on first request, not at import time
# ---------------------------------------------------------------------------

_pipeline = None
_model_meta: dict = {}


def get_pipeline():
    global _pipeline, _model_meta
    if _pipeline is None:
        if not BEST_PIPELINE_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model pipeline not found. "
                    "Please run `python -m src.train` first."
                ),
            )
        _pipeline = load_pipeline(BEST_PIPELINE_PATH)
        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, encoding="utf-8") as fh:
                    _model_meta = json.load(fh)
            except Exception:
                _model_meta = {}
    return _pipeline


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        example="Free entry in 2 a wkly comp to win FA Cup final tkts!",
        description="Raw SMS message text to classify.",
    )


class PredictResponse(BaseModel):
    text: str
    predicted_label: str           # "ham" or "spam"
    predicted_label_index: int     # 0 = ham, 1 = spam
    score: Optional[float]         # model-estimated score, or null
    score_type: str                # "probability" | "decision_score" | "not_available"
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    selected_pipeline_name: str
    best_model_name: str
    best_vectorizer: str
    cv_mean_f1: Optional[float]
    test_f1: Optional[float]
    test_accuracy: Optional[float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness check. Also reports whether the trained model file exists."""
    return {
        "status": "ok",
        "model_loaded": BEST_PIPELINE_PATH.exists(),
    }


@app.get("/model", response_model=ModelInfoResponse, tags=["meta"])
def model_info():
    """Return metadata about the selected pipeline from metrics.json."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="metrics.json not found. Run train.py first.",
        )
    with open(METRICS_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    return {
        "selected_pipeline_name": meta.get("selected_pipeline_name", "unknown"),
        "best_model_name":        meta.get("best_model_name", "unknown"),
        "best_vectorizer":        meta.get("best_vectorizer", "unknown"),
        "cv_mean_f1":             meta.get("cv_mean_f1"),
        "test_f1":                meta.get("test_f1"),
        "test_accuracy":          meta.get("test_accuracy"),
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    """
    Classify a single SMS message as spam or ham.

    **Score semantics:**
    - LogisticRegression or MultinomialNB: `score` is the model-estimated
      P(spam) and `score_type` is `"probability"`.
    - LinearSVC: `score` is the raw decision-function value and `score_type`
      is `"decision_score"`. Higher values indicate stronger spam signal,
      but this is **not** a calibrated probability.
    """
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty or whitespace only.")

    pipeline = get_pipeline()
    try:
        result = _predict(req.text, pipeline=pipeline)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


# ---------------------------------------------------------------------------
# Run with: uvicorn src.api:app --reload
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
