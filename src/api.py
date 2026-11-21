"""
Fraud scoring API.

Run locally from the project folder:
    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
import json
import math
import os
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "predictions.jsonl"

# Load once when the server starts, not on every request
model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
meta = json.loads((MODEL_DIR / "metadata.json").read_text())

FEATURES: list[str] = meta["features"]
CATEGORIES: dict[str, list[str]] = meta["categorical"]
THRESHOLD: float = meta["threshold"]
REFERENCE = meta.get("reference_scores")

# Recent predictions kept in memory for the /monitor endpoint
RECENT = deque(maxlen=5000)
MIN_FOR_DRIFT = 100


class Transaction(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Feature name -> value. Leave out or send null for missing values.",
        examples=[{"TransactionAmt": 117.0, "card1": 13926, "ProductCD": "W"}],
    )


class Reason(BaseModel):
    feature: str
    value: Any
    impact: float


class Prediction(BaseModel):
    request_id: str
    fraud_score: float
    decision: str
    threshold: float
    missing_or_unseen: int
    top_reasons: list[Reason]


app = FastAPI(
    title="Fraud Scoring API",
    version="1.1.0",
    description="Scores a transaction for fraud risk and returns a block/approve decision.",
)


def to_frame(raw: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Turn the request into a one-row DataFrame exactly like the training data."""
    unknown = sorted(set(raw) - set(FEATURES))
    if unknown:
        raise HTTPException(422, f"Unknown features: {unknown[:5]}")

    row, missing = {}, 0
    for f in FEATURES:
        v = raw.get(f)
        if f in CATEGORIES:
            v = None if v is None else str(v)
            if v not in CATEGORIES[f]:      # missing, or a category never seen in training
                v = None
        else:
            if v is not None:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise HTTPException(422, f"Feature '{f}' must be a number, got {v!r}")
                if math.isnan(v):
                    v = None
        missing += v is None
        row[f] = v

    df = pd.DataFrame([row], columns=FEATURES)
    for f in FEATURES:
        if f in CATEGORIES:
            df[f] = pd.Categorical(df[f], categories=CATEGORIES[f])
        else:
            df[f] = df[f].astype("float64")
    return df, missing


def log_prediction(record: dict) -> None:
    """Append one line per prediction. Real systems send this to a database or log service."""
    RECENT.append(record)
    with LOG_FILE.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def score_psi(scores: list[float]) -> float | None:
    """PSI between live scores and the scores seen when the model was trained."""
    if REFERENCE is None or len(scores) < MIN_FOR_DRIFT:
        return None
    edges = np.array(REFERENCE["edges"])
    expected = np.clip(np.array(REFERENCE["proportions"]), 1e-6, None)
    actual = np.histogram(np.clip(scores, 0, 1), edges)[0] / len(scores)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    start = time.perf_counter()
    df, missing = to_frame(tx.features)

    score = float(model.predict(df)[0])

    # LightGBM's built-in SHAP values: how much each feature pushed this score
    contrib = model.predict(df, pred_contrib=True)[0][:-1]
    top = np.argsort(-np.abs(contrib))[:3]
    reasons = [
        Reason(feature=FEATURES[i], value=tx.features.get(FEATURES[i]), impact=round(float(contrib[i]), 3))
        for i in top
    ]
    decision = "block" if score >= THRESHOLD else "approve"
    request_id = str(uuid.uuid4())

    log_prediction({
        "request_id": request_id,
        "time": datetime.now(timezone.utc).isoformat(),
        "score": round(score, 4),
        "decision": decision,
        "missing_or_unseen": missing,
        "amount": tx.features.get("TransactionAmt"),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })

    return Prediction(
        request_id=request_id,
        fraud_score=round(score, 4),
        decision=decision,
        threshold=THRESHOLD,
        missing_or_unseen=missing,
        top_reasons=reasons,
    )


@app.get("/monitor")
def monitor():
    """Summary of recent traffic: volume, block rate, speed and score drift."""
    if not RECENT:
        return {"n_predictions": 0}
    scores = [r["score"] for r in RECENT]
    latencies = [r["latency_ms"] for r in RECENT]
    psi = score_psi(scores)
    if psi is None:
        status = f"need {MIN_FOR_DRIFT}+ predictions for a drift check"
    elif psi < 0.1:
        status = "stable"
    elif psi < 0.25:
        status = "watch"
    else:
        status = "shifted: investigate"
    return {
        "n_predictions": len(RECENT),
        "block_rate": round(float(np.mean([r["decision"] == "block" for r in RECENT])), 4),
        "mean_score": round(float(np.mean(scores)), 4),
        "mean_missing_or_unseen": round(float(np.mean([r["missing_or_unseen"] for r in RECENT])), 2),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 2),
        "score_psi": None if psi is None else round(psi, 4),
        "drift_status": status,
    }
