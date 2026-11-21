"""
Fraud scoring API.

Run locally from the project folder:
    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""
import json
import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

# Load once when the server starts, not on every request
model = lgb.Booster(model_file=str(MODEL_DIR / "model.txt"))
meta = json.loads((MODEL_DIR / "metadata.json").read_text())

FEATURES: list[str] = meta["features"]
CATEGORIES: dict[str, list[str]] = meta["categorical"]
THRESHOLD: float = meta["threshold"]


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
    fraud_score: float
    decision: str
    threshold: float
    missing_or_unseen: int
    top_reasons: list[Reason]


app = FastAPI(
    title="Fraud Scoring API",
    version="1.0.0",
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


@app.get("/health")
def health():
    return {"status": "ok", "n_features": len(FEATURES), "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction):
    df, missing = to_frame(tx.features)

    score = float(model.predict(df)[0])

    # LightGBM's built-in SHAP values: how much each feature pushed this score
    contrib = model.predict(df, pred_contrib=True)[0][:-1]
    top = np.argsort(-np.abs(contrib))[:3]
    reasons = [
        Reason(feature=FEATURES[i], value=tx.features.get(FEATURES[i]), impact=round(float(contrib[i]), 3))
        for i in top
    ]

    return Prediction(
        fraud_score=round(score, 4),
        decision="block" if score >= THRESHOLD else "approve",
        threshold=THRESHOLD,
        missing_or_unseen=missing,
        top_reasons=reasons,
    )
