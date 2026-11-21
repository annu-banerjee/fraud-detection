"""
Automatic checks for the fraud API.

Run from the project folder:
    pytest -v
"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app, FEATURES, THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def load(name):
    return json.loads((ROOT / "examples" / f"{name}.json").read_text())


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["n_features"] == len(FEATURES)


@pytest.mark.parametrize("name", ["fraud_example", "legit_example"])
def test_response_shape(name):
    r = client.post("/predict", json=load(name))
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["decision"] in {"block", "approve"}
    assert body["threshold"] == THRESHOLD
    assert len(body["top_reasons"]) == 3


def test_decision_matches_threshold():
    body = client.post("/predict", json=load("fraud_example")).json()
    expected = "block" if body["fraud_score"] >= THRESHOLD else "approve"
    assert body["decision"] == expected


def test_known_cases():
    # regression check: these two real transactions should keep their decisions
    assert client.post("/predict", json=load("fraud_example")).json()["decision"] == "block"
    assert client.post("/predict", json=load("legit_example")).json()["decision"] == "approve"


def test_same_input_same_score():
    a = client.post("/predict", json=load("fraud_example")).json()["fraud_score"]
    b = client.post("/predict", json=load("fraud_example")).json()["fraud_score"]
    assert a == b


def test_all_missing_still_scores():
    r = client.post("/predict", json={"features": {}})
    assert r.status_code == 200
    assert r.json()["missing_or_unseen"] == len(FEATURES)


def test_unseen_category_is_treated_as_missing():
    cat_feature = next(f for f in FEATURES if f in app_categories())
    r = client.post("/predict", json={"features": {cat_feature: "never-seen-before"}})
    assert r.status_code == 200


def test_unknown_feature_rejected():
    r = client.post("/predict", json={"features": {"not_a_real_feature": 1}})
    assert r.status_code == 422


def test_text_in_number_field_rejected():
    num_feature = next(f for f in FEATURES if f not in app_categories())
    r = client.post("/predict", json={"features": {num_feature: "abc"}})
    assert r.status_code == 422


def app_categories():
    from src.api import CATEGORIES
    return CATEGORIES


def test_prediction_is_logged_and_monitored():
    client.post("/predict", json=load("legit_example"))
    body = client.get("/monitor").json()
    assert body["n_predictions"] >= 1
    assert 0.0 <= body["block_rate"] <= 1.0
