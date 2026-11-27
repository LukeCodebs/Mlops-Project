"""
MP181 - API Tests
Group G037 - pytest tests using TestClient
Note: Tests are self-contained and do not require CSV files or a trained model.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "loom_age_yr": 5.0,
    "thread_count": 120,
    "power_cuts_min": 60.0,
    "weaver_exp_yr": 8.0,
    "humidity_pct": 55.0,
    "shift_hours": 8.0,
    "fabric": "cotton",
    "shed": "Shed 1",
}


# ── Test 1: Health endpoint ───────────────────────────────────────────────────
def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ── Test 2: Business rule — power_cuts_min > 250 must be rejected ─────────────
def test_predict_rejects_power_cuts_above_250():
    payload = {**VALID_PAYLOAD, "power_cuts_min": 300.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ── Test 3: Business rule — shift_hours < 3 must be rejected ──────────────────
def test_predict_rejects_shift_hours_below_3():
    payload = {**VALID_PAYLOAD, "shift_hours": 1.5}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ── Test 4: Valid request returns a prediction ────────────────────────────────
def test_predict_valid_request_returns_prediction():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))


# ── Test 5: Missing required field returns 422 ────────────────────────────────
def test_predict_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "fabric"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
