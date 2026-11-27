"""
MP181 - Power Loom Output Prediction API
Group G037 - FastAPI service with business validation rules
"""

import logging
import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# ── JSON logging setup ────────────────────────────────────────────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp181")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ── Load model (graceful fallback if not yet trained) ─────────────────────────
MODEL_PATH = "models/model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception:
    model = None
    logger.info("No model file found — running without model")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="MP181 Power Loom Output Prediction API")

NUMERIC_FEATURES = [
    "loom_age_yr", "thread_count", "power_cuts_min",
    "weaver_exp_yr", "humidity_pct", "shift_hours",
]
CATEGORICAL_FEATURES = ["fabric", "shed"]


class PredictRequest(BaseModel):
    loom_age_yr: float
    thread_count: int
    power_cuts_min: float = Field(..., le=250, description="Max 250 minutes")
    weaver_exp_yr: float
    humidity_pct: float
    shift_hours: float = Field(..., ge=3, description="Min 3 hours")
    fabric: str
    shed: str


class PredictResponse(BaseModel):
    prediction: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    data = req.model_dump()
    logger.info(json.dumps({
        "event": "prediction_request",
        "shed": data["shed"],
        "fabric": data["fabric"],
        "shift_hours": data["shift_hours"],
    }))

    if model is None:
        return PredictResponse(prediction=0.0)

    df = pd.DataFrame([data])
    prediction = float(model.predict(df)[0])

    logger.info(json.dumps({
        "event": "prediction_result",
        "prediction": prediction,
    }))

    return PredictResponse(prediction=prediction)
