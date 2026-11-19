from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT_DIR / "models/logistic_regression_final.joblib"))
HOLDOUT_PATH = Path(os.getenv("HOLDOUT_PATH", ROOT_DIR / "data/raw/holdout.csv"))
FEATURE_CONFIG_PATH = Path(os.getenv("FEATURE_CONFIG_PATH", ROOT_DIR / "config/features.json"))

if not FEATURE_CONFIG_PATH.exists():
    # Also support a standalone backend folder with the config copied beside the repository.
    FEATURE_CONFIG_PATH = APP_DIR.parent.parent / "config/features.json"

with open(FEATURE_CONFIG_PATH, encoding="utf-8") as f:
    MODEL_FEATURES = json.load(f)["model_features"]

app = FastAPI(
    title="30-Day Readmission Risk API",
    description="Local decision-support API for prioritizing discharge follow-up.",
    version="1.0.0",
)

_model = None
_holdout = None

class PatientRecord(BaseModel):
    admission_id: str | None = None
    features: dict[str, Any]

class BatchRequest(BaseModel):
    records: list[PatientRecord] = Field(min_length=1, max_length=2000)

def load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model

def validate_and_frame(records: list[PatientRecord]) -> pd.DataFrame:
    rows = []
    for i, record in enumerate(records):
        missing = [name for name in MODEL_FEATURES if name not in record.features]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={"record_index": i, "missing_features": missing},
            )
        rows.append({name: record.features[name] for name in MODEL_FEATURES})
    return pd.DataFrame(rows, columns=MODEL_FEATURES)

def explain_record(record: dict[str, Any]) -> list[str]:
    """Plain-language descriptive signals; these are not causal explanations."""
    reasons = []
    checks = [
        ("n_prior_admissions_12m", lambda x: float(x) >= 2, "Two or more prior admissions in the last 12 months"),
        ("n_ed_visits_6m", lambda x: float(x) >= 2, "Two or more emergency visits in the last 6 months"),
        ("comorbidity_count", lambda x: float(x) >= 4, "Higher recorded comorbidity count"),
        ("los_days", lambda x: float(x) >= 7, "Longer hospital stay"),
        ("num_medications", lambda x: float(x) >= 10, "Higher number of medications"),
        ("discharge_disposition", lambda x: str(x).lower() in {"skilled nursing", "rehab"}, "Discharge to skilled nursing or rehabilitation"),
        ("followup_visit_scheduled", lambda x: int(float(x)) == 0, "No follow-up visit recorded as scheduled at discharge"),
        ("lab_creatinine_last", lambda x: float(x) >= 1.5, "Creatinine value is elevated relative to the simple review threshold"),
        ("lab_sodium_last", lambda x: float(x) < 135 or float(x) > 145, "Sodium value is outside the simple review range"),
        ("heart_rate_last", lambda x: float(x) < 60 or float(x) > 100, "Heart rate is outside the simple review range"),
        ("systolic_bp_last", lambda x: float(x) < 90 or float(x) > 140, "Systolic blood pressure is outside the simple review range"),
    ]
    for field, condition, message in checks:
        value = record.get(field)
        if value is None or pd.isna(value):
            continue
        try:
            if condition(value):
                reasons.append(message)
        except (TypeError, ValueError):
            continue
    return reasons[:4] or ["No selected high-risk signals identified by the simple rules; review the overall score and clinical context."]

def score_records(records: list[PatientRecord]) -> list[dict[str, Any]]:
    frame = validate_and_frame(records)
    model = load_model()
    scores = model.predict_proba(frame)[:, 1]
    output = []
    for record, score in zip(records, scores):
        raw = record.features
        output.append({
            "admission_id": record.admission_id,
            "risk_score": float(score),
            "risk_band": "Higher relative risk" if score >= 0.25 else ("Moderate relative risk" if score >= 0.12 else "Lower relative risk"),
            "reasons": explain_record(raw),
            "score_note": "Model risk estimate for prioritization; not a diagnosis or a guarantee of readmission.",
        })
    return output

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_artifact_exists": MODEL_PATH.exists(),
        "holdout_file_exists": HOLDOUT_PATH.exists(),
        "feature_count": len(MODEL_FEATURES),
    }

@app.post("/predict")
def predict(record: PatientRecord):
    return score_records([record])[0]

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    return {"count": len(request.records), "predictions": score_records(request.records)}

@app.get("/call-list")
def call_list():
    global _holdout
    if _holdout is None:
        if not HOLDOUT_PATH.exists():
            raise HTTPException(status_code=500, detail=f"Holdout data not found: {HOLDOUT_PATH}")
        _holdout = pd.read_csv(HOLDOUT_PATH)
    missing = [name for name in MODEL_FEATURES if name not in _holdout.columns]
    if missing:
        raise HTTPException(status_code=500, detail={"holdout_missing_features": missing})

    model = load_model()
    scores = model.predict_proba(_holdout[MODEL_FEATURES])[:, 1]
    n = len(_holdout)
    capacity = math.ceil(0.05 * n)
    ranked = _holdout.copy()
    ranked["risk_score"] = scores
    ranked = ranked.sort_values("risk_score", ascending=False).head(capacity)

    result = []
    for _, row in ranked.iterrows():
        raw = row.to_dict()
        result.append({
            "admission_id": str(row["admission_id"]),
            "patient_id": str(row["patient_id"]),
            "risk_score": float(row["risk_score"]),
            "risk_band": "Higher relative risk" if row["risk_score"] >= 0.25 else ("Moderate relative risk" if row["risk_score"] >= 0.12 else "Lower relative risk"),
            "reasons": explain_record(raw),
        })
    return {
        "total_admissions": n,
        "capacity_percent": 5,
        "call_slots": capacity,
        "returned": len(result),
        "predictions": result,
        "note": "Top 5% ranked by model score. Scores support prioritization and do not replace clinical judgment.",
    }
