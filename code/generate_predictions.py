#!/usr/bin/env python3
"""Train final Logistic Regression on all labelled development data and score holdout.

Run from repository root:
    uv run python code/generate_predictions.py

Reads:
    data/raw/train.csv
    data/raw/holdout.csv

Writes:
    models/logistic_regression_final.joblib
    predictions.csv
    results/phase5_prediction_summary.json

The holdout file must not contain readmitted_30d. The script never accesses holdout labels.
"""
from pathlib import Path
import json
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from feature_config import MODEL_FEATURES, TARGET
from preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/raw/train.csv"
HOLDOUT_PATH = ROOT / "data/raw/holdout.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_csv(TRAIN_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)

    if TARGET not in train.columns:
        raise ValueError("Training file must contain readmitted_30d.")
    if TARGET in holdout.columns:
        raise ValueError("Holdout must not contain readmitted_30d; do not use holdout labels.")
    if not train["admission_id"].is_unique or not holdout["admission_id"].is_unique:
        raise ValueError("Admission IDs must be unique within each file.")
    if set(train["admission_id"]).intersection(set(holdout["admission_id"])):
        raise ValueError("Train and holdout admission IDs overlap.")
    missing_train = sorted(set(MODEL_FEATURES) - set(train.columns))
    missing_holdout = sorted(set(MODEL_FEATURES) - set(holdout.columns))
    if missing_train or missing_holdout:
        raise ValueError(f"Missing model features. train={missing_train}, holdout={missing_holdout}")

    model = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])

    start = time.time()
    model.fit(train[MODEL_FEATURES], train[TARGET].astype(int))
    fit_seconds = time.time() - start
    scores = model.predict_proba(holdout[MODEL_FEATURES])[:, 1]

    predictions = pd.DataFrame({
        "admission_id": holdout["admission_id"],
        "risk_score": scores,
    })
    if len(predictions) != len(holdout):
        raise RuntimeError("Prediction row count does not match holdout.")
    if predictions["admission_id"].duplicated().any():
        raise RuntimeError("Duplicate admission IDs in predictions.")
    if predictions["risk_score"].isna().any() or not predictions["risk_score"].between(0, 1).all():
        raise RuntimeError("Risk scores must be non-null values between 0 and 1.")

    predictions.to_csv(ROOT / "predictions.csv", index=False)
    joblib.dump(model, MODEL_DIR / "logistic_regression_final.joblib")

    summary = {
        "selected_model": "Logistic Regression",
        "training_rows": int(len(train)),
        "training_prevalence": float(train[TARGET].mean()),
        "holdout_rows_scored": int(len(holdout)),
        "holdout_labels_accessed": False,
        "prediction_columns": ["admission_id", "risk_score"],
        "prediction_row_count": int(len(predictions)),
        "unique_admission_ids": int(predictions["admission_id"].nunique()),
        "risk_score_min": float(predictions["risk_score"].min()),
        "risk_score_max": float(predictions["risk_score"].max()),
        "risk_score_mean": float(predictions["risk_score"].mean()),
        "fit_seconds": round(fit_seconds, 2),
        "model_parameters": {"C": 1.0, "max_iter": 1000, "random_state": 42},
        "features": MODEL_FEATURES,
        "excluded_features": [
            "admission_id", "patient_id", "admit_date", "discharge_date",
            "post_discharge_call_logged", TARGET
        ],
    }
    (RESULTS_DIR / "phase5_prediction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
