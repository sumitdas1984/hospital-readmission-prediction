#!/usr/bin/env python3
"""Train Logistic Regression baseline and tune a small Random Forest grid.

Run from repository root:
    uv run python code/train_models.py

Uses data/processed/train.csv and validation.csv only.
Does not read internal_test.csv or the 2024 holdout.
"""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from feature_config import MODEL_FEATURES, TARGET
from preprocessing import make_preprocessor

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "processed" / "train.csv"
VALID = ROOT / "data" / "processed" / "validation.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def top_k_metrics(y_true, scores, fraction=0.05):
    y = np.asarray(y_true)
    s = np.asarray(scores)
    k = max(1, int(np.ceil(len(y) * fraction)))
    idx = np.argsort(s)[::-1][:k]
    precision = float(y[idx].mean())
    recall = float(y[idx].sum() / y.sum()) if y.sum() else 0.0
    return precision, recall, k


def evaluate(name, model, X, y):
    scores = model.predict_proba(X)[:, 1]
    p5, r5, k = top_k_metrics(y, scores)
    return scores, {
        "model": name,
        "precision_at_5pct": p5,
        "recall_at_5pct": r5,
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)),
        "top_5pct_count": k,
    }


def main():
    train = pd.read_csv(TRAIN)
    valid = pd.read_csv(VALID)
    X_train, y_train = train[MODEL_FEATURES], train[TARGET]
    X_valid, y_valid = valid[MODEL_FEATURES], valid[TARGET]

    lr = Pipeline([
        ("preprocessor", make_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    started = time.time()
    lr.fit(X_train, y_train)
    lr_seconds = time.time() - started
    lr_scores, lr_metrics = evaluate("Logistic Regression baseline", lr, X_valid, y_valid)
    lr_metrics["fit_seconds"] = round(lr_seconds, 2)

    candidates = [
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 10, "max_features": "sqrt"},
    ]
    rf_rows, best = [], None
    for i, params in enumerate(candidates, 1):
        rf = Pipeline([
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(
                **params, random_state=42, n_jobs=-1, class_weight=None
            )),
        ])
        started = time.time()
        rf.fit(X_train, y_train)
        fit_seconds = time.time() - started
        scores, metrics = evaluate(f"Random Forest candidate {i}", rf, X_valid, y_valid)
        metrics.update(params)
        metrics["fit_seconds"] = round(fit_seconds, 2)
        rf_rows.append(metrics)
        key = (metrics["precision_at_5pct"], metrics["recall_at_5pct"], metrics["pr_auc"])
        if best is None or key > best[0]:
            best = (key, rf, params, scores, metrics)

    joblib.dump(lr, MODEL_DIR / "logistic_regression_baseline.joblib")
    joblib.dump(best[1], MODEL_DIR / "random_forest_selected.joblib")
    pd.DataFrame([lr_metrics] + rf_rows).to_csv(
        RESULT_DIR / "validation_model_comparison.csv", index=False
    )
    with open(RESULT_DIR / "model_selection.json", "w", encoding="utf-8") as f:
        json.dump({
            "selected_random_forest_params": best[2],
            "selection_rule": "Highest validation Precision@5%; tie-break by Recall@5%, then PR-AUC.",
            "validation_only": True,
        }, f, indent=2)
    pd.DataFrame({
        "admission_id": valid["admission_id"],
        "y_true": y_valid,
        "logistic_regression_risk": lr_scores,
        "random_forest_risk": best[3],
    }).to_csv(RESULT_DIR / "validation_predictions.csv", index=False)

    print(pd.DataFrame([lr_metrics] + rf_rows).to_string(index=False))
    print("\\nSelected RF parameters:", best[2])
    print("Internal test and holdout were not read.")


if __name__ == "__main__":
    main()
