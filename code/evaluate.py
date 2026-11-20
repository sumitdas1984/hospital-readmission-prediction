"""
Evaluate the selected Phase 3 models on the untouched internal-test split.
Run from the repository root:
    uv run python code/phase4_evaluate.py
The script does not read or use the 2024 holdout file.
"""
from pathlib import Path
from math import ceil
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from feature_config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TEST_PATH = ROOT / "data/processed/internal_test.csv"
LR_PATH = ROOT / "models/logistic_regression_baseline.joblib"
RF_PATH = ROOT / "models/random_forest_selected.joblib"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

def score_model(name, model, X, y):
    prob = model.predict_proba(X)[:, 1]
    k = ceil(0.05 * len(y))
    selected = np.argsort(-prob)[:k]
    return {
        "model": name,
        "n": len(y),
        "positive_count": int(y.sum()),
        "prevalence": float(y.mean()),
        "top_5pct_slots": k,
        "precision_at_5pct": float(y.iloc[selected].mean()),
        "recall_at_5pct": float(y.iloc[selected].sum() / y.sum()),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "brier_score": float(brier_score_loss(y, prob)),
    }, prob

def calibration_deciles(y, prob):
    frame = pd.DataFrame({"actual": np.asarray(y), "probability": prob})
    frame["decile"] = pd.qcut(frame["probability"], 10, labels=False, duplicates="drop") + 1
    return frame.groupby("decile", as_index=False).agg(
        n=("actual", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_readmission_rate=("actual", "mean")
    )

def main():
    test = pd.read_csv(TEST_PATH)
    if "readmitted_30d" not in test:
        raise ValueError("Internal test must contain readmitted_30d.")
    X = test[FEATURES]
    y = test["readmitted_30d"].astype(int)

    models = {
        "Logistic Regression": joblib.load(LR_PATH),
        "Random Forest": joblib.load(RF_PATH),
    }
    rows = []
    for name, model in models.items():
        row, prob = score_model(name, model, X, y)
        rows.append(row)
        safe_name = name.lower().replace(" ", "_")
        pd.DataFrame({
            "admission_id": test["admission_id"],
            "actual": y,
            "risk_score": prob
        }).to_csv(OUT / f"{safe_name}_internal_test_predictions.csv", index=False)
        calibration_deciles(y, prob).to_csv(
            OUT / f"{safe_name}_calibration_deciles.csv", index=False
        )

    pd.DataFrame(rows).to_csv(OUT / "phase4_internal_test_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()
