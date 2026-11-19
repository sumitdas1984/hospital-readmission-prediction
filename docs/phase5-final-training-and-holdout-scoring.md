# Phase 5 — Final model training and holdout scoring

## Final model
Logistic Regression was selected after it outperformed the selected Random Forest on both validation and the untouched internal-test period. The final Logistic Regression pipeline retains the Phase 3 configuration: median imputation and standardization for numeric fields; most-frequent imputation and one-hot encoding for categorical fields; `C=1.0`, `max_iter=1000`, `random_state=42`.

## Final training procedure
After model selection and internal-test evaluation, the selected model was refitted on all labelled development data (`data/raw/train.csv`, 60,000 admissions, January 2021–December 2023 admission dates). The preprocessing steps were fitted only on this labelled development dataset. The final pipeline was then used to score the 8,000 rows in `data/raw/holdout.csv`.

The holdout file contains no `readmitted_30d` column. No holdout labels were accessed. The model excludes identifiers, raw dates, the target, and `post_discharge_call_logged` (a post-discharge field). It uses the 20 features specified in the Phase 3 feature configuration.

## Prediction-file validation
- Exactly 8,000 rows.
- Exactly two columns, in order: `admission_id`, `risk_score`.
- All admission IDs are unique and correspond to holdout admissions.
- All risk scores are non-null and between 0 and 1.
- The output preserves the holdout row order.

## Observed score distribution
- Minimum: 0.0117
- Maximum: 0.7700
- Mean: 0.1254

These are model risk scores. Given the calibration limitations documented in Phase 4, do not describe them as perfectly calibrated individual probabilities.

## Reproducibility
From the repository root, run:
`uv run python code/generate_predictions.py`

This reads the two raw CSVs, retrains the final pipeline, writes `models/logistic_regression_final.joblib`, writes `predictions.csv`, and writes `results/phase5_prediction_summary.json`. It does not read the temporal internal-test file or any holdout labels.
