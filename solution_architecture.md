# Solution Architecture — 30-Day Hospital Readmission Risk

## End-to-end flow

```text
Synthetic schema + documented assumptions
                  |
                  v
       synthetic_data_generator/
                  |
           +------+------+
           |             |
           v             v
 data/raw/train.csv  data/raw/holdout.csv
 (60k, 26 cols)      (8k, 25 cols; no label)
           |             |
           v             |
  Data audit + temporal split
           |
           v
  Preprocessing + model comparison
  Logistic Regression baseline
  Random Forest alternative
           |
           v
  Temporal validation and internal test
           |
           v
  Select Logistic Regression
           |
           v
  Refit on all 60k labelled development rows
           |
           v
  models/logistic_regression_final.joblib
           |
           +---------------------------+
                                       |
                                       v
                               FastAPI backend
                       /health /predict /predict/batch
                                /call-list
                                       |
                                       v
                                Streamlit frontend
                         ranked weekly top-5% call list
                         patient-level score and cues
```

## Data generation and audit
`synthetic_data_generator/generate.py` creates the training and holdout CSVs from the assignment schema using seed 42. All value distributions and label relationships are synthetic design assumptions. The audit checks schema, row counts, date ranges, missingness, uniqueness, history consistency, and key logical rules. No external clinical data or pretrained models are used.

## Model development and lifecycle
The prediction point is discharge. The feature configuration contains 20 predictors and excludes admission/patient identifiers, raw dates, the post-discharge call field, and the target. Numeric missing values are median-imputed; categorical missing values are most-frequent-imputed; categorical values are one-hot encoded. Logistic Regression and Random Forest were compared using a temporal train/validation/internal-test design. Logistic Regression was selected because it led on primary capacity-constrained metrics, secondary ranking metrics, and Brier score on both validation and internal test. The selected pipeline was refitted on all labelled development data and saved as `models/logistic_regression_final.joblib`.

## Backend
FastAPI loads the saved sklearn pipeline without retraining. It exposes:
- `GET /health` — service and artifact health.
- `POST /predict` — score one admission.
- `POST /predict/batch` — score a batch with request-size validation.
- `GET /call-list` — score holdout admissions, rank descending, and return `ceil(5% × N)` rows (400 of 8,000 for this assignment).

The backend validates required model features and returns a risk score, relative risk band, and simple descriptive cues. The cues are rule-based and should not be mistaken for causal explanations or complete model feature attribution.

## Frontend
The Streamlit application calls only the backend API. It shows the weekly ranked list, total admissions, 5% capacity, selected admission score, risk band, and review cues. A visible note states that scores are model risk estimates, not diagnoses or guarantees, and that clinical judgment remains necessary.

## Reproducibility and operational boundary
The prediction-generation entry point is `uv run python code/generate_predictions.py`. The backend and frontend can be started together using the local launcher documented in the repository README (or separately with the two documented commands). Install packages only from the approved internal Artifactory source. This is a synthetic-data demonstration, not a clinically validated service.
