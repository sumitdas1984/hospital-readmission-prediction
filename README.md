# Hospital Readmission Prediction

End-to-end ML solution for predicting 30-day hospital readmission risk,
including synthetic data generation, temporal model evaluation, probability
calibration, holdout scoring, a FastAPI backend, and a Streamlit frontend.

The model ranks discharged patients by readmission risk so a care-management
team can work a weekly call list within a 5% outreach capacity.

> **All data is synthetic.** This is a demonstration prototype, not a clinically
> validated service. Do not use it for patient care.

## Repository Structure

```text
hospital-readmission-prediction/
├── synthetic_data_generator/   # Synthetic dataset generation (seed 42)
│   ├── __init__.py
│   └── generate.py
├── data/                       # Generated and derived datasets (gitignored)
│   ├── raw/                    #   train.csv (60,000), holdout.csv (8,000)
│   └── processed/              #   train.csv, validation.csv, internal_test.csv
├── code/                       # Splitting, training, evaluation, prediction
│   ├── create_temporal_splits.py
│   ├── feature_config.py       # Shared: model features and target column
│   ├── preprocessing.py        # Shared: imputation, scaling, one-hot encoding
│   ├── train_models.py
│   ├── evaluate.py
│   ├── generate_predictions.py
│   ├── run_app.py              # Starts backend and frontend together
│   └── README.md
├── config/
│   └── features.json           # Feature list the API validates requests against
├── models/                     # Saved sklearn pipelines (gitignored)
├── results/                    # Metrics, calibration tables, selection artifacts
├── backend/
│   └── app/
│       ├── __init__.py
│       └── main.py             # FastAPI scoring service
├── frontend/
│   └── app.py                  # Streamlit care-management UI
├── docs/                       # Phase-by-phase supporting notes
├── predictions.csv             # Holdout risk scores (8,000 rows)
├── decision_log.md
├── recommendation.md
├── solution_architecture.md
├── pyproject.toml
├── uv.lock
└── README.md
```

## Status

All implementation phases are complete.

| Phase | Description | Status |
|---|---|---|
| 0 | Repository and environment setup | Complete |
| 1 | Synthetic data design and generation | Complete |
| 2 | Data audit and validation | Complete |
| 3 | Baseline and stronger model development | Complete |
| 4 | Model evaluation and calibration | Complete |
| 5 | Holdout prediction generation | Complete |
| 6 | Backend API | Complete |
| 7 | Frontend application | Complete |
| 8 | Documentation and final submission | Complete |

**Selected model: Logistic Regression.** It outperformed the tuned Random
Forest on validation and internal-test metrics, including the primary
capacity-constrained metrics and Brier score, while being simpler to explain.

| Split | Model | Precision@5% | Recall@5% | PR-AUC | ROC-AUC | Brier |
|---|---|---:|---:|---:|---:|---:|
| Validation | Logistic Regression | 0.5524 | 0.1747 | 0.3989 | 0.7522 | 0.1155 |
| Validation | Random Forest | 0.5302 | 0.1677 | 0.3752 | 0.7422 | 0.1182 |
| Internal test | Logistic Regression | 0.4631 | 0.1586 | 0.3412 | 0.7432 | 0.1119 |
| Internal test | Random Forest | 0.4572 | 0.1566 | 0.3280 | 0.7302 | 0.1133 |

Metrics declined from validation to the later internal-test period, so the
scores should not be read as stable production performance. See
[`recommendation.md`](recommendation.md) and
[`decision_log.md`](decision_log.md).

## Setup with uv

Install `uv` if it is not already available, then run:

```bash
uv sync
```

This creates/updates the project virtual environment and installs the
dependencies declared in `pyproject.toml`, including the development
dependency group.

## Reproducibility

Run the following from the repository root, in order:

```bash
# 1. Generate the synthetic datasets
uv run python synthetic_data_generator/generate.py

# 2. Create the temporal train / validation / internal-test splits
uv run python code/create_temporal_splits.py

# 3. Train the baseline and the Random Forest candidates, select a model
uv run python code/train_models.py

# 4. Evaluate and calibrate on the internal test set
uv run python code/evaluate.py

# 5. Refit on all labelled data and score the 2024 holdout
uv run python code/generate_predictions.py
```

The generator uses random seed `42` and overwrites `data/raw/train.csv` and
`data/raw/holdout.csv` when it runs. Do not add labels to the holdout CSV or
use internal holdout diagnostics for model selection.

Steps 3 and 4 read only the labelled training data and the temporal splits.
Neither reads the 2024 holdout. Step 5 is the only script that reads the
holdout, and it never reads holdout labels.

The full script run order is also documented in
[`code/README.md`](code/README.md).

### Dependency management

Add a runtime dependency:

```bash
uv add <package>
```

Add a development dependency:

```bash
uv add --dev <package>
```

After changing dependencies, `uv` updates `pyproject.toml` and `uv.lock`.
Commit both files so the environment is reproducible.

## Data generation

The generator requires `numpy` and `pandas` and uses random seed `42`.

Outputs:
- `data/raw/train.csv` — 60,000 rows, 26 columns, includes `readmitted_30d`.
- `data/raw/holdout.csv` — 8,000 rows, 25 columns, excludes `readmitted_30d`.

The training admission-date window is 2021-01-01 to 2023-12-31; the holdout
window is 2024-01-01 to 2024-03-31.

### Key design decisions

See [`docs/data-generation-spec.md`](docs/data-generation-spec.md) for the agreed assumptions, feature timing, leakage controls, missingness, and target-generation approach.

## Evaluation package

```bash
uv run python code/evaluate.py
```

Prerequisites: Phase 3 model artifacts, temporal splits, and `code/feature_config.py` must already be present. The script reads only `data/processed/internal_test.csv`; it does not access the 2024 holdout.

## Holdout prediction generation

```bash
uv run python code/generate_predictions.py
```

The project must already contain `data/raw/train.csv` and `data/raw/holdout.csv` and the required approved dependencies (`pandas`, `scikit-learn`, `joblib`). Do not add `readmitted_30d` to holdout.

The script writes `predictions.csv` at the repository root with `admission_id`
and `risk_score` columns, `models/logistic_regression_final.joblib`, and
`results/phase5_prediction_summary.json`. It validates row count, duplicate
identifiers, missing values, and the probability range before writing.

## Local backend and care-management frontend

The existing model artifact must be at `models/logistic_regression_final.joblib`,
and the holdout CSV must be at `data/raw/holdout.csv`. The backend reads the
20 model features from `config/features.json`.

### Start both services

```bash
uv run python code/run_app.py
```

This starts the FastAPI backend on `127.0.0.1:8000` and the Streamlit frontend,
and stops both when interrupted.

### Start the backend

From repository root:
```bash
uv run uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Health check:
`http://127.0.0.1:8000/health`

Interactive API docs:
`http://127.0.0.1:8000/docs`

### Start the frontend

Open a second terminal from repository root:
```bash
uv run streamlit run frontend/app.py
```
The frontend calls only the backend API. Default backend URL is `http://127.0.0.1:8000`.
To override:
```bash
API_BASE_URL=http://127.0.0.1:8000 uv run streamlit run frontend/app.py
```
On Windows PowerShell:
```powershell
$env:API_BASE_URL="http://127.0.0.1:8000"
uv run streamlit run frontend/app.py
```

### API endpoints
- `GET /health` — service and artifact availability
- `POST /predict` — score one record; JSON body: `{"admission_id":"A123","features":{...20 model features...}}`
- `POST /predict/batch` — score 1–2,000 records
- `GET /call-list` — ranks holdout admissions and returns `ceil(5% × N)` records

### Important design notes
- The backend loads the final saved sklearn pipeline; it does not retrain.
- The call list is generated from the holdout admissions and capped at 5%. For the 8,000-row holdout this is 400 call-list slots.
- The API validates that required model features are present.
- The UI displays risk ranking, simple descriptive signals, and score limitations.
- Reason strings are transparent rule-based cues, not SHAP explanations, causal explanations, or the model's full feature attribution.
- The API serves predictions; it does not train or retrain the model.
- Do not expose this synthetic prototype as a real clinical service. The dataset is synthetic and the model is not clinically validated.
- The assignment specifies use of approved internal package sources only. Install dependencies only through the approved Artifactory/package process; do not use public PyPI/npm.

## Supporting documentation

| Document | Contents |
|---|---|
| [`decision_log.md`](decision_log.md) | Assumptions, alternatives, decisions, limitations, and trade-offs |
| [`recommendation.md`](recommendation.md) | Recommendation to the care-management lead |
| [`solution_architecture.md`](solution_architecture.md) | End-to-end architecture and data flow |
| [`docs/data-generation-spec.md`](docs/data-generation-spec.md) | Synthetic data design and label mechanism |
| [`docs/phase2-data-audit.md`](docs/phase2-data-audit.md) | Data quality checks and corrected distributions |
| [`docs/phase3-temporal-split.md`](docs/phase3-temporal-split.md) | Temporal partition design |
| [`docs/phase3-feature-and-preprocessing.md`](docs/phase3-feature-and-preprocessing.md) | Feature eligibility, exclusions, and preprocessing |
| [`docs/phase3-model-development.md`](docs/phase3-model-development.md) | Baseline and candidate model development |
| [`docs/phase4-evaluation-and-model-decision.md`](docs/phase4-evaluation-and-model-decision.md) | Evaluation, calibration, and model selection |
| [`docs/phase5-final-training-and-holdout-scoring.md`](docs/phase5-final-training-and-holdout-scoring.md) | Final refit and holdout scoring |
| [`docs/phase6-architecture-and-usage.md`](docs/phase6-architecture-and-usage.md) | Backend and frontend implementation notes |
| [`docs/project-execution-plan.md`](docs/project-execution-plan.md) | Objective, phases, and working agreement |

## Limitations

- All data is synthetic. Nothing here establishes real-world clinical effectiveness.
- The model predicts risk at discharge using only features known at that point; it does not use post-discharge information.
- Scores are relative rankings for prioritization, not diagnoses or guaranteed calibrated probabilities.
- The temporal split approximates future-period deployment but is not a prospective clinical validation.
- Repeated patients can appear across temporal partitions; patient ID is excluded as a predictor and history features use only earlier events.
- Results are based on a synthetic internal test period and showed some decline from validation to that later period.
