# Hospital Readmission Prediction

End-to-end ML solution for predicting 30-day hospital readmission risk,
including synthetic data generation, model evaluation, probability
calibration, a FastAPI backend, and a Streamlit frontend.

## Repository Structure

```text
hospital-readmission-prediction/
├── synthetic_data_generator/  # Synthetic dataset generation
├── data/
│   ├── raw/                   # Generated raw datasets
│   └── processed/             # Cleaned/transformed datasets
├── code/                      # Splitting, training, evaluation, prediction
├── models/                    # Saved model and preprocessing artifacts
├── backend/                   # FastAPI scoring service
├── frontend/                  # Streamlit application
├── docs/                      # Supporting documentation
├── predictions.csv
├── decision_log.md
├── recommendation.md
├── solution_architecture.md
├── pyproject.toml             # Project metadata and dependencies
├── uv.lock                    # Reproducible dependency lockfile (generated)
└── README.md
```

## Setup with uv

Install `uv` if it is not already available, then run:

```bash
uv sync
```

This creates/updates the project virtual environment and installs the
dependencies declared in `pyproject.toml`, including the development
dependency group.

Run Python scripts from the repository root, for example:

```bash
uv run python code/train_models.py
```

Run the API (once implemented) with:

```bash
uv run uvicorn backend.main:app --reload
```

Run the Streamlit app (once implemented) with:

```bash
uv run streamlit run frontend/app.py
```

## Dependency Management

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

## Status

Initial project scaffold. Implementation will be completed in phases.

## Reproducibility

Data generation, model training, evaluation, and prediction commands will
be documented here as they are implemented. Check the assignment's
submission requirements before deciding which generated data and model
artifacts must be included in the final repository.

## Generate the datasets

From the repository root:

```bash
python synthetic_data_generator/generate.py
```

The generator requires `numpy` and `pandas` and uses random seed `42`.

Outputs:
- `data/raw/train.csv` — 60,000 rows, 26 columns, includes `readmitted_30d`.
- `data/raw/holdout.csv` — 8,000 rows, 25 columns, excludes `readmitted_30d`.

The generator overwrites these two files when run. Do not add labels to the holdout CSV or use internal holdout diagnostics for model selection.

### Environment

If using the project's `uv` setup:

```bash
uv sync
uv run python synthetic_data_generator/generate.py
```

### Key design decisions

See [`docs/data-generation-spec.md`](docs/data-generation-spec.md) for the agreed assumptions, feature timing, leakage controls, missingness, and target-generation approach.

All data is synthetic. It is not suitable for clinical use.


## Evaluation package

Run from the repository root:
```bash
uv run python code/evaluate.py
```
Prerequisites: Phase 3 model artifacts, temporal splits, and `code/feature_config.py` must already be present. The script reads only `data/processed/internal_test.csv`; it does not access the 2024 holdout.


## Phase 5 — Holdout Prediction Generation

Run from repository root:
```bash
uv run python code/generate_predictions.py
```
The project must already contain `data/raw/train.csv` and `data/raw/holdout.csv` and the required approved dependencies (`pandas`, `scikit-learn`, `joblib`). Do not add `readmitted_30d` to holdout.


## Phase 6 — Local backend and care-management frontend

The existing Phase 5 model artifact must be at `models/logistic_regression_final.joblib`.
The existing holdout CSV must be at `data/raw/holdout.csv`.

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
- The call list is generated from the holdout admissions and capped at 5%.
- The API validates that required model features are present.
- The UI displays risk ranking, simple descriptive signals, and score limitations.
- Reason strings are transparent rule-based cues, not SHAP explanations, causal explanations, or the model's full feature attribution.
- Do not expose this synthetic prototype as a real clinical service. The dataset is synthetic and the model is not clinically validated.
- The assignment specifies use of approved internal package sources only. Install dependencies only through the approved Artifactory/package process; do not use public PyPI/npm.
