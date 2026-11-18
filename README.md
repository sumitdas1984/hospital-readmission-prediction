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
├── code/                      # Audit, training, evaluation, prediction
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

Run Python scripts with:

```bash
uv run python code/run_pipeline.py
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
