# ML Pipeline Code

This directory contains the temporal splitting, preprocessing, training,
evaluation, and prediction scripts.

Run from repository root, in this order:

- Create temporal splits: `uv run python code/create_temporal_splits.py`
- Train and select a model: `uv run python code/train_models.py`
- Evaluate on the internal test set: `uv run python code/evaluate.py`
- Regenerate the required holdout predictions: `uv run python code/generate_predictions.py`
- Start the FastAPI backend and Streamlit frontend together: `uv run python code/run_app.py`

`feature_config.py` and `preprocessing.py` are shared modules, not entry points.
The model features, the prediction moment, and the leakage exclusions are
defined in `feature_config.py`.

The prediction script reads `data/raw/train.csv` and `data/raw/holdout.csv`, retrains the selected Logistic Regression pipeline, and writes `predictions.csv`. The holdout file must not contain the target column. The app launcher starts both local processes and stops them when interrupted.

The data audit was performed once against the generated datasets and is
documented in `docs/phase2-data-audit.md`; it is not a repeatable script.
