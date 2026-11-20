# ML Pipeline Code

This directory contains the data audit, preprocessing, training, evaluation,
prediction, and pipeline orchestration scripts.

Run from repository root:

- Regenerate the required holdout predictions: `uv run python code/generate_predictions.py`
- Start the FastAPI backend and Streamlit frontend together: `uv run python code/run_app.py`

The prediction script reads `data/raw/train.csv` and `data/raw/holdout.csv`, retrains the selected Logistic Regression pipeline, and writes `predictions.csv`. The holdout file must not contain the target column. The app launcher starts both local processes and stops them when interrupted.

Other scripts for data splitting, model training, and evaluation should remain in this folder and be described here with their run order.
