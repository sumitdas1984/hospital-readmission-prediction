# Project Execution Plan

## 1. Objective

Build and submit a reproducible end-to-end machine-learning solution that predicts the risk of **30-day hospital readmission after discharge** and helps a care-management team prioritize a weekly call list when it has capacity to contact approximately **5% of discharged patients**.

The solution must satisfy the assignment deliverables while remaining simple enough to explain and defend in the follow-up discussion.

## 2. Working Principles

- Work phase by phase; complete and validate each phase before moving on.
- Keep implementation, documentation, and explanations concise and understandable.
- Prioritize sound reasoning, data integrity, leakage prevention, and reproducibility over unnecessary complexity.
- Use synthetic data only, following the assignment restrictions. Do not introduce real patient data, external clinical datasets, or pretrained models.
- Use `uv` for Python dependency and environment management. Keep `pyproject.toml` and `uv.lock` in version control.
- Do not add tools, infrastructure, or architectural complexity unless the assignment requires them.
- Record important assumptions, alternatives, decisions, limitations, and trade-offs so they can be explained in the follow-up interview.

## 3. Agreed Repository Structure

```text
hospital-readmission-prediction/
├── synthetic_data_generator/
│   ├── __init__.py
│   └── generate.py
├── data/
│   ├── raw/
│   └── processed/
├── code/
│   ├── create_temporal_splits.py
│   ├── feature_config.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── generate_predictions.py
│   ├── run_app.py
│   └── README.md
├── models/
├── backend/
│   └── app/
│       ├── __init__.py
│       └── main.py
├── frontend/
│   └── app.py
├── docs/
│   ├── README.md
│   └── project-execution-plan.md
├── predictions.csv
├── decision_log.md
├── recommendation.md
├── solution_architecture.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .gitignore
└── README.md
```

This structure is frozen. Change it only if an implementation or explicit assignment requirement makes a change necessary.

Implementation changed it as follows. The planned `run_pipeline.py`, `data_audit.py`, and `predict.py` were never implemented — they remained empty stubs from the initial scaffold — so they were removed and their roles are filled by `create_temporal_splits.py`, `train_models.py`, `evaluate.py`, and `generate_predictions.py`, each run as its own step and documented in `code/README.md`. The one-time data audit is recorded in `docs/phase2-data-audit.md` rather than as a script. `train.py` became `train_models.py`, `generate_data.py` became `generate.py`, and the backend moved to `backend/app/main.py` to match the `backend.app.main:app` import path used to serve the API.

## 4. Execution Phases

### Phase 0 — Repository and Environment Setup

**Status: Complete**

- Create the GitHub repository and local working copy.
- Establish the agreed directory and file structure.
- Configure the project as a `uv` project.
- Define dependencies in `pyproject.toml`.
- Generate and commit `uv.lock` after running `uv sync`.
- Confirm the initial scaffold is committed and pushed.

**Exit criteria:** The project runs in its `uv` environment and the initial scaffold is available in GitHub.

### Phase 1 — Synthetic Dataset Design and Generation

**Status: Next**

Assignment dataset requirements:
- Training dataset: **60,000 admissions**.
- Holdout dataset: **8,000 admissions**.
- Training period: **January 2021–December 2023**.
- Holdout period: **January–March 2024**.
- Target: 30-day readmission after discharge.

Tasks:
1. Translate the assignment into a clear data dictionary and target definition.
2. Define a plausible synthetic population, admission patterns, patient-level variation, clinical/utilization features, missingness, and time effects.
3. Ensure features are available at the discharge-time prediction point.
4. Generate separate training and holdout datasets with the required sizes and date ranges.
5. Make generation deterministic through a fixed random seed.
6. Save generated files under `data/raw/`.

**Exit criteria:** Both datasets meet the specified sizes, schema, date ranges, target definition, and documented generation assumptions.

### Phase 2 — Data Audit and Validation

Tasks:
1. Inspect schema, data types, row counts, date ranges, target prevalence, and missing values.
2. Check duplicates, invalid values, feature distributions, and relationships with the target.
3. Check repeated admissions per patient and patient overlap between training and holdout; document the chosen handling.
4. Check for temporal or target leakage, especially information that would only be known after discharge.
5. Record findings and any corrective actions.

**Exit criteria:** Data-quality checks pass, leakage risks are addressed, and the audit findings are documented.

### Phase 3 — Feature Preparation and Baseline Modeling

Tasks:
1. Define the prediction unit and prediction timestamp clearly.
2. Build a reproducible preprocessing and feature-engineering pipeline.
3. Establish a simple baseline model.
4. Train a small, justified set of candidate models only if useful.
5. Use a time-aware validation strategy within the training period; keep the final holdout untouched.
6. Persist the fitted preprocessing and model artifacts under `models/`.

**Suggested temporal split to evaluate:** Train on 2021–2022, validate on the first half of 2023, and use the second half of 2023 as an internal test set. Confirm the exact split after inspecting the generated data.

**Exit criteria:** A reproducible training pipeline and a selected model are available, with the selection rationale recorded.

### Phase 4 — Model Evaluation and Decision Analysis

Tasks:
1. Evaluate ranking quality using **Precision@5%** and **Recall@5%**, reflecting the care team's limited call capacity.
2. Also report suitable overall discrimination and probability-quality metrics, such as PR-AUC, ROC-AUC, and Brier score.
3. Inspect calibration and select a calibration approach only if supported by validation results.
4. Explain the threshold/ranking approach and translate model results into operational implications.
5. Do not use the final holdout to tune models, features, thresholds, or calibration.

**Exit criteria:** Evaluation results support a clear model choice and a defensible weekly prioritization approach.

### Phase 5 — Holdout Scoring and `predictions.csv`

Tasks:
1. Apply the frozen preprocessing and selected model to the 8,000 holdout admissions.
2. Generate a readmission probability and prediction for each required record, following the assignment's required output schema.
3. Validate row count, identifiers, column names, missing predictions, and probability range.
4. Save the final file at the repository root as `predictions.csv`.

**Exit criteria:** `predictions.csv` matches the assignment format and contains valid predictions for all holdout records.

### Phase 6 — Backend API

Implement the FastAPI service to load the saved model and preprocessing artifacts and provide:
- `GET /health`
- `POST /predict` for an individual admission
- `POST /predict/batch` for batch scoring
- `GET /call-list` for a ranked call list

The API serves predictions; it does not train or retrain the model.

**Exit criteria:** Endpoints accept validated inputs, return the expected outputs, and handle basic errors.

### Phase 7 — Frontend Application

Implement a simple Streamlit interface for care-management users to:
- View the weekly ranked call list.
- See readmission risk scores/probabilities.
- Identify the highest-priority patients within the available call capacity.
- Understand the ranking and the intended decision-support use.

Keep the UI focused on the assignment workflow rather than adding unrelated features.

**Exit criteria:** The frontend can retrieve/display a ranked list through the backend and makes the 5% capacity constraint clear.

### Phase 8 — Documentation, Testing, and Final Packaging

Complete and review the required deliverables:
- `predictions.csv`
- `decision_log.md` — within the assignment's two-page limit.
- `recommendation.md` — the requested five-sentence recommendation to the care-management lead.
- `solution_architecture.md`
- Source code, README, and reproducible setup/run instructions.
- Synthetic data generator and application components.

Also:
1. Add focused tests for data generation, preprocessing/prediction, and API behavior.
2. Verify setup from a clean environment using `uv sync`.
3. Run the complete pipeline from documented commands.
4. Confirm all required files are present and conform to the assignment.
5. Review the repository for accidental sensitive data, unnecessary artifacts, and undocumented assumptions.
6. Prepare to explain the target, data generation, leakage controls, validation design, metrics, model choice, calibration, API, and operational recommendation.

**Exit criteria:** A clean, reproducible, complete submission is ready and the solution can be explained end to end.

## 5. Evaluation Priorities

The assignment emphasizes reasoning and communication (**80%**), application (**15%**), and holdout performance (**5%**). Accordingly:

1. Make every major design choice explicit and defensible.
2. Ensure the application works and demonstrates the intended workflow.
3. Treat holdout performance as an unbiased final check, not a tuning target.

The 5% call capacity is central to the use case. A model should be assessed by how effectively it ranks patients for that limited capacity, not only by generic classification accuracy.

## 6. Project Working Agreement

For each phase, follow this loop:

1. Agree on the immediate objective and acceptance criteria.
2. Explain the design briefly before implementation.
3. Implement only the agreed scope.
4. Run checks and inspect the outputs together.
5. Document assumptions, decisions, and results.
6. Mark the phase complete before proceeding.

Avoid jumping ahead or expanding scope while a phase is still being validated. If the assignment wording conflicts with this plan, the assignment takes precedence and the plan must be updated accordingly.

## 7. Current Project Status

| Phase | Description | Status |
|---|---|---|
| 0 | Repository and environment setup | Complete |
| 1 | Synthetic Data Design and Generation | Complete |
| 2 | Data Audit and Validation | Complete |
| 3 | Baseline and Stronger Model Development | Complete |
| 4 | Model Evaluation and Calibration | Complete |
| 5 | Holdout Prediction Generation | Complete |
| 6 | Backend Development | Complete |
| 7 | Frontend Development | **Complete** |
| 8 | Documentation and Final Submission | Next |
