# Phase 6 — Backend and frontend implementation notes

## Backend
FastAPI loads the final Phase 5 sklearn pipeline from `models/logistic_regression_final.joblib`. It exposes health, single-record scoring, batch scoring, and ranked call-list endpoints. Input validation checks the 20 model features. The batch endpoint caps requests at 2,000 records to keep the local prototype bounded.

## Call-list behavior
The `/call-list` endpoint scores all holdout admissions, sorts descending by score, and returns `ceil(5% × number_of_admissions)` records. For the 8,000-row holdout this is 400 call-list slots. The endpoint does not use any target labels.

## Frontend
Streamlit retrieves the call list only through the backend API. It displays total admissions, call capacity, ranked patients, score bands, and simple descriptive signals for a selected admission.

## Explainability and limitations
The displayed signals are rule-based descriptions of selected input fields, not a faithful decomposition of the Logistic Regression prediction. They are not causal. Scores are for prioritization and should not be interpreted as diagnoses or guaranteed calibrated probabilities. The system is a synthetic-data demonstration, not a clinically validated deployment.
