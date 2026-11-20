"""Canonical feature definitions for the discharge-time readmission model."""

TARGET = "readmitted_30d"

# Identifiers, raw timestamps, target, and post-discharge information are excluded.
EXCLUDED_FEATURES = [
    "admission_id",
    "patient_id",
    "admit_date",
    "discharge_date",
    "post_discharge_call_logged",
    TARGET,
]

NUMERIC_FEATURES = [
    "age",
    "comorbidity_count",
    "los_days",
    "n_prior_admissions_12m",
    "n_ed_visits_6m",
    "num_medications",
    "bmi",
    "lab_creatinine_last",
    "lab_sodium_last",
    "lab_hemoglobin_last",
    "systolic_bp_last",
    "heart_rate_last",
    "weekend_discharge",
    "followup_visit_scheduled",
]

CATEGORICAL_FEATURES = [
    "hospital_site",
    "sex",
    "insurance_type",
    "admission_source",
    "primary_dx_group",
    "discharge_disposition",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
