# Phase 3 — Step 2: Feature Definition and Preprocessing

## Prediction point

The model predicts risk **at hospital discharge**. A feature is eligible only if it is known at or before that point.

## Canonical feature set

There are 20 model features: 14 numeric/binary and 6 categorical.

### Numeric and binary features (14)

- `age`
- `comorbidity_count`
- `los_days`
- `n_prior_admissions_12m`
- `n_ed_visits_6m`
- `num_medications`
- `bmi`
- `lab_creatinine_last`
- `lab_sodium_last`
- `lab_hemoglobin_last`
- `systolic_bp_last`
- `heart_rate_last`
- `weekend_discharge`
- `followup_visit_scheduled`

### Categorical features (6)

- `hospital_site`
- `sex`
- `insurance_type`
- `admission_source`
- `primary_dx_group`
- `discharge_disposition`

## Excluded columns

| Column | Reason |
|---|---|
| `admission_id` | Unique record identifier; no clinical predictive meaning intended |
| `patient_id` | Patient identifier; do not let the model memorize identity |
| `admit_date` | Raw timestamp excluded; temporal ordering is handled by the split |
| `discharge_date` | Raw timestamp excluded; prediction occurs at discharge and derived features are used instead |
| `post_discharge_call_logged` | May only be known after discharge; direct timing leakage risk |
| `readmitted_30d` | Target label, never a predictor |

`followup_visit_scheduled` is included under the explicit synthetic-data assumption that it is known by discharge. The five `*_last` lab/vital measurements are assumed to have been collected and available by discharge. These timing assumptions must be revisited if the assignment's data dictionary or deployment workflow says otherwise.

## Preprocessing

| Feature type | Logistic Regression | Random Forest |
|---|---|---|
| Numeric/binary missing values | Median imputation | Median imputation |
| Numeric scaling | StandardScaler | None |
| Categorical missing values | Most-frequent imputation | Most-frequent imputation |
| Categorical encoding | OneHotEncoder, `handle_unknown='ignore'` | Same |
| Unused columns | Dropped | Dropped |

Scaling is useful for Logistic Regression because it is sensitive to feature scale. Tree-based Random Forest splits do not require standardized numeric inputs.

All preprocessing is implemented as part of an sklearn `Pipeline`/`ColumnTransformer`. It must be fitted only on the training partition, then applied unchanged to validation and internal test. This prevents imputation statistics, scaling parameters, and category vocabulary from being learned from evaluation data.

## Class imbalance

The training target prevalence is approximately 15%, so the dataset is imbalanced. Start with the default `class_weight=None` for both models. We are evaluating a probability ranking and a fixed top-5% call capacity; class weighting can change probability behavior. We can test it as a controlled validation-only experiment if the initial models show a reason to do so.

## Implementation files

- `code/feature_config.py` — canonical feature lists and exclusions.
- `code/preprocessing.py` — reusable preprocessing builder.

## Validation performed for Step 2

- Confirmed the 20 selected feature names exist in both training and holdout schemas.
- Confirmed excluded columns are not in the model feature list.
- Verified that the preprocessing design can transform a training sample for both model configurations.
- No model was trained in this step.

## Decision

Use the same 20 eligible features for both candidate models to make the comparison fair. Fit all preprocessing only on the training partition. Keep the 2024 holdout untouched.
