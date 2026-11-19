"""
Generate synthetic hospital readmission datasets for the Lakeshore assignment.

Usage:
    python synthetic_data_generator/generate.py

Dependencies: numpy, pandas
Random seed: 42
Outputs:
    data/raw/train.csv   (60,000 rows; 26 columns)
    data/raw/holdout.csv (8,000 rows; 25 columns; no target column)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_TRAIN = 60_000
N_HOLDOUT = 8_000
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

COLUMNS = [
    "admission_id", "patient_id", "hospital_site", "admit_date",
    "discharge_date", "age", "sex", "insurance_type", "admission_source",
    "primary_dx_group", "comorbidity_count", "los_days",
    "n_prior_admissions_12m", "n_ed_visits_6m", "num_medications", "bmi",
    "lab_creatinine_last", "lab_sodium_last", "lab_hemoglobin_last",
    "systolic_bp_last", "heart_rate_last", "discharge_disposition",
    "weekend_discharge", "followup_visit_scheduled",
    "post_discharge_call_logged", "readmitted_30d",
]

SITE_VALUES = [f"Site_{i}" for i in range(1, 7)]
SITE_P = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
DX_VALUES = [
    "Cardiovascular", "Respiratory", "Endocrine_Diabetes", "Renal",
    "Gastrointestinal", "Neurological", "Musculoskeletal",
    "Infectious_Diseases", "Oncology", "Mental_Health", "Hematology", "Other",
]
DX_P = [0.15, 0.12, 0.12, 0.08, 0.09, 0.07, 0.07, 0.10, 0.06, 0.05, 0.03, 0.06]


def weighted_choice(rng, values, probabilities, size=None):
    return rng.choice(values, size=size, p=probabilities)


def make_patient_population(rng, n_patients: int):
    # Age groups: 15%, 20%, 25%, 25%, 15%.
    age_group = rng.choice(5, size=n_patients, p=[.15, .20, .25, .25, .15])
    bounds = [(18, 35), (36, 50), (51, 65), (66, 80), (81, 95)]
    ages = np.array([rng.integers(bounds[g][0], bounds[g][1] + 1) for g in age_group])
    sex = rng.choice(["F", "M"], size=n_patients, p=[.52, .48])
    return pd.DataFrame({
        "patient_id": [f"P{i:06d}" for i in range(n_patients)],
        "base_age": ages,
        "sex": sex,
        "utilization_propensity": rng.normal(0, 1, n_patients),
    })


def allocate_admissions(rng, n_total: int, n_patients: int):
    """
    Make approximately 30% of patients have multiple admissions.
    Each repeat patient receives at least 2 admissions; remaining admissions
    are distributed among repeat patients.
    """
    n_repeat = int(round(0.30 * n_patients))
    counts = np.ones(n_patients, dtype=int)
    counts[:n_repeat] = 2
    extra = n_total - counts.sum()
    if extra < 0:
        raise ValueError("Patient count too small for requested repeat share.")
    # Additional admissions are concentrated among repeat patients.
    recipients = rng.choice(n_repeat, size=extra, replace=True)
    np.add.at(counts, recipients, 1)
    rng.shuffle(counts)
    # Re-shuffle patient ordering is handled by patient IDs; count sum is exact.
    return counts


def make_admission_schedule(rng, patients: pd.DataFrame, counts: np.ndarray):
    frames = []
    patient_idx = np.repeat(np.arange(len(patients)), counts)
    rng.shuffle(patient_idx)
    # Assign exactly 60k admissions to 2021-23 and 8k to 2024 Q1.
    train_dates = pd.to_datetime(rng.integers(
        pd.Timestamp("2021-01-01").value // 10**9,
        pd.Timestamp("2024-01-01").value // 10**9,
        size=N_TRAIN
    ), unit="s").normalize()
    holdout_dates = pd.to_datetime(rng.integers(
        pd.Timestamp("2024-01-01").value // 10**9,
        pd.Timestamp("2024-04-01").value // 10**9,
        size=N_HOLDOUT
    ), unit="s").normalize()
    dates = np.concatenate([train_dates.values, holdout_dates.values])
    periods = np.array(["train"] * N_TRAIN + ["holdout"] * N_HOLDOUT)
    order = np.argsort(dates, kind="stable")
    patient_idx = patient_idx[order]
    dates = dates[order]
    periods = periods[order]
    df = pd.DataFrame({
        "patient_index": patient_idx,
        "admit_date": pd.to_datetime(dates),
        "period": periods,
    })
    # Ensure patient histories are temporally plausible by allowing repeated
    # admissions at least 35 days apart. Reassign collisions forward.
    df = df.sort_values(["patient_index", "admit_date"]).reset_index(drop=True)
    for idx in range(1, len(df)):
        if df.loc[idx, "patient_index"] == df.loc[idx - 1, "patient_index"]:
            min_date = df.loc[idx - 1, "admit_date"] + pd.Timedelta(days=35)
            if df.loc[idx, "admit_date"] < min_date:
                df.loc[idx, "admit_date"] = min_date
    # Restore requested period sizes after date adjustment; period follows the
    # original generated admission slot, and dates are kept inside broad bounds.
    # A small number may cross a boundary after collision adjustment; clip safely.
    train_mask = df["period"].eq("train")
    df.loc[train_mask, "admit_date"] = df.loc[train_mask, "admit_date"].clip(
        lower=pd.Timestamp("2021-01-01"), upper=pd.Timestamp("2023-12-31"))
    hold_mask = ~train_mask
    df.loc[hold_mask, "admit_date"] = df.loc[hold_mask, "admit_date"].clip(
        lower=pd.Timestamp("2024-01-01"), upper=pd.Timestamp("2024-03-31"))
    # Stable ordering not required in output.
    return df


def generate():
    rng = np.random.default_rng(SEED)

    # 68,000 admissions; 47,600 unique patients gives 30% repeat patients
    # by construction (some repeat patients receive >2 admissions).
    n_patients = 47_600
    patients = make_patient_population(rng, n_patients)
    counts = allocate_admissions(rng, N_TRAIN + N_HOLDOUT, n_patients)
    schedule = make_admission_schedule(rng, patients, counts)
    df = schedule.merge(patients.reset_index().rename(columns={"index": "patient_index"}),
                        on="patient_index", how="left")
    df = df.sort_values(["admit_date", "patient_id"]).reset_index(drop=True)
    df["admission_id"] = [f"A{i:07d}" for i in range(len(df))]

    # Admission-specific characteristics.
    n = len(df)
    age_elapsed = (df["admit_date"] - pd.Timestamp("2021-01-01")).dt.days / 365.25
    df["age"] = (df["base_age"] + np.floor(age_elapsed / 4)).clip(18, 100).astype(int)
    df["hospital_site"] = weighted_choice(rng, SITE_VALUES, SITE_P, n)
    df["insurance_type"] = weighted_choice(
        rng, ["Medicare", "Commercial", "Medicaid", "Self-pay"], [.45, .30, .20, .05], n)
    df["admission_source"] = weighted_choice(
        rng, ["Emergency", "Referral", "Transfer"], [.60, .25, .15], n)
    df["primary_dx_group"] = weighted_choice(rng, DX_VALUES, DX_P, n)

    # Comorbidities correlated with age.
    age_factor = (df["age"].to_numpy() - 18) / 82
    comorb = rng.poisson(0.9 + 2.8 * age_factor, n)
    comorb += rng.binomial(1, .10, n)
    df["comorbidity_count"] = np.clip(comorb, 0, 10).astype(int)

    # Length of stay: diagnosis and complexity influence the distribution.
    los_base = rng.choice([1, 2, 3, 4, 5, 6, 7, 10, 14, 20],
                          size=n, p=[.12, .13, .18, .17, .10, .08, .07, .08, .04, .03])
    complexity = (df["comorbidity_count"].to_numpy() >= 4).astype(int)
    df["los_days"] = np.clip(los_base + rng.binomial(2, .25, n) + complexity, 1, 30).astype(int)
    df["discharge_date"] = df["admit_date"] + pd.to_timedelta(df["los_days"], unit="D")

    # Count prior admissions in the preceding 12 months from generated records.
    prior_counts = np.zeros(n, dtype=int)
    ed_counts = np.zeros(n, dtype=int)
    for p_idx, group in df.groupby("patient_index", sort=False):
        positions = group.index.to_numpy()
        dates = group["admit_date"].to_numpy(dtype="datetime64[D]")
        for j, pos in enumerate(positions):
            if j:
                delta = (dates[j] - dates[:j]).astype("timedelta64[D]").astype(int)
                prior_counts[pos] = int(np.sum(delta <= 365))
    df["n_prior_admissions_12m"] = np.minimum(prior_counts, 8)

    # ED use: correlated with past admissions and patient utilization propensity.
    prop = df["utilization_propensity"].to_numpy()
    ed_lambda = np.clip(.35 + .30 * np.maximum(prop, -1) + .30 * df["n_prior_admissions_12m"].to_numpy(), .10, 3.5)
    df["n_ed_visits_6m"] = np.clip(rng.poisson(ed_lambda), 0, 8).astype(int)

    # Discharge medication burden correlated with age/comorbidities.
    med_mean = 3.0 + .75 * df["comorbidity_count"].to_numpy() + .025 * df["age"].to_numpy()
    df["num_medications"] = np.clip(rng.poisson(med_mean), 0, 30).astype(int)

    # BMI: mixture across broad categories, with plausible continuous values.
    bmi_ranges = [(14, 18.49), (18.5, 24.9), (25, 29.9), (30, 34.9), (35, 39.9), (40, 55)]
    bmi_probs = [.05, .25, .30, .20, .12, .08]
    bmi_group = rng.choice(len(bmi_ranges), n, p=bmi_probs)
    df["bmi"] = np.array([rng.uniform(*bmi_ranges[g]) for g in bmi_group]).round(1)

    dx = df["primary_dx_group"].to_numpy()
    renal = dx == "Renal"
    cardio = dx == "Cardiovascular"
    respiratory = dx == "Respiratory"
    infectious = dx == "Infectious_Diseases"

    creat = rng.lognormal(mean=np.log(1.0) + .10 * df["comorbidity_count"].to_numpy()
                          + .35 * renal, sigma=.35, size=n)
    df["lab_creatinine_last"] = np.clip(creat, .4, 5.0).round(2)
    sodium = rng.normal(139 - 1.5 * infectious - .6 * df["comorbidity_count"].to_numpy(), 2.5, n)
    df["lab_sodium_last"] = np.clip(sodium, 125, 150).round(1)
    hemoglobin = rng.normal(13.2 - .35 * df["comorbidity_count"].to_numpy()
                            - .8 * (df["sex"].to_numpy() == "F")
                            - .7 * (dx == "Hematology"), 1.5, n)
    df["lab_hemoglobin_last"] = np.clip(hemoglobin, 6, 18).round(1)
    sbp = rng.normal(125 + .25 * (df["age"].to_numpy() - 50)
                     + 4 * cardio, 15, n)
    df["systolic_bp_last"] = np.clip(sbp, 75, 210).round(0)
    hr = rng.normal(78 + 5 * respiratory + 4 * infectious, 12, n)
    df["heart_rate_last"] = np.clip(hr, 40, 150).round(0)

    # Discharge disposition influenced by age and complexity.
    disp_score = (df["age"].to_numpy() > 75).astype(float) + .25 * df["comorbidity_count"].to_numpy()
    probs = np.column_stack([
        np.exp(-.20 * disp_score), np.exp(.10 * disp_score),
        np.exp(.30 * disp_score), np.exp(.20 * disp_score)
    ])
    probs = probs / probs.sum(axis=1, keepdims=True)
    disp_values = np.array(["Home", "Home with services", "Skilled nursing", "Rehab"])
    df["discharge_disposition"] = [rng.choice(disp_values, p=probs[i]) for i in range(n)]

    df["weekend_discharge"] = df["discharge_date"].dt.dayofweek.ge(5).astype(int)
    follow_prob = np.clip(.55 + .08 * (df["age"].to_numpy() > 65)
                          + .025 * df["comorbidity_count"].to_numpy(), .40, .90)
    df["followup_visit_scheduled"] = rng.binomial(1, follow_prob, n)
    df["post_discharge_call_logged"] = rng.binomial(1, .30, n)

    # Synthetic label: multi-factor logit plus noise, then intercept calibration
    # to an expected 15% prevalence. Holdout labels are removed from the CSV.
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

    creat_arr = df["lab_creatinine_last"].to_numpy()
    sodium_arr = df["lab_sodium_last"].to_numpy()
    hb_arr = df["lab_hemoglobin_last"].to_numpy()
    sbp_arr = df["systolic_bp_last"].to_numpy()
    hr_arr = df["heart_rate_last"].to_numpy()
    dx_risk = np.select(
        [renal, cardio, respiratory, infectious, dx == "Oncology", dx == "Mental_Health"],
        [.45, .30, .25, .25, .35, .12], default=0.0)
    linear = (
        .022 * (df["age"].to_numpy() - 60)
        + .20 * df["comorbidity_count"].to_numpy()
        + .32 * df["n_prior_admissions_12m"].to_numpy()
        + .16 * df["n_ed_visits_6m"].to_numpy()
        + .035 * (df["los_days"].to_numpy() - 4)
        + .035 * (df["num_medications"].to_numpy() - 8)
        + .55 * np.maximum(creat_arr - 1.2, 0)
        + .07 * np.abs(sodium_arr - 139)
        + .10 * np.maximum(12 - hb_arr, 0)
        + .012 * np.abs(sbp_arr - 125)
        + .012 * np.abs(hr_arr - 80)
        + dx_risk
        + .20 * (df["admission_source"].to_numpy() == "Emergency")
        + .18 * np.isin(df["discharge_disposition"].to_numpy(), ["Skilled nursing", "Rehab"])
        - .30 * df["followup_visit_scheduled"].to_numpy()
        + .12 * (df["bmi"].to_numpy() >= 35)
        + rng.normal(0, .35, n)
    )
    lo, hi = -12.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(linear + mid).mean() > .15:
            hi = mid
        else:
            lo = mid
    probabilities = sigmoid(linear + (lo + hi) / 2)
    df["readmitted_30d"] = rng.binomial(1, probabilities, n)

    # Ensure the requested exact row counts and preserve time-based datasets.
    train = df[df["period"] == "train"].copy()
    holdout = df[df["period"] == "holdout"].copy()
    # Dates are generated in the respective ranges; remove internal-only fields.
    train = train[COLUMNS]
    holdout = holdout[[c for c in COLUMNS if c != "readmitted_30d"]]
    for part in (train, holdout):
        part["admit_date"] = pd.to_datetime(part["admit_date"]).dt.strftime("%Y-%m-%d")
        part["discharge_date"] = pd.to_datetime(part["discharge_date"]).dt.strftime("%Y-%m-%d")

    # Required missingness: approximately 4% independently in each clinical field.
    missing_rng = np.random.default_rng(SEED + 1)
    missing_cols = [
        "lab_creatinine_last", "lab_sodium_last", "lab_hemoglobin_last",
        "systolic_bp_last", "heart_rate_last"
    ]
    for part in (train, holdout):
        for col in missing_cols:
            mask = missing_rng.random(len(part)) < .04
            part.loc[mask, col] = np.nan

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(RAW_DIR / "train.csv", index=False)
    holdout.to_csv(RAW_DIR / "holdout.csv", index=False)

    print(f"Generated: {RAW_DIR / 'train.csv'} ({len(train):,} rows, {len(train.columns)} columns)")
    print(f"Generated: {RAW_DIR / 'holdout.csv'} ({len(holdout):,} rows, {len(holdout.columns)} columns)")
    print(f"Training readmission prevalence: {train['readmitted_30d'].mean():.2%}")
    print(f"Internal holdout prevalence diagnostic (not exported): {df.loc[df['period'] == 'holdout', 'readmitted_30d'].mean():.2%}")
    print(f"Unique patients: {df['patient_id'].nunique():,}")
    print(f"Patients with >1 admission: {(df.groupby('patient_id').size() > 1).mean():.2%}")


if __name__ == "__main__":
    generate()
