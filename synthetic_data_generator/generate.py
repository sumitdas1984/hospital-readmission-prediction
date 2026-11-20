"""Generate the assignment's synthetic hospital readmission datasets.

Run from the repository root:
    python synthetic_data_generator/generate.py
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TRAIN = 60_000
N_HOLDOUT = 8_000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"

SITES = [f"Site_{i}" for i in range(1, 7)]
SITE_P = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
DIAGNOSES = [
    "Cardiovascular", "Respiratory", "Endocrine / Diabetes", "Renal",
    "Gastrointestinal", "Neurological", "Musculoskeletal",
    "Infectious Diseases", "Oncology", "Mental Health", "Hematology", "Other",
]
DIAG_P = [0.15, 0.12, 0.12, 0.08, 0.09, 0.07, 0.07, 0.10, 0.06, 0.05, 0.03, 0.06]

TRAIN_COLUMNS = [
    "admission_id", "patient_id", "hospital_site", "admit_date", "discharge_date",
    "age", "sex", "insurance_type", "admission_source", "primary_dx_group",
    "comorbidity_count", "los_days", "n_prior_admissions_12m", "n_ed_visits_6m",
    "num_medications", "bmi", "lab_creatinine_last", "lab_sodium_last",
    "lab_hemoglobin_last", "systolic_bp_last", "heart_rate_last",
    "discharge_disposition", "weekend_discharge", "followup_visit_scheduled",
    "post_discharge_call_logged", "readmitted_30d",
]
HOLDOUT_COLUMNS = [c for c in TRAIN_COLUMNS if c != "readmitted_30d"]


def weighted_choice(rng: np.random.Generator, values, probabilities, size=None):
    return rng.choice(values, size=size, p=probabilities)


def sample_age(rng: np.random.Generator, n: int) -> np.ndarray:
    bands = [(18, 35), (36, 50), (51, 65), (66, 80), (81, 95)]
    probs = [0.15, 0.20, 0.25, 0.25, 0.15]
    choices = rng.choice(len(bands), size=n, p=probs)
    return np.array([rng.integers(bands[i][0], bands[i][1] + 1) for i in choices])


def sample_comorbidities(rng: np.random.Generator, ages: np.ndarray) -> np.ndarray:
    # Age shifts the distribution upward while preserving the agreed overall mix approximately.
    out = np.empty(len(ages), dtype=int)
    for i, age in enumerate(ages):
        probs = np.array([0.15, 0.35, 0.30, 0.15, 0.05], dtype=float)
        if age >= 66:
            probs += np.array([-0.06, -0.05, 0.02, 0.05, 0.04])
        elif age <= 35:
            probs += np.array([0.08, 0.02, -0.05, -0.03, -0.02])
        probs = np.clip(probs, 0.005, None)
        probs /= probs.sum()
        band = rng.choice(5, p=probs)
        out[i] = [0, rng.integers(1, 3), rng.integers(3, 5), rng.integers(5, 7), rng.integers(7, 11)][band]
    return out


def sample_los(rng: np.random.Generator, diagnoses: np.ndarray) -> np.ndarray:
    # Keep the agreed overall LOS mix; diagnosis-specific variation is deliberately
    # modest so it does not overwhelm the documented marginal distribution.
    base = np.array([0.25, 0.35, 0.25, 0.12, 0.03], dtype=float)
    result = np.empty(len(diagnoses), dtype=int)
    higher_complexity = {"Cardiovascular", "Renal", "Oncology", "Infectious Diseases"}
    for i, dx in enumerate(diagnoses):
        p = base.copy()
        if dx in higher_complexity:
            p += np.array([-0.01, -0.005, 0.005, 0.007, 0.003])
        else:
            p += np.array([0.003, 0.002, -0.002, -0.002, -0.001])
        p = np.clip(p, 0.001, None)
        p /= p.sum()
        band = rng.choice(5, p=p)
        result[i] = [rng.integers(1, 3), rng.integers(3, 5), rng.integers(5, 8), rng.integers(8, 15), rng.integers(15, 22)][band]
    return result


def build_event_schedule(rng: np.random.Generator, patients: np.ndarray, event_patient_ids: np.ndarray):
    """Create exact train/holdout counts and non-overlapping patient timelines.

    Each patient selected for holdout has at least one earlier training admission.
    Repeated admissions are clustered sufficiently closely to create meaningful
    12-month history while retaining a minimum 35-day gap after prior discharge.
    """
    counts = pd.Series(event_patient_ids).value_counts()
    repeat_patients = counts[counts >= 2].index.to_numpy()
    single_patients = counts[counts == 1].index.to_numpy()
    # Keep a meaningful overlap between train and holdout while retaining a
    # substantial set of previously unseen patients in the future period.
    n_shared_holdout = min(2_000, len(repeat_patients))
    n_new_holdout = N_HOLDOUT - n_shared_holdout
    if len(single_patients) < n_new_holdout:
        raise RuntimeError("Not enough single-admission patients for holdout allocation")
    holdout_patients = set(rng.choice(repeat_patients, size=n_shared_holdout, replace=False).tolist())
    holdout_patients.update(rng.choice(single_patients, size=n_new_holdout, replace=False).tolist())
    records = []
    train_start, train_end = date(2021, 1, 1), date(2023, 12, 31)
    holdout_start, holdout_end = date(2024, 1, 1), date(2024, 3, 31)

    for pid in patients:
        n_events = int(counts.get(pid, 0))
        has_holdout = pid in holdout_patients
        n_train = n_events - int(has_holdout)
        # A 56-100 day interval leaves room for a maximum 21-day stay plus
        # a 35-day separation before the next admission.
        if n_train:
            span = (train_end - train_start).days
            first_latest = train_end - timedelta(days=100 * (n_train - 1))
            first_offset = int(rng.integers(0, max(1, (first_latest - train_start).days + 1)))
            current = train_start + timedelta(days=first_offset)
            for j in range(n_train):
                if j:
                    current += timedelta(days=int(rng.integers(56, 101)))
                records.append((pid, current, "train"))
        if has_holdout:
            hold_date = holdout_start + timedelta(days=int(rng.integers(0, (holdout_end-holdout_start).days + 1)))
            records.append((pid, hold_date, "holdout"))

    schedule = pd.DataFrame(records, columns=["patient_id", "admit_date", "split"])
    # If a patient's final training stay would overlap the holdout admission, move
    # the holdout admission forward within its permitted quarter.
    for pid, group in schedule.groupby("patient_id"):
        tr = group[group.split == "train"].sort_values("admit_date")
        ho_idx = group[group.split == "holdout"].index
        if len(tr) and len(ho_idx):
            earliest = tr.iloc[-1].admit_date + timedelta(days=56)
            idx = ho_idx[0]
            if schedule.loc[idx, "admit_date"] < earliest:
                schedule.loc[idx, "admit_date"] = earliest
    # A 35-day gap is enforced between the prior discharge and next admission.
    schedule = schedule.sort_values(["patient_id", "admit_date"]).reset_index(drop=True)
    for pid, group in schedule.groupby("patient_id"):
        prev_discharge = None
        for idx in group.index:
            admit = schedule.loc[idx, "admit_date"]
            if prev_discharge is not None and admit < prev_discharge + timedelta(days=56):
                adjusted = prev_discharge + timedelta(days=56)
                split = schedule.loc[idx, "split"]
                upper = train_end if split == "train" else holdout_end
                if adjusted > upper:
                    raise RuntimeError(f"Unable to schedule non-overlapping admissions for {pid}")
                schedule.loc[idx, "admit_date"] = adjusted
                admit = adjusted
            prev_discharge = admit
    if (schedule.split == "train").sum() != N_TRAIN or (schedule.split == "holdout").sum() != N_HOLDOUT:
        raise RuntimeError("Schedule did not preserve required split sizes")
    return schedule.sort_values("admit_date").reset_index(drop=True)


def make_dates(rng: np.random.Generator, n: int, start: date, end: date) -> list[date]:
    span = (end - start).days + 1
    offsets = rng.integers(0, span, size=n)
    return [start + timedelta(days=int(x)) for x in offsets]


def make_patient_assignments(rng: np.random.Generator, n_total: int):
    # 70% single-admission patients; 20% have two admissions; 10% have five.
    # This makes exactly 30% of patients multi-admission patients and sums to n_total.
    n_patients = round(n_total / 1.6)
    n_single = int(round(n_patients * 0.70))
    n_double = int(round(n_patients * 0.20))
    n_five = n_patients - n_single - n_double
    counts = np.array([1] * n_single + [2] * n_double + [5] * n_five, dtype=int)
    # Reconcile rounding while preserving total admissions.
    delta = n_total - int(counts.sum())
    if delta > 0:
        counts[-delta:] += 1
    elif delta < 0:
        for idx in range(len(counts) - 1, -1, -1):
            if delta == 0:
                break
            if counts[idx] > 1:
                counts[idx] -= 1
                delta += 1
    if counts.sum() != n_total:
        raise RuntimeError("Could not reconcile patient admission counts")
    patient_ids = np.array([f"P{i:06d}" for i in range(n_patients)], dtype=object)
    event_patient_ids = np.repeat(patient_ids, counts)
    rng.shuffle(event_patient_ids)
    return patient_ids, event_patient_ids


def calibrate_intercept(linear_predictor: np.ndarray, target: float = 0.15) -> float:
    lo, hi = -15.0, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        mean_p = np.mean(1 / (1 + np.exp(-np.clip(linear_predictor + mid, -30, 30))))
        if mean_p > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate() -> None:
    rng = np.random.default_rng(SEED)
    n_total = N_TRAIN + N_HOLDOUT
    patients, event_patient_ids = make_patient_assignments(rng, n_total)

    # Build chronological patient histories with exact split sizes and no overlapping stays.
    events = build_event_schedule(rng, patients, event_patient_ids)

    patient_age = dict(zip(patients, sample_age(rng, len(patients))))
    patient_sex = dict(zip(patients, weighted_choice(rng, ["F", "M"], [0.52, 0.48], len(patients))))
    patient_ed_rate = dict(zip(patients, rng.uniform(0.15, 1.2, len(patients))))

    # Per-patient previous admission dates are tracked as records are generated chronologically.
    history: dict[str, list[date]] = {p: [] for p in patients}
    rows = []
    for idx, ev in events.iterrows():
        pid = ev.patient_id
        admit = ev.admit_date
        prior_dates = history[pid]
        prior_12m = sum((admit - d).days <= 365 for d in prior_dates if d < admit)
        prior_6m_admits = sum((admit - d).days <= 183 for d in prior_dates if d < admit)
        # ED visits include a patient-specific background propensity plus prior acute admissions.
        ed_lambda = patient_ed_rate[pid] + 0.35 * prior_6m_admits
        ed_count = min(int(rng.poisson(ed_lambda)), 8)
        age = min(95, patient_age[pid] + max(0, (admit - date(2021, 1, 1)).days // 365))
        sex = patient_sex[pid]

        site = weighted_choice(rng, SITES, SITE_P)
        insurance = weighted_choice(rng, ["Medicare", "Commercial", "Medicaid", "Self-pay"], [0.45, 0.30, 0.20, 0.05])
        source = weighted_choice(rng, ["Emergency", "Referral", "Transfer"], [0.60, 0.25, 0.15])
        dx = weighted_choice(rng, DIAGNOSES, DIAG_P)
        comorb = int(sample_comorbidities(rng, np.array([age]))[0])
        los = int(sample_los(rng, np.array([dx]))[0])

        # Correlated discharge medication burden, BMI, labs and disposition.
        med_mu = 5.0 + 1.15 * comorb + (1.2 if age >= 65 else 0)
        meds = int(np.clip(round(rng.normal(med_mu, 3.0)), 0, 28))
        bmi = float(np.clip(rng.normal(27.5 + (1.0 if dx == "Endocrine / Diabetes" else 0), 6.2), 14, 55))
        renal_shift = 0.8 if dx == "Renal" else 0.0
        creat = float(np.clip(rng.lognormal(mean=np.log(1.0 + renal_shift + 0.07 * comorb), sigma=0.45), 0.4, 5.0))
        sodium = float(np.clip(rng.normal(140 - (1.0 if dx in {"Renal", "Cardiovascular"} else 0) - 0.25 * comorb, 3.2), 125, 150))
        hemoglobin = float(np.clip(rng.normal((13.2 if sex == "M" else 12.2) - 0.25 * comorb - (0.7 if dx == "Hematology" else 0), 1.8), 6, 18))
        systolic = float(np.clip(rng.normal(124 + 0.18 * age - 0.9 * comorb, 17), 75, 210))
        heart_rate = float(np.clip(rng.normal(76 + 0.7 * comorb + (4 if source == "Emergency" else 0), 12), 40, 150))

        # The baseline mix follows the agreed overall proportions. Small risk-related
        # shifts preserve realistic variation without distorting the marginal mix.
        disp_logits = np.log(np.array([0.55, 0.20, 0.15, 0.10]))
        severity = 0.004 * max(age - 60, 0) + 0.025 * max(comorb - 2, 0)
        disp_logits = disp_logits + np.array([-0.35, 0.05, 0.18, 0.12]) * severity
        disp_p = np.exp(disp_logits - disp_logits.max()); disp_p /= disp_p.sum()
        disposition = weighted_choice(rng, ["Home", "Home with services", "Skilled nursing", "Rehab"], disp_p)
        followup_p = np.clip(0.65 + 0.05 * (comorb >= 4) + 0.03 * (age >= 70) - 0.04 * (source == "Emergency"), 0.45, 0.85)
        followup = int(rng.random() < followup_p)
        post_call = int(rng.random() < 0.30)
        discharge = admit + timedelta(days=los)
        weekend = int(discharge.weekday() >= 5)

        # Retain raw values; missingness is applied to the five required lab/vital fields.
        clinical = [creat, sodium, hemoglobin, systolic, heart_rate]
        for j in range(len(clinical)):
            if rng.random() < 0.04:
                clinical[j] = np.nan

        rows.append({
            "admission_id": f"A{idx + 1:07d}", "patient_id": pid, "hospital_site": site,
            "admit_date": admit, "discharge_date": discharge, "age": int(age), "sex": sex,
            "insurance_type": insurance, "admission_source": source, "primary_dx_group": dx,
            "comorbidity_count": comorb, "los_days": los, "n_prior_admissions_12m": int(prior_12m),
            "n_ed_visits_6m": ed_count, "num_medications": meds, "bmi": round(bmi, 1),
            "lab_creatinine_last": round(clinical[0], 2) if pd.notna(clinical[0]) else np.nan,
            "lab_sodium_last": round(clinical[1], 1) if pd.notna(clinical[1]) else np.nan,
            "lab_hemoglobin_last": round(clinical[2], 1) if pd.notna(clinical[2]) else np.nan,
            "systolic_bp_last": round(clinical[3], 1) if pd.notna(clinical[3]) else np.nan,
            "heart_rate_last": round(clinical[4], 1) if pd.notna(clinical[4]) else np.nan,
            "discharge_disposition": disposition, "weekend_discharge": weekend,
            "followup_visit_scheduled": followup, "post_discharge_call_logged": post_call,
        })
        history[pid].append(admit)

    df = pd.DataFrame(rows)
    # Generate labels using only discharge-time predictors; post-discharge call is not used.
    dx_effect = {
        "Cardiovascular": 0.35, "Respiratory": 0.30, "Endocrine / Diabetes": 0.20,
        "Renal": 0.40, "Gastrointestinal": 0.05, "Neurological": 0.10,
        "Musculoskeletal": -0.10, "Infectious Diseases": 0.25, "Oncology": 0.35,
        "Mental Health": 0.10, "Hematology": 0.20, "Other": 0.0,
    }
    lp = (
        0.018 * (df.age.to_numpy() - 60)
        + 0.22 * df.comorbidity_count.to_numpy()
        + 0.34 * df.n_prior_admissions_12m.to_numpy()
        + 0.13 * df.n_ed_visits_6m.to_numpy()
        + 0.025 * (df.los_days.to_numpy() - 4)
        + 0.035 * (df.num_medications.to_numpy() - 8)
        + np.array([dx_effect[x] for x in df.primary_dx_group])
        + 0.35 * (df.discharge_disposition.isin(["Skilled nursing", "Rehab"]).to_numpy())
        + 0.20 * (df.discharge_disposition.eq("Home with services").to_numpy())
        - 0.30 * df.followup_visit_scheduled.to_numpy()
        + 0.12 * (df.admission_source.eq("Emergency").to_numpy())
        + 0.10 * (df.weekend_discharge.to_numpy())
        + 0.08 * np.nan_to_num((df.lab_creatinine_last.to_numpy() - 1.0), nan=0.0).clip(0, 3)
        + 0.05 * np.nan_to_num((140 - df.lab_sodium_last.to_numpy()), nan=0.0).clip(0, 15) / 5
        + 0.06 * np.nan_to_num((12 - df.lab_hemoglobin_last.to_numpy()), nan=0.0).clip(0, 6) / 2
        + 0.04 * np.nan_to_num((df.heart_rate_last.to_numpy() - 80), nan=0.0).clip(0, 50) / 10
        + rng.normal(0, 0.65, len(df))
    )
    intercept = calibrate_intercept(lp, 0.15)
    probability = 1 / (1 + np.exp(-np.clip(lp + intercept, -30, 30)))
    df["readmitted_30d"] = rng.binomial(1, probability).astype(int)

    train_mask = pd.to_datetime(df.admit_date) < pd.Timestamp("2024-01-01")
    train = df.loc[train_mask, TRAIN_COLUMNS].copy()
    holdout = df.loc[~train_mask, HOLDOUT_COLUMNS].copy()
    # Keep dates as ISO strings for portable CSV interchange.
    for frame in (train, holdout):
        frame["admit_date"] = pd.to_datetime(frame.admit_date).dt.strftime("%Y-%m-%d")
        frame["discharge_date"] = pd.to_datetime(frame.discharge_date).dt.strftime("%Y-%m-%d")
    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "train.csv", index=False)
    holdout.to_csv(OUT / "holdout.csv", index=False)

    print(f"Training rows/columns: {train.shape}; readmission prevalence: {train.readmitted_30d.mean():.3%}")
    print(f"Holdout rows/columns: {holdout.shape}; internal prevalence diagnostic: {df.loc[~train_mask, 'readmitted_30d'].mean():.3%}")
    print("Missingness in specified clinical fields (train):")
    print(train[["lab_creatinine_last", "lab_sodium_last", "lab_hemoglobin_last", "systolic_bp_last", "heart_rate_last"]].isna().mean().round(4).to_string())
    multi_share = (pd.Series(event_patient_ids).value_counts() > 1).mean()
    print(f"Patients with multiple admissions: {multi_share:.3%}")
    print(f"Wrote: {OUT / 'train.csv'} and {OUT / 'holdout.csv'}")


if __name__ == "__main__":
    generate()
