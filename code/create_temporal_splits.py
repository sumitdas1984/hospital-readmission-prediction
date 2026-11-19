#!/usr/bin/env python3
"""Create chronological train, validation and internal-test partitions.

Run from the repository root:
    uv run python code/create_temporal_splits.py

Input:  data/raw/train.csv
Output: data/processed/{train,validation,internal_test}.csv
        data/processed/temporal_split_summary.csv

The Jan-Mar 2024 holdout is deliberately not read by this script.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

DATE_COL = "admit_date"
TARGET_COL = "readmitted_30d"
PATIENT_COL = "patient_id"
ID_COL = "admission_id"

SPLITS = {
    "train": ("2021-01-01", "2022-12-31"),
    "validation": ("2023-01-01", "2023-06-30"),
    "internal_test": ("2023-07-01", "2023-12-31"),
}


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Training data not found: {INPUT}")

    df = pd.read_csv(INPUT, parse_dates=[DATE_COL, "discharge_date"])
    required = {DATE_COL, TARGET_COL, PATIENT_COL, ID_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[DATE_COL].isna().any():
        raise ValueError(f"{DATE_COL} contains missing or invalid dates.")
    if df[ID_COL].duplicated().any():
        raise ValueError("admission_id must be unique before splitting.")

    masks = {
        name: df[DATE_COL].between(pd.Timestamp(start), pd.Timestamp(end))
        for name, (start, end) in SPLITS.items()
    }
    assigned = sum(mask.astype(int) for mask in masks.values())
    if not (assigned == 1).all():
        raise ValueError(
            f"{int((assigned != 1).sum())} rows fall outside or overlap split ranges."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, rows = {}, []
    for name, mask in masks.items():
        part = df.loc[mask].copy().sort_values(DATE_COL).reset_index(drop=True)
        if part.empty:
            raise ValueError(f"{name} partition is empty.")
        path = OUTPUT_DIR / f"{name}.csv"
        part.to_csv(path, index=False)
        frames[name] = part
        rows.append({
            "split": name,
            "start_date": part[DATE_COL].min().date().isoformat(),
            "end_date": part[DATE_COL].max().date().isoformat(),
            "rows": len(part),
            "unique_patients": part[PATIENT_COL].nunique(),
            "readmission_prevalence": part[TARGET_COL].mean(),
            "output_file": str(path.relative_to(ROOT)),
        })

    # Returning patients across time periods are expected in this temporal design.
    patient_sets = {n: set(f[PATIENT_COL].dropna().unique()) for n, f in frames.items()}
    for earlier, later in [
        ("train", "validation"),
        ("train", "internal_test"),
        ("validation", "internal_test"),
    ]:
        overlap = patient_sets[earlier] & patient_sets[later]
        later_n = len(patient_sets[later])
        rows.append({
            "split": f"patient_overlap:{earlier}->{later}",
            "start_date": "",
            "end_date": "",
            "rows": len(overlap),
            "unique_patients": len(overlap),
            "readmission_prevalence": len(overlap) / later_n if later_n else 0.0,
            "output_file": f"{len(overlap)} shared patients; {len(overlap)/later_n:.1%} of later split",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT_DIR / "temporal_split_summary.csv", index=False)

    total = sum(len(f) for f in frames.values())
    if total != len(df):
        raise AssertionError(f"Partition total {total} does not equal source rows {len(df)}")

    print(summary.to_string(index=False))
    print(f"Total partitioned rows: {total:,}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
