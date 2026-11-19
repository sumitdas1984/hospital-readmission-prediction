import os
import requests
import pandas as pd
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Readmission Care Call List", layout="wide")
st.title("30-Day Readmission Risk — Weekly Call List")
st.caption("Decision-support prototype for care-management prioritization")

st.warning(
    "Scores are model-generated risk estimates, not diagnoses or guarantees. "
    "Use them alongside clinical judgment and local care protocols."
)

left, right = st.columns([1, 3])
with left:
    if st.button("Refresh call list", type="primary"):
        st.cache_data.clear()

try:
    response = requests.get(f"{API_BASE}/call-list", timeout=60)
    response.raise_for_status()
    payload = response.json()
except requests.RequestException as exc:
    st.error(f"Cannot reach backend API at {API_BASE}. Start the API and try again. Details: {exc}")
    st.stop()

predictions = payload.get("predictions", [])
st.subheader("Weekly capacity")
c1, c2, c3 = st.columns(3)
c1.metric("Admissions scored", f"{payload.get('total_admissions', 0):,}")
c2.metric("Available call slots (5%)", f"{payload.get('call_slots', 0):,}")
c3.metric("Patients in this list", f"{payload.get('returned', 0):,}")

st.info(payload.get("note", "Ranked list for review."))

if not predictions:
    st.info("No records returned.")
    st.stop()

df = pd.DataFrame(predictions)
df.insert(0, "rank", range(1, len(df) + 1))
df["risk_score_pct"] = (df["risk_score"] * 100).round(1)
display = df[["rank", "admission_id", "patient_id", "risk_score_pct", "risk_band"]].rename(
    columns={"risk_score_pct": "Risk score (%)", "risk_band": "Relative risk band"}
)
st.subheader("Ranked call list")
st.dataframe(display, use_container_width=True, hide_index=True)

st.subheader("Patient-level review")
selected = st.selectbox(
    "Select an admission",
    options=df["admission_id"].tolist(),
    format_func=lambda x: f"Rank {int(df.loc[df.admission_id == x, 'rank'].iloc[0])} — {x}",
)
row = df.loc[df["admission_id"] == selected].iloc[0]
a, b, c = st.columns(3)
a.metric("Rank", int(row["rank"]))
b.metric("Risk score", f"{row['risk_score']*100:.1f}%")
c.metric("Risk band", row["risk_band"])

st.markdown("**Signals to review**")
for reason in row["reasons"]:
    st.write(f"- {reason}")

st.caption(
    "The signals above are simple descriptive rules over available fields. "
    "They are not causal explanations and may not fully explain the model score."
)
