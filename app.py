
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Hospital Readmission Predictor")
st.caption("COMP 9170 — Data Science Research Project")

st.info(
    "Research and educational demonstration only. "
    "Not intended for clinical decision-making."
)

st.header("Model Performance Comparison")

metrics_path = Path("results/model_comparison_final.csv")

if metrics_path.exists():
    df = pd.read_csv(metrics_path)

    st.dataframe(df, use_container_width=True)

    st.subheader("PR-AUC Comparison")
    st.bar_chart(
        df.set_index("Model")[["PR-AUC"]]
    )

    st.caption(
        "Results are based on the held-out test dataset. "
        "Kumo-Tabular is excluded from this comparison."
    )
else:
    st.error("Model comparison metrics file not found.")

st.divider()
st.write(
    "Patient-level prediction and explainability "
    "will be added after model deployment validation."
)
