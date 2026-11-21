
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

# -------------------------------------------
# Random Forest Model Loading
# -------------------------------------------

import io
import joblib
import hashlib
import sklearn

st.divider()
st.header("Random Forest — Deployment Verification")

MODEL_DIR = Path("models")

MODEL_PARTS = [
    MODEL_DIR / "rf_model.part01",
    MODEL_DIR / "rf_model.part02",
    MODEL_DIR / "rf_model.part03",
]


@st.cache_resource
def load_random_forest():

    # Confirm all model parts exist
    for part in MODEL_PARTS:
        if not part.exists():
            raise FileNotFoundError(
                f"Missing model part: {part}"
            )

    # Reconstruct the original model in memory
    model_bytes = bytearray()
    sha256 = hashlib.sha256()

    for part in MODEL_PARTS:
        with open(part, "rb") as file:
            data = file.read()
            model_bytes.extend(data)
            sha256.update(data)

    # Load only our own trusted model artifact
    bundle = joblib.load(
        io.BytesIO(model_bytes)
    )

    model = bundle["model"]
    feature_names = bundle["feature_names"]
    threshold = bundle["threshold"]

    if model.n_features_in_ != 133:
        raise ValueError("Unexpected feature count")

    if len(feature_names) != 133:
        raise ValueError("Invalid feature list")

    if abs(threshold - 0.33) > 1e-8:
        raise ValueError("Unexpected threshold")

    return {
        "model": model,
        "features": feature_names,
        "threshold": threshold,
        "sha256": sha256.hexdigest(),
        "size_mb": len(model_bytes) / 1024**2,
    }


try:
    rf_bundle = load_random_forest()

    st.success("Random Forest model loaded successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Input Features",
        len(rf_bundle["features"])
    )

    col2.metric(
        "Classification Threshold",
        rf_bundle["threshold"]
    )

    col3.metric(
        "Model Size",
        f'{rf_bundle["size_mb"]:.2f} MB'
    )

    st.write("Scikit-learn version:", sklearn.__version__)

    st.caption(
        "Model loading verified. Patient prediction "
        "is not enabled yet."
    )

except Exception as error:
    st.error("Unable to load Random Forest model.")
    st.code(str(error))
