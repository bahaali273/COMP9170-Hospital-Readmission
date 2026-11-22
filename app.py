
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


    # Model inference sanity check (synthetic input only)
    import numpy as np

    st.subheader("Model Inference Sanity Check")

    # Synthetic zero-valued feature vector.
    # This is NOT a clinically valid patient record.
    synthetic_input = pd.DataFrame(
        np.zeros((1, len(rf_bundle["features"]))),
        columns=rf_bundle["features"]
    )

    probabilities = rf_bundle["model"].predict_proba(
        synthetic_input
    )[0]

    positive_idx = list(
        rf_bundle["model"].classes_
    ).index(1)

    synthetic_probability = float(
        probabilities[positive_idx]
    )

    assert np.isfinite(synthetic_probability)
    assert 0.0 <= synthetic_probability <= 1.0

    st.success("Model inference test passed!")
    st.write("Output shape:", probabilities.shape)
    st.write("Probability sum:", round(float(probabilities.sum()), 6))
    st.caption(
        "Synthetic input is used exclusively to verify model "
        "execution. This output must not be interpreted as a "
        "patient risk prediction."
    )

    
    # -------------------------------------------
    # Patient Prediction — Feature Schema
    # -------------------------------------------

    st.divider()
    st.header("Patient Prediction — Feature Schema")

    feature_names = rf_bundle["features"]

    assert len(feature_names) == 133
    assert len(set(feature_names)) == 133

    feature_schema = pd.DataFrame({
        "Feature Index": range(1, len(feature_names) + 1),
        "Feature Name": feature_names
    })

    st.success("Feature schema verified successfully!")

    st.write("Required input features:", len(feature_names))

    with st.expander("View all 133 model features"):
        st.dataframe(
            feature_schema,
            hide_index=True,
            width="stretch"
        )

    st.download_button(
        label="Download Feature Schema",
        data=feature_schema.to_csv(index=False),
        file_name="random_forest_feature_schema.csv",
        mime="text/csv"
    )

    st.caption(
        "The prediction form will be constructed using "
        "the model's exact feature schema. "
        "Patient predictions are not enabled yet."
    )


    
    # -------------------------------------------
    # Visit Preprocessing Module Test
    # -------------------------------------------

    from src.patient_preprocessing import add_visit_features

    st.divider()
    st.header("Patient Preprocessing Verification")

    example_patient = pd.DataFrame([{
        "number_outpatient": 2,
        "number_emergency": 1,
        "number_inpatient": 3
    }])

    processed_patient = add_visit_features(example_patient)

    assert processed_patient["total_visits"].iloc[0] == 6

    assert np.isclose(
        processed_patient["log1p_total_visits"].iloc[0],
        np.log(7)
    )

    st.success("Patient preprocessing module test passed!")

    st.dataframe(
        processed_patient.round(4),
        hide_index=True
    )

    st.caption(
        "Synthetic example only. This verifies visit-related "
        "feature engineering, not clinical prediction accuracy."
    )


    
    # -------------------------------------------
    # Demographic Preprocessing Verification
    # -------------------------------------------

    from src.patient_preprocessing import add_demographic_features

    st.divider()
    st.header("Demographic Preprocessing Verification")

    example_demographics = pd.DataFrame([
        {"age": "[60-70)", "gender": "Male"},
        {"age": "[30-40)", "gender": "Female"}
    ])

    processed_demographics = add_demographic_features(
        example_demographics
    )

    assert processed_demographics["age_mid"].tolist() == [65, 35]
    assert processed_demographics["gender_male"].tolist() == [1, 0]

    st.success("Demographic preprocessing test passed!")

    st.dataframe(
        processed_demographics,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Synthetic examples verify age and gender transformations. "
        "Patient prediction remains disabled."
    )

    
    # -------------------------------------------
    # Medication Preprocessing Verification
    # -------------------------------------------

    from src.patient_preprocessing import add_medication_features

    st.divider()
    st.header("Medication Preprocessing Verification")

    example_medications = pd.DataFrame([
        {
            "metformin": "Steady",
            "insulin": "Up",
            "glipizide": "No"
        },
        {
            "metformin": "No",
            "insulin": "Down",
            "glipizide": "Steady"
        }
    ])

    medication_columns = [
        "metformin",
        "insulin",
        "glipizide"
    ]

    processed_medications = add_medication_features(
        example_medications,
        medication_columns
    )

    assert processed_medications["n_meds_on"].tolist() == [2, 2]
    assert processed_medications["n_med_changes"].tolist() == [1, 1]

    st.success("Medication preprocessing test passed!")

    st.dataframe(
        processed_medications,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Synthetic medication examples verify counting logic. "
        "The full medication list and end-to-end preprocessing "
        "have not yet been validated."
    )
        # -------------------------------------------
    # Diagnosis Preprocessing Verification
    # -------------------------------------------

    from src.patient_preprocessing import add_diagnosis_features

    st.divider()
    st.header("Diagnosis Preprocessing Verification")

    example_diagnoses = pd.DataFrame([
        {
            "diag_1": "250.13",
            "diag_2": "428",
            "diag_3": "486"
        },
        {
            "diag_1": "715",
            "diag_2": "V45",
            "diag_3": None
        }
    ])

    processed_diagnoses = add_diagnosis_features(
        example_diagnoses
    )

    expected_groups = [
        ["Diabetes", "Circulatory", "Respiratory"],
        ["Musculoskeletal", "Other", "Missing"]
    ]

    actual_groups = processed_diagnoses[
        ["diag_1_group", "diag_2_group", "diag_3_group"]
    ].values.tolist()

    assert actual_groups == expected_groups

    st.success("Diagnosis preprocessing test passed!")

    st.dataframe(
        processed_diagnoses,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Synthetic ICD-9 examples verify diagnosis grouping. "
        "This is not a clinical diagnosis tool."
    )


    # -------------------------------------------
    # One-Hot Encoding Verification
    # -------------------------------------------

    from src.patient_preprocessing import encode_model_features

    st.divider()
    st.header("One-Hot Encoding Verification")

    example_data = pd.DataFrame([
        {
            "age_mid": 65,
            "race": "Caucasian",
            "admission_type": "Emergency"
        },
        {
            "age_mid": 35,
            "race": "Asian",
            "admission_type": "Elective"
        }
    ])

    categorical_columns = [
        "race",
        "admission_type"
    ]

    expected_features = [
        "age_mid",
        "race_Asian",
        "race_Caucasian",
        "admission_type_Elective",
        "admission_type_Emergency"
    ]

    encoded_data = encode_model_features(
        example_data,
        categorical_columns,
        expected_features
    )

    assert encoded_data.columns.tolist() == expected_features
    assert encoded_data.shape == (2, 5)
    assert encoded_data["race_Caucasian"].tolist() == [1, 0]
    assert encoded_data["admission_type_Emergency"].tolist() == [1, 0]

    st.success("One-hot encoding test passed!")

    st.dataframe(
        encoded_data,
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "Synthetic test of categorical encoding and column "
        "alignment. Full 133-feature preprocessing is not "
        "yet validated."
    )

    
    # -------------------------------------------
    # Batch Prediction — Prepared Feature CSV
    # -------------------------------------------

    st.divider()
    st.header("Hospital Readmission Prediction")

    st.info(
        "Research demonstration only. Upload prepared "
        "133-feature data. Not for clinical decisions."
    )

    uploaded = st.file_uploader(
        "Upload prepared patient features (CSV)",
        type=["csv"],
        key="prediction_csv"
    )

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            expected = rf_bundle["features"]

            # Allow test CSVs containing the known label,
            # but never use the label as a predictor.
            X_input = input_df.drop(
                columns=["readmit_30"],
                errors="ignore"
            )

            missing = sorted(
                set(expected) - set(X_input.columns)
            )
            extra = sorted(
                set(X_input.columns) - set(expected)
            )

            if missing or extra:
                raise ValueError(
                    f"Missing features: {missing}\n"
                    f"Unexpected columns: {extra}"
                )

            X_input = X_input.loc[:, expected]

            if X_input.empty:
                raise ValueError("CSV contains no records.")

            if len(X_input) > 1000:
                raise ValueError(
                    "Upload at most 1000 records."
                )

            if not all(
                pd.api.types.is_numeric_dtype(dtype)
                for dtype in X_input.dtypes
            ):
                raise ValueError(
                    "All features must be numeric."
                )

            values = X_input.to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    "Features contain missing or "
                    "infinite values."
                )

            st.success(
                f"Validated {len(X_input)} records "
                "with 133 features."
            )

            if st.button("Run Predictions"):
                classes = list(
                    rf_bundle["model"].classes_
                )
                positive_index = classes.index(1)

                probabilities = (
                    rf_bundle["model"]
                    .predict_proba(X_input)[:, positive_index]
                )

                predictions = (
                    probabilities >= rf_bundle["threshold"]
                ).astype(int)

                results_df = pd.DataFrame({
                    "Readmission Probability": probabilities,
                    "Predicted Class": predictions
                })

                st.dataframe(
                    results_df,
                    hide_index=True,
                    width="stretch"
                )

                st.download_button(
                    "Download Predictions",
                    results_df.to_csv(index=False),
                    file_name="readmission_predictions.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Predictions use the model's "
                    "validation-selected threshold. "
                    "These scores are not calibrated "
                    "clinical risk estimates."
                )

        except (ValueError, KeyError, TypeError) as error:
            st.error(f"Invalid input: {error}")

except Exception as error:
    st.error("Unable to load Random Forest model.")
    st.code(str(error))




