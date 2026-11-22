"""COMP 9170 — Hospital Readmission Research Dashboard.

Run from the repository root: streamlit run app.py
Prediction accepts *preprocessed* 133-feature CSVs only.
"""

import hashlib
import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "model_comparison_final.csv"
PARTS = [ROOT / "models" / f"rf_model.part{i:02d}" for i in range(1, 4)]
MAX_ROWS = 1000
EXPECTED_FEATURES = 133
EXPECTED_THRESHOLD = 0.33

st.set_page_config(
    page_title="Hospital Readmission | COMP 9170",
    page_icon="🏥",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_comparison():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Missing comparison file: {RESULTS_PATH}")
    df = pd.read_csv(RESULTS_PATH)
    required = {"Model", "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Comparison CSV missing columns: {sorted(missing)}")
    return df


@st.cache_resource(show_spinner="Loading the Random Forest model…")
def load_random_forest():
    missing = [p.name for p in PARTS if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model parts: {', '.join(missing)}")

    buffer = io.BytesIO()
    digest = hashlib.sha256()
    for path in PARTS:
        data = path.read_bytes()
        buffer.write(data)
        digest.update(data)
    buffer.seek(0)

    # joblib loads Python pickle content; only load files from your trusted repository.
    bundle = joblib.load(buffer)
    model = bundle["model"]
    features = list(bundle["feature_names"])
    threshold = float(bundle["threshold"])

    if len(features) != EXPECTED_FEATURES or len(set(features)) != EXPECTED_FEATURES:
        raise ValueError("Model bundle contains an invalid 133-feature schema")
    if model.n_features_in_ != EXPECTED_FEATURES:
        raise ValueError("Model input feature count does not match the bundle")
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != features:
        raise ValueError("Model feature ordering does not match the saved schema")
    if not np.isclose(threshold, EXPECTED_THRESHOLD, rtol=0, atol=1e-9):
        raise ValueError("Unexpected classification threshold")
    if 1 not in list(model.classes_):
        raise ValueError("Model does not contain positive class 1")

    return {
        "model": model,
        "features": features,
        "threshold": threshold,
        "sha256": digest.hexdigest(),
        "size_mb": buffer.getbuffer().nbytes / (1024 * 1024),
    }


def validate_uploaded_csv(uploaded, feature_names):
    df = pd.read_csv(uploaded)
    if df.empty:
        raise ValueError("The CSV contains no records")
    if len(df) > MAX_ROWS:
        raise ValueError(f"Upload at most {MAX_ROWS:,} records (received {len(df):,})")
    if not df.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names")

    # Labels may be present in previously prepared test data, but are NEVER predictors.
    x = df.drop(columns=["readmit_30"], errors="ignore")
    missing = sorted(set(feature_names) - set(x.columns))
    extra = sorted(set(x.columns) - set(feature_names))
    if missing or extra:
        raise ValueError(f"Feature mismatch. Missing: {missing}; Unexpected: {extra}")

    x = x.loc[:, feature_names]
    for column in x.columns:
        if not pd.api.types.is_numeric_dtype(x[column]):
            raise ValueError(f"Non-numeric feature: {column}")

    values = x.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("CSV contains missing or infinite feature values")

    # Simple schema and data-quality checks; not a substitute for full preprocessing QA.
    binary_columns = [c for c in feature_names if c.startswith((
        "race_", "admission_type_", "discharge_group_", "admission_source_group_",
        "medical_specialty_", "payer_code_", "diag_1_group_", "diag_2_group_",
        "diag_3_group_",
    ))]
    # Keep medication dummies and other binary indicators in the separate training QA;
    # the feature list alone does not identify every categorical source unambiguously.
    for column in binary_columns:
        if not np.isin(x[column].to_numpy(), [0, 1]).all():
            raise ValueError(f"Invalid one-hot values in: {column}")
    return x


st.title("🏥 Hospital Readmission Research Dashboard")
st.caption("BCIT COMP 9170 · Predicting 30-day readmission using machine learning")
st.info(
    "Educational research demonstration. This app is not validated for clinical use. "
    "Do not upload identifiable patient information."
)

tab_compare, tab_predict, tab_methods = st.tabs([
    "📊 Model Comparison", "🔍 Readmission Prediction", "📚 Research & Methodology"
])

with tab_compare:
    st.header("Held-Out Test Model Comparison")
    try:
        comparison = load_comparison()
        st.dataframe(comparison, hide_index=True, width="stretch")
        st.subheader("Precision–Recall AUC (PR-AUC)")
        st.bar_chart(comparison.set_index("Model")[["PR-AUC"]], horizontal=True)
        st.caption(
            "Five models evaluated on the held-out test split. NVIDIA Kumo-Tabular "
            "is excluded because it used different development samples. "
            "Observed score differences have not been established as statistically significant."
        )
        st.markdown(
            "**Interpretation:** PR-AUC is especially informative because readmission "
            "is uncommon (~8.97% of records). The regularized MLP had the highest "
            "observed PR-AUC (0.1579); Random Forest had the highest ROC-AUC "
            "(0.6477), and XGBoost the highest F1 (0.2337)."
        )
    except (OSError, ValueError) as exc:
        st.error(f"Cannot display model comparison: {exc}")

with tab_predict:
    st.header("Predict from Prepared Features")
    st.write(
        "Upload a CSV produced using the project's original preprocessing pipeline. "
        f"Each row must contain exactly {EXPECTED_FEATURES} numeric model features. "
        "The optional `readmit_30` label column is ignored during prediction."
    )
    st.warning(
        "This interface does not convert raw clinical records into model features. "
        "It cannot be used to assess real patients or make healthcare decisions."
    )
    try:
        rf = load_random_forest()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Model", "Random Forest")
        col_b.metric("Features", len(rf["features"]))
        col_c.metric("Classification threshold", f'{rf["threshold"]:.2f}')
        with st.expander("Model deployment details"):
            st.write(f"Model size: {rf['size_mb']:.2f} MiB")
            st.write(f"Scikit-learn runtime: {sklearn.__version__}")
            st.code(f"SHA-256 (computed, not externally verified): {rf['sha256']}")
            st.caption("Model files should be trusted and their original checksum verified separately.")
            st.download_button(
                "Download feature schema",
                pd.DataFrame({
                    "Feature Index": range(1, len(rf["features"]) + 1),
                    "Feature Name": rf["features"],
                }).to_csv(index=False),
                "random_forest_feature_schema.csv",
                "text/csv",
            )
        uploaded = st.file_uploader(
            "Prepared feature CSV (up to 1,000 rows)",
            type=["csv"],
            key="prediction_csv",
        )
        if uploaded is not None:
            try:
                x = validate_uploaded_csv(uploaded, rf["features"])
                st.success(f"Validated {len(x):,} rows with {x.shape[1]} features")
                if st.button("Run Predictions", type="primary"):
                    pos_idx = list(rf["model"].classes_).index(1)
                    p = rf["model"].predict_proba(x)[:, pos_idx]
                    prediction = (p >= rf["threshold"]).astype(int)
                    output = pd.DataFrame({
                        "Readmission Probability": p,
                        "Predicted Class": prediction,
                    })
                    st.dataframe(output, hide_index=True, width="stretch")
                    st.download_button(
                        "Download predictions CSV",
                        output.to_csv(index=False),
                        "readmission_predictions.csv",
                        "text/csv",
                        key="download_predictions",
                    )
                    st.caption(
                        "Class 1 indicates predicted readmission within 30 days at "
                        "the validation-selected 0.33 threshold. Probabilities are not "
                        "calibrated clinical risk estimates."
                    )
            except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
                st.error(f"Invalid input: {exc}")
    except Exception as exc:
        st.error("The trained model could not be loaded. Check deployment and model-part files.")
        with st.expander("Technical details"):
            st.code(str(exc))

          
    # -------------------------------------------
    # SHAP Explainability Visualizations
    # -------------------------------------------

    st.divider()
    st.header("SHAP Model Explainability")

    st.write(
        "SHAP explains how individual features contribute "
        "to the Random Forest predictions. These are "
        "model explanations, not causal or clinical conclusions."
    )

    shap_images = [
        (
            "Global SHAP Feature Importance",
            "shap_importance.png",
            "Mean absolute SHAP values across 100 test records."
        ),
        (
            "SHAP Beeswarm",
            "shap_beeswarm.png",
            "Direction and magnitude of feature contributions "
            "across the same 100 records."
        ),
        (
            "Local SHAP — Example Patient",
            "shap_waterfall_patient_1.png",
            "Explanation of the first test record, "
            "with predicted probability approximately 0.0983."
        ),
    ]

    for title, filename, description in shap_images:
        image_path = ROOT / "results" / filename

        st.subheader(title)

        if image_path.is_file():
            st.image(str(image_path), width="stretch")
            st.caption(description)
        else:
            st.warning(f"Missing SHAP image: {filename}")

    st.info(
        "SHAP plots are based on a fixed research sample. "
        "The local waterfall explains the first test record, "
        "not a newly uploaded patient."
    )


    
# -------------------------------------------
# Manual Patient Input — Research Demo
# -------------------------------------------

with tab_predict:
    st.divider()
    st.header("Manual Patient Input")
    st.caption(
        "Enter synthetic patient information for a "
        "research demonstration. Manual predictions "
        "are not enabled until preprocessing is validated."
    )

    with st.form("manual_patient_form"):
        st.subheader("Patient Demographics")

        age = st.selectbox(
            "Age Group",
            [f"[{i}-{i+10})" for i in range(0, 100, 10)]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        race = st.selectbox(
            "Race",
            [
                "Caucasian",
                "AfricanAmerican",
                "Asian",
                "Hispanic",
                "Other"
            ]
        )

        st.subheader("Hospital Visit Information")

        time_in_hospital = st.number_input(
            "Length of Hospital Stay (days)",
            min_value=1,
            max_value=14,
            value=3
        )

        num_lab_procedures = st.number_input(
            "Number of Lab Procedures",
            min_value=0,
            value=40
        )

        num_procedures = st.number_input(
            "Number of Medical Procedures",
            min_value=0,
            value=1
        )

        num_medications = st.number_input(
            "Number of Medications",
            min_value=0,
            value=10
        )

        number_diagnoses = st.number_input(
            "Number of Diagnoses",
            min_value=1,
            value=5
        )

        st.subheader("Previous Hospital Visits")

        outpatient = st.number_input(
            "Previous Outpatient Visits",
            min_value=0,
            value=0
        )

        emergency = st.number_input(
            "Previous Emergency Visits",
            min_value=0,
            value=0
        )

        inpatient = st.number_input(
            "Previous Inpatient Visits",
            min_value=0,
            value=0
        )

        
        st.subheader("Diagnosis Information")

        diag_1 = st.text_input(
            "Primary ICD-9 Diagnosis",
            value="250.13"
        )

        diag_2 = st.text_input(
            "Secondary ICD-9 Diagnosis",
            value="428"
        )

        diag_3 = st.text_input(
            "Third ICD-9 Diagnosis",
            value="486"
        )

        st.subheader("Diabetes Medications")

        medication_options = ["No", "Steady", "Up", "Down"]

        metformin = st.selectbox(
            "Metformin",
            medication_options
        )

        insulin = st.selectbox(
            "Insulin",
            medication_options
        )

        glipizide = st.selectbox(
            "Glipizide",
            medication_options
        )

        st.caption(
            "Additional medication fields will be "
            "included after validating the complete "
            "training medication list."
        )

        
        # Hospital Admission Information
        st.subheader("Hospital Admission Information")

        admission_type = st.selectbox(
            "Admission Type",
            ["Emergency", "Urgent", "Elective", "Other"]
        )

        discharge_group = st.selectbox(
            "Discharge Disposition",
            ["Home", "Transferred", "Other", "Unknown"]
        )

        admission_source_group = st.selectbox(
            "Admission Source",
            ["Referral", "Transfer", "Other", "Emergency room"]
        )


        submitted = st.form_submit_button(
            "Review Patient Information"
        )

    
        # -------------------------------------------
        # Additional Clinical and Administrative Inputs
        # -------------------------------------------

        st.subheader("Additional Hospital Information")

        medical_specialty = st.selectbox(
            "Medical Specialty",
            [
                "Cardiology",
                "Emergency/Trauma",
                "Family/GeneralPractice",
                "InternalMedicine",
                "Nephrology",
                "Orthopedics",
                "Orthopedics-Reconstructive",
                "Radiologist",
                "Rare",
                "Surgery-General",
                "Unknown",
            ],
            index=10
        )

        payer_code = st.selectbox(
            "Payer Code",
            ["BC", "CM", "CP", "HM", "MC",
             "MD", "Rare", "SP", "UN", "Unknown"],
            index=9
        )

        max_glu_serum = st.selectbox(
            "Maximum Glucose Serum Test",
            ["Not measured", "Norm", ">200", ">300"]
        )

        A1Cresult = st.selectbox(
            "A1C Test Result",
            ["Not measured", "Norm", ">7", ">8"]
        )

        weight_recorded = st.checkbox(
            "Patient weight was recorded",
            value=False
        )

        st.subheader("Additional Diabetes Medications")

        repaglinide = st.selectbox(
            "Repaglinide", medication_options
        )

        glimepiride = st.selectbox(
            "Glimepiride", medication_options
        )

        glyburide = st.selectbox(
            "Glyburide", medication_options
        )

        pioglitazone = st.selectbox(
            "Pioglitazone", medication_options
        )

        rosiglitazone = st.selectbox(
            "Rosiglitazone", medication_options
        )

        st.subheader("Medication Treatment Information")

        med_change = st.checkbox(
            "Medication changed during hospitalization"
        )

        diabetes_med = st.checkbox(
            "Patient received diabetes medication"
        )

        # -------------------------------------------
        # Remaining 15 Medication Inputs
        # -------------------------------------------

        remaining_medications = [
            "nateglinide",
            "chlorpropamide",
            "acetohexamide",
            "tolbutamide",
            "acarbose",
            "miglitol",
            "troglitazone",
            "tolazamide",
            "examide",
            "citoglipton",
            "glyburide-metformin",
            "glipizide-metformin",
            "glimepiride-pioglitazone",
            "metformin-rosiglitazone",
            "metformin-pioglitazone",
        ]

        with st.expander("Other Diabetes Medications (15)"):
            st.caption(
                "Select the status of every medication. "
                "No means the medication was not prescribed."
            )

            extra_medications = {}

            for medication in remaining_medications:
                extra_medications[medication] = st.selectbox(
                    medication,
                    ["No", "Steady", "Up", "Down"],
                    key=f"extra_med_{medication}"
                )


   
    if submitted:
        patient_info = {
            "age": age,
            "gender": gender,
            "race": race,
            "time_in_hospital": time_in_hospital,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_diagnoses": number_diagnoses,
            "number_outpatient": outpatient,
            "number_emergency": emergency,
            "number_inpatient": inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "metformin": metformin,
            "insulin": insulin,
            "glipizide": glipizide,
            "admission_type": admission_type,
            "discharge_group": discharge_group,
            "admission_source_group": admission_source_group,
            "medical_specialty": medical_specialty,
            "payer_code": payer_code,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            "weight_recorded": int(weight_recorded),
            "repaglinide": repaglinide,
            "glimepiride": glimepiride,
            "glyburide": glyburide,
            "pioglitazone": pioglitazone,
            "rosiglitazone": rosiglitazone,
            "med_change": int(med_change),
            "diabetes_med": int(diabetes_med),
        }

        # Include all additional medication responses
        patient_info.update(extra_medications)

        st.success("Patient information collected!")

        with st.expander("Review Patient Information"):
            st.dataframe(
                pd.DataFrame([patient_info]),
                hide_index=True,
                width="stretch"
            )

        
        st.subheader("Manual Random Forest Prediction")

        try:
            import io
            import joblib
            from pathlib import Path
            from src.manual_prediction import manual_to_model_row

            model_parts = [
                Path("models") / f"rf_model.part{i:02d}"
                for i in range(1, 4)
            ]

            model_bytes = b"".join(
                part.read_bytes() for part in model_parts
            )

            rf_bundle = joblib.load(io.BytesIO(model_bytes))

            manual_features = manual_to_model_row(
                patient_info,
                list(rf_bundle["feature_names"])
            )

            model = rf_bundle["model"]
            classes = list(model.classes_)
            positive_idx = classes.index(1)

            probability = float(
                model.predict_proba(manual_features)[0, positive_idx]
            )

            threshold = float(rf_bundle["threshold"])
            predicted_class = int(probability >= threshold)

            st.success("Manual input processed successfully!")

            col1, col2 = st.columns(2)

            col1.metric(
                "Model Readmission Score",
                f"{probability:.2%}"
            )

            col2.metric(
                "Predicted Class",
                predicted_class
            )

            st.caption(
                f"Classification threshold: {threshold:.2f}"
            )

            st.warning(
                "Research demonstration only. "
                "Manual preprocessing has not been independently "
                "validated against the original training pipeline. "
                "Not for clinical use."
            )

        except Exception as err:
            st.error(
                f"Manual prediction failed: "
                f"{type(err).__name__}: {err}"
            )



with tab_methods:
    st.header("Research & Methodology")

    st.markdown(
        """
**Research question.** How do traditional and advanced machine learning methods
compare when predicting 30-day readmission following hospital encounters?

**Dataset.** UCI Diabetes 130-US Hospitals (1999–2008). After cleaning and retaining
one encounter per patient, the project uses 69,970 records. The held-out test set
has 13,994 records and approximately 8.97% positive labels.

**Models evaluated.** Logistic Regression, Random Forest, XGBoost, regularized
multilayer perceptron (MLP), and a patient-similarity GraphSAGE experiment.

**Evaluation.** PR-AUC (primary comparison), ROC-AUC, precision, recall, and F1.
Classification thresholds were selected on validation data and then applied to
the held-out test set without test-based tuning.

**Model deployed.** Random Forest, with 500 trees and a validation-selected
classification threshold of 0.33. It is not necessarily the best performer on
every metric; it is used here to demonstrate reproducible batch inference.

**Limitations.** The dataset is historical, class-imbalanced, and does not prove
deployment suitability at a new hospital. Model scores may not be calibrated.
The manual input preprocessing path has not yet been independently validated
against the original training pipeline. This prototype is not a clinical
decision-support system.

**Interpretability.** Global SHAP feature importance, beeswarm plots,
and example patient-level waterfall explanations are included as research
demonstrations. They are not causal or clinical conclusions.
"""
    )

st.divider()
st.caption("COMP 9170 · Research prototype · No clinical use")
