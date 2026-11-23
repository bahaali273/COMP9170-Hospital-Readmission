"""Manual research-demo inputs -> trained Random Forest 133-feature schema.

This module expects *already grouped* administrative categories from the form.
It must not be used for clinical decision making. It does not perform calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from src.patient_preprocessing import (
    add_visit_features,
    add_demographic_features,
    add_medication_features,
    add_diagnosis_features,
)

MEDICATION_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]
RETAINED_MEDICATIONS = [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
]
CATEGORICAL_COLUMNS = [
    "race", "admission_type", "discharge_group", "admission_source_group",
    "medical_specialty", "payer_code", "diag_1_group", "diag_2_group",
    "diag_3_group", "max_glu_serum", "A1Cresult", *RETAINED_MEDICATIONS,
]
NUMERIC_COLUMNS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "total_visits",
    "log1p_number_outpatient", "log1p_number_emergency",
    "log1p_number_inpatient", "log1p_total_visits", "age_mid",
    "n_meds_on", "n_med_changes", "gender_male", "med_change",
    "diabetes_med", "weight_recorded",
]
ALLOWED_CATEGORIES = {
    "race": ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other", "Unknown"],
    "admission_type": ["Elective", "Emergency", "Other", "Unknown", "Urgent"],
    "discharge_group": ["Home", "Home with care", "Left AMA", "Other", "Transferred", "Unknown"],
    "admission_source_group": ["Emergency room", "Other", "Referral", "Transfer", "Unknown"],
    "medical_specialty": ["Cardiology", "Emergency/Trauma", "Family/GeneralPractice", "InternalMedicine", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist", "Rare", "Surgery-General", "Unknown"],
    "payer_code": ["BC", "CM", "CP", "HM", "MC", "MD", "Rare", "SP", "UN", "Unknown"],
    "diag_1_group": ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Injury", "Missing", "Musculoskeletal", "Neoplasms", "Other", "Respiratory"],
    "diag_2_group": ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Injury", "Missing", "Musculoskeletal", "Neoplasms", "Other", "Respiratory"],
    "diag_3_group": ["Circulatory", "Diabetes", "Digestive", "Genitourinary", "Injury", "Missing", "Musculoskeletal", "Neoplasms", "Other", "Respiratory"],
    "max_glu_serum": [">200", ">300", "Norm", "Not measured"],
    "A1Cresult": [">7", ">8", "Norm", "Not measured"],
}
for _med in RETAINED_MEDICATIONS:
    ALLOWED_CATEGORIES[_med] = ["Down", "No", "Steady", "Up"]


def manual_to_model_row(patient_info: dict, feature_names: list[str]) -> pd.DataFrame:
    """Reject unsupported inputs and produce exact feature order for RF inference."""
    features = list(feature_names)
    if len(features) != 133 or len(set(features)) != 133:
        raise ValueError("Expected a unique 133-feature model schema")
    if not set(NUMERIC_COLUMNS).issubset(features):
        raise ValueError("Model schema differs from expected 20 numeric features")
    for prefix, categories in ALLOWED_CATEGORIES.items():
        columns_in_schema = {f for f in features if f.startswith(prefix + "_")}
        expected_cols = {prefix + "_" + val for val in categories}
        if columns_in_schema != expected_cols:
            raise ValueError(f"Model categorical schema mismatch: {prefix}")

    essential = [
        "age", "gender", "diag_1", "diag_2", "diag_3",
        "race", "admission_type", "discharge_group", "admission_source_group",
        "medical_specialty", "payer_code", "max_glu_serum", "A1Cresult",
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_diagnoses", "number_outpatient",
        "number_emergency", "number_inpatient", "weight_recorded",
        "med_change", "diabetes_med", *MEDICATION_COLUMNS,
    ]
    missing = [x for x in essential if x not in patient_info]
    if missing:
        raise ValueError(f"Missing manual form fields: {missing}")
    if patient_info["examide"] != "No" or patient_info["citoglipton"] != "No":
        raise ValueError("examide and citoglipton were constant 'No' in training")
    for med in MEDICATION_COLUMNS:
        if patient_info[med] not in ("No", "Steady", "Up", "Down"):
            raise ValueError(f"Invalid medication status: {med}")
    for name in ["weight_recorded", "med_change", "diabetes_med"]:
        if patient_info[name] not in (0, 1, False, True):
            raise ValueError(f"{name} must be 0 or 1")
    for name in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_diagnoses", "number_outpatient", "number_emergency", "number_inpatient"]:
        val = patient_info[name]
        if not isinstance(val, (int, np.integer)) or val < (1 if name in ("time_in_hospital", "number_diagnoses") else 0):
            raise ValueError(f"Invalid nonnegative count: {name}")
    if not 1 <= patient_info["time_in_hospital"] <= 14:
        raise ValueError("Hospital stay must be 1–14 days (training range)")
    for cat in ["race", "admission_type", "discharge_group", "admission_source_group", "medical_specialty", "payer_code", "max_glu_serum", "A1Cresult"]:
        if patient_info[cat] not in ALLOWED_CATEGORIES[cat]:
            raise ValueError(f"Unsupported {cat}: {patient_info[cat]}")
    # Prevent double counting: medication flags must agree with the 23 statuses.
    medications_on = sum(patient_info[m] != "No" for m in MEDICATION_COLUMNS)
    medications_changed = sum(patient_info[m] in ("Up", "Down") for m in MEDICATION_COLUMNS)
    if int(patient_info["diabetes_med"]) != int(medications_on > 0):
        raise ValueError("diabetes_med checkbox conflicts with medication statuses")
    if medications_changed > 0 and not int(patient_info["med_change"]):
        raise ValueError("med_change cannot be unchecked when doses changed")

    df = pd.DataFrame([patient_info])
    df = add_visit_features(df)
    df = add_demographic_features(df)
    df = add_medication_features(df, MEDICATION_COLUMNS)
    df = add_diagnosis_features(df)

    if not all(df[f"diag_{n}_group"].iloc[0] in ALLOWED_CATEGORIES[f"diag_{n}_group"] for n in (1, 2, 3)):
        raise ValueError("Diagnosis grouping does not match trained category labels")

    pre = df.loc[:, NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    # Training's pd.get_dummies(..., dtype=int) convention.
    encoded = pd.get_dummies(pre, columns=CATEGORICAL_COLUMNS, dtype=int)
    # Never silently lose an input category; the ALLOWED list above is comprehensive.
    extra = set(encoded.columns) - set(features)
    if extra:
        raise ValueError(f"Features not understood by the model: {sorted(extra)}")
    aligned = encoded.reindex(columns=features, fill_value=0)
    if aligned.shape != (1, 133) or not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("Invalid 133-feature model input")
    return aligned
