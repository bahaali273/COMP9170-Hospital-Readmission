
import numpy as np
import pandas as pd


VISIT_COLUMNS = [
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]


def add_visit_features(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the visit-related feature engineering
    used in the training preprocessing notebook.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Patient records containing the three
        previous-visit count columns.

    Returns
    -------
    pd.DataFrame
        A copy containing total_visits and
        the corresponding log1p features.
    """
    result = patient_df.copy()

    for column in VISIT_COLUMNS:
        if column not in result.columns:
            raise ValueError(f"Missing required column: {column}")

        values = pd.to_numeric(result[column], errors="raise")

        if values.isna().any():
            raise ValueError(f"Missing values in {column}")

        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {column}")

        if (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(
                f"{column} must contain non-negative integers"
            )

        result[column] = values.astype(int)

    result["total_visits"] = result[VISIT_COLUMNS].sum(axis=1)

    for column in VISIT_COLUMNS + ["total_visits"]:
        result[f"log1p_{column}"] = np.log1p(result[column])

    return result

    
def add_demographic_features(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce Jason's age and gender feature engineering.

    Required columns:
    - age: original age interval, e.g. '[60-70)'
    - gender: 'Male' or 'Female'
    """
    result = patient_df.copy()

    required = ["age", "gender"]

    for column in required:
        if column not in result.columns:
            raise ValueError(f"Missing required column: {column}")

    # Age interval to midpoint
    age_start = result["age"].str.extract(
        r"\[(\d+)-"
    )[0]

    if age_start.isna().any():
        raise ValueError("Invalid age interval")

    result["age_mid"] = age_start.astype(int) + 5

    # Gender to binary indicator
    valid_genders = ["Male", "Female"]

    if not result["gender"].isin(valid_genders).all():
        raise ValueError("Gender must be Male or Female")

    result["gender_male"] = (
        result["gender"] == "Male"
    ).astype(int)

    return result


   
def add_medication_features(
    patient_df: pd.DataFrame,
    medication_columns: list[str]
) -> pd.DataFrame:
    """
    Reproduce medication count features from
    the original preprocessing notebook.

    Each medication status must be one of:
    No, Steady, Up, Down.
    """
    result = patient_df.copy()

    if not medication_columns:
        raise ValueError("Medication column list cannot be empty")

    valid_statuses = {"No", "Steady", "Up", "Down"}

    for column in medication_columns:
        if column not in result.columns:
            raise ValueError(f"Missing medication: {column}")

        if not result[column].isin(valid_statuses).all():
            raise ValueError(
                f"Invalid medication status in {column}"
            )

    # Number of medications being taken
    result["n_meds_on"] = (
        result[medication_columns] != "No"
    ).sum(axis=1)

    # Number of medications with changed doses
    result["n_med_changes"] = (
        result[medication_columns].isin(["Up", "Down"])
    ).sum(axis=1)

    return result


def icd9_group(code):
    """
    Group an ICD-9 diagnosis code using the
    original project's feature engineering rules.
    """
    if pd.isna(code):
        return "Missing"

    code = str(code).strip()

    if not code or code == "?":
        return "Missing"

    if code[0].upper() in ("V", "E"):
        return "Other"

    try:
        value = float(code)
    except ValueError:
        return "Other"

    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value < 460 or value == 785:
        return "Circulatory"
    if 460 <= value < 520 or value == 786:
        return "Respiratory"
    if 520 <= value < 580 or value == 787:
        return "Digestive"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or value == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasm"

    return "Other"


def add_diagnosis_features(patient_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the three original diagnosis columns
    into grouped categorical features.
    """
    result = patient_df.copy()

    for i in (1, 2, 3):
        source = f"diag_{i}"
        target = f"diag_{i}_group"

        if source not in result.columns:
            raise ValueError(f"Missing diagnosis column: {source}")

        result[target] = result[source].map(icd9_group)

    return result


def encode_model_features(
    patient_df: pd.DataFrame,
    nominal_columns: list[str],
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Encode already-engineered features using the
    same one-hot encoding convention as training.

    Rare-category merging and medication filtering
    must be applied before calling this function.
    """
    missing_columns = [
        col for col in nominal_columns
        if col not in patient_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing categorical columns: {missing_columns}"
        )

    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Duplicate model feature names")

    encoded = pd.get_dummies(
        patient_df,
        columns=nominal_columns,
        dtype=int
    )

    # Retain the exact feature order expected by
    # the trained Random Forest model.
    encoded = encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    if encoded.isna().any().any():
        raise ValueError("Encoded features contain missing values")

    return encoded

