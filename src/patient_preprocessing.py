
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

