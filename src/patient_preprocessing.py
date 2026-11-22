
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
