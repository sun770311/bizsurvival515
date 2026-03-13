"""
Prediction tools for the NYC Business Survival project Streamlit app.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


STANDARD_SURVIVAL_TIMES = [12, 36, 60, 120]


def predict_logistic_profile(
    pipeline: Any,
    profile_df: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Predict the survival probability and class for a given profile.
    """
    try:
        probability = float(pipeline.predict_proba(profile_df)[0, 1])
        predicted_class = int(pipeline.predict(profile_df)[0])
    except Exception as exc:
        raise RuntimeError(
            "Failed to run logistic prediction. "
            "The saved sklearn pipeline is likely incompatible with the "
            "current sklearn version in this environment."
        ) from exc

    return {
        "predicted_survival_probability": probability,
        "predicted_class": predicted_class,
    }


def predict_standard_cox_profile(
    model: Any,
    scaler: Any,
    kept_columns: list[str],
    profile_df: pd.DataFrame,
    survival_times: list[int] | None = None,
) -> dict[str, float]:
    """
    Predict the partial hazard and survival probabilities for a given profile.
    """
    times = survival_times or STANDARD_SURVIVAL_TIMES

    feature_df = profile_df[kept_columns].copy()
    scaled_array = scaler.transform(feature_df)
    scaled_df = pd.DataFrame(scaled_array, columns=kept_columns, index=feature_df.index)

    partial_hazard = float(model.predict_partial_hazard(scaled_df).iloc[0])
    survival_df = model.predict_survival_function(scaled_df, times=times).T

    result: dict[str, float] = {"partial_hazard": partial_hazard}
    for month in times:
        result[f"survival_prob_{month}m"] = float(survival_df[month].iloc[0])

    return result


def predict_time_varying_cox_profile(
    model: Any,
    scaler: Any,
    kept_columns: list[str],
    profile_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Predict the partial hazard for a given time-varying profile.
    """
    feature_df = profile_df[kept_columns].copy()
    scaled_array = scaler.transform(feature_df)
    scaled_df = pd.DataFrame(scaled_array, columns=kept_columns, index=feature_df.index)

    partial_hazard = float(model.predict_partial_hazard(scaled_df).iloc[0])

    return {"partial_hazard": partial_hazard}


def predict_time_varying_cox_profiles(
    model: Any,
    scaler: Any,
    kept_columns: list[str],
    profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict the partial hazard for time-varying profiles.
    """
    feature_df = profiles_df[kept_columns].copy()
    scaled_array = scaler.transform(feature_df)
    scaled_df = pd.DataFrame(scaled_array, columns=kept_columns, index=feature_df.index)

    partial_hazards = model.predict_partial_hazard(scaled_df)

    results = profiles_df.copy()
    results["partial_hazard"] = partial_hazards.to_numpy(dtype=float)
    return results


def top_positive_negative(
    summary_df: pd.DataFrame,
    coefficient_col: str,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the top positive and negative coefficients from the summary dataframe.
    """
    ordered = summary_df.sort_values(coefficient_col, ascending=False).copy()
    positive = ordered.head(top_n)
    negative = ordered.tail(top_n).sort_values(coefficient_col, ascending=True)
    return positive, negative
