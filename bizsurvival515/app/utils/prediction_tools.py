"""Provide prediction helpers for logistic, standard Cox, and time-varying Cox models.

This module centralizes reusable prediction utilities for the NYC Business
Survival Streamlit app. It provides helpers for scoring single-row logistic
profiles, single-row standard Cox profiles, and single-row or multi-row
time-varying Cox profiles using already loaded model artifacts. It also
includes a utility for extracting the top positive and negative coefficients
from model summary tables for display in the app.

Constants:
- STANDARD_SURVIVAL_TIMES:
  Default survival horizons, in months, used for standard Cox predictions.

Functions:
- predict_logistic_profile:
  Predict survival probability and class for a logistic profile.
- predict_standard_cox_profile:
  Predict partial hazard and survival probabilities for a standard Cox profile.
- predict_time_varying_cox_profile:
  Predict partial hazard for a single time-varying Cox profile.
- predict_time_varying_cox_profiles:
  Predict partial hazards for multiple time-varying Cox profiles.
- top_positive_negative:
  Extract the top positive and negative coefficients from a model summary table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


STANDARD_SURVIVAL_TIMES = [12, 36, 60, 120]


def predict_logistic_profile(
    pipeline: Any,
    profile_df: pd.DataFrame,
) -> dict[str, float | int]:
    """Predict survival probability and class for a logistic model profile.

    Args:
        pipeline: Fitted logistic-model pipeline implementing ``predict_proba``
            and ``predict``.
        profile_df: Single-row profile dataframe aligned to the model's expected
            feature schema.

    Returns:
        A dictionary containing:
        - ``predicted_survival_probability``: Predicted probability of survival.
        - ``predicted_class``: Predicted binary class label.

    Raises:
        RuntimeError: If prediction fails, typically due to artifact or
            sklearn-version incompatibility.
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
    """Predict partial hazard and survival probabilities for a standard Cox profile.

    Args:
        model: Fitted standard Cox model implementing survival and hazard prediction.
        scaler: Fitted feature scaler used to transform model inputs.
        kept_columns: Feature columns expected by the trained Cox model.
        profile_df: Single-row profile dataframe aligned to the model input schema.
        survival_times: Optional survival horizons, in months, at which to compute
            survival probabilities. Defaults to ``STANDARD_SURVIVAL_TIMES``.

    Returns:
        A dictionary containing the profile's partial hazard and survival
        probabilities at the requested time horizons.

    Raises:
        KeyError: If one or more required kept columns are missing from ``profile_df``.
        ValueError: If the scaler or model receives incompatible input dimensions.
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
    """Predict partial hazard for a single time-varying Cox profile.

    Args:
        model: Fitted time-varying Cox model implementing partial-hazard prediction.
        scaler: Fitted feature scaler used to transform model inputs.
        kept_columns: Feature columns expected by the trained time-varying Cox model.
        profile_df: Single-row profile dataframe aligned to the model input schema.

    Returns:
        A dictionary containing the profile's predicted partial hazard.

    Raises:
        KeyError: If one or more required kept columns are missing from ``profile_df``.
        ValueError: If the scaler or model receives incompatible input dimensions.
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
    """Predict partial hazards for multiple time-varying Cox profiles.

    Args:
        model: Fitted time-varying Cox model implementing partial-hazard prediction.
        scaler: Fitted feature scaler used to transform model inputs.
        kept_columns: Feature columns expected by the trained time-varying Cox model.
        profiles_df: Multi-row profile dataframe aligned to the model input schema.

    Returns:
        A copy of the input dataframe with an added ``partial_hazard`` column.

    Raises:
        KeyError: If one or more required kept columns are missing from ``profiles_df``.
        ValueError: If the scaler or model receives incompatible input dimensions.
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
    """Extract the top positive and top negative coefficients from a summary table.

    Args:
        summary_df: Model summary dataframe containing a coefficient column.
        coefficient_col: Name of the coefficient column to rank by.
        top_n: Number of top positive and top negative rows to return.

    Returns:
        A tuple containing:
        - A dataframe of the top positive coefficients.
        - A dataframe of the top negative coefficients.
    """
    ordered = summary_df.sort_values(coefficient_col, ascending=False).copy()
    positive = ordered.head(top_n)
    negative = ordered.tail(top_n).sort_values(coefficient_col, ascending=True)
    return positive, negative
