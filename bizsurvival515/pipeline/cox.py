"""Fit penalized Cox survival models from the joined monthly panel.

This module trains two survival-analysis variants using joined_dataset.csv:
a time-varying Cox model built from monthly business records and a standard
one-row-per-business Cox proportional hazards model.

Input:
- joined_dataset.csv

Processing steps:
- Load and validate the joined monthly business panel
- Build time-varying and business-level survival datasets
- Remove redundant or near-constant predictors
- Standardize retained predictors
- Fit penalized Cox survival models
- Save model artifacts and coefficient summaries for downstream use

Outputs:
- Time-varying Cox model artifacts
- Standard CoxPH model artifacts
- Coefficient summary CSV files

Classes:
- CoxConfig:
  Stores pipeline configuration, including input path, output directory,
  study end date, variance threshold, and Cox penalizer.
- PreparedModelData:
  Stores the prepared modeling dataframe together with the fitted scaler
  and feature-selection results used to create it.

Functions:
- get_model_drop_columns:
  Return columns that should be excluded before Cox model fitting.
- build_time_varying_panel:
  Construct the monthly time-varying survival panel with start, stop,
  and event columns.
- build_business_level_dataset:
  Construct the one-row-per-business survival dataset for standard CoxPH
  modeling.
- get_feature_columns:
  Return candidate predictor columns after excluding identifiers and
  outcome-related fields.
- select_nonconstant_features:
  Remove low-variance predictors using a variance-threshold filter.
- scale_features:
  Standardize retained predictor columns and return the scaled dataframe
  and fitted scaler.
- prepare_time_varying_model_data:
  Build the final scaled modeling dataframe for the time-varying Cox model.
- prepare_business_level_model_data:
  Build the final scaled modeling dataframe for the standard CoxPH model.
- fit_time_varying_cox_model:
  Fit a penalized Cox time-varying survival model.
- fit_standard_cox_model:
  Fit a penalized standard Cox proportional hazards model.
- build_coefficient_summary:
  Create a sorted coefficient summary dataframe for model inspection and export.
- save_time_varying_artifacts:
  Save fitted artifacts for the time-varying Cox pipeline.
- save_standard_cox_artifacts:
  Save fitted artifacts for the standard CoxPH pipeline.
- run_time_varying_pipeline:
  Run the complete time-varying Cox modeling workflow.
- run_standard_cox_pipeline:
  Run the complete standard CoxPH modeling workflow.
- run_full_pipeline:
  Run both Cox modeling workflows and return their results.
- parse_args:
  Parse command-line arguments into a CoxConfig instance.
- main:
  Execute the full Cox pipeline from the command line and print results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from bizsurvival515.pipeline.utils import (
    STUDY_END,
    VARIANCE_THRESHOLD,
    FeatureSelectionResult,
    add_standard_modeling_args,
    load_joined_dataset,
    save_dataframe_artifact,
    save_pickle_artifact,
    validate_joined_dataset,
)


DEFAULT_PENALIZER = 0.1


@dataclass(frozen=True)
class CoxConfig:
    """Store configuration values for the Cox modeling pipeline.

    Attributes:
        data_path: Path to the preprocessed joined monthly panel CSV.
        output_dir: Directory where model artifacts and summaries are saved.
        study_end: Final month included in the observation window.
        variance_threshold: Minimum variance required for a feature to be kept.
        penalizer: L2-style penalty strength passed to the Cox estimators.
    """

    data_path: Path
    output_dir: Path
    study_end: pd.Timestamp = STUDY_END
    variance_threshold: float = VARIANCE_THRESHOLD
    penalizer: float = DEFAULT_PENALIZER


@dataclass(frozen=True)
class PreparedModelData:
    """Store a prepared modeling dataset together with preprocessing artifacts.

    Attributes:
        modeling_df: Final dataframe used for model fitting.
        scaler: Fitted scaler used to standardize retained predictors.
        feature_selection: Result describing which features were kept or dropped.
    """

    modeling_df: pd.DataFrame
    scaler: StandardScaler
    feature_selection: FeatureSelectionResult


def get_model_drop_columns() -> list[str]:
    """Return columns that are excluded from Cox model fitting.

    Returns:
        A list of column names dropped before feature selection and model fitting.
    """
    return [
        "open",
        "business_category_sum",
        "complaint_sum",
        "total_311",
        "location_cluster",
        "months_since_first_license",
    ]


def build_time_varying_panel(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a monthly time-varying survival panel from the joined business data.

    Args:
        joined: Preprocessed joined monthly business panel.
        study_end: Final month to include in the survival panel.

    Returns:
        A dataframe with one row per business-month, including start, stop,
        and event columns for time-varying Cox modeling.
    """
    panel = joined.sort_values(["business_id", "month"]).copy()
    panel = panel.loc[panel["month"] <= study_end].copy()

    panel["start"] = panel["months_since_first_license"] - 1
    panel["stop"] = panel["months_since_first_license"]

    last_month_by_business = panel.groupby("business_id")["month"].transform("max")
    panel["event"] = (
        (panel["month"] == last_month_by_business) & (panel["month"] < study_end)
    ).astype(int)

    panel = panel.drop(columns=get_model_drop_columns(), errors="ignore").copy()
    return panel


def build_business_level_dataset(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build a one-row-per-business survival dataset for standard CoxPH modeling.

    Args:
        joined: Preprocessed joined monthly business panel.
        study_end: Final month to include when computing durations and events.

    Returns:
        A dataframe with one row per business containing static features,
        duration in months, and event status.
    """
    business_df = joined.sort_values(["business_id", "month"]).copy()
    business_df = business_df.loc[business_df["month"] <= study_end].copy()

    first_month = business_df.groupby("business_id")["month"].min()
    last_month = business_df.groupby("business_id")["month"].max()

    business_event = last_month.lt(study_end).astype(int).rename("event")
    observed_end = last_month.clip(upper=study_end)

    duration_months = (
        ((observed_end.dt.year - first_month.dt.year) * 12)
        + (observed_end.dt.month - first_month.dt.month)
    ).rename("duration_months")

    business_features = (
        business_df.sort_values(["business_id", "month"])
        .groupby("business_id")
        .first()
        .reset_index()
    )

    coxph_df = business_features.merge(
        duration_months.reset_index(),
        on="business_id",
        how="left",
    ).merge(
        business_event.reset_index(),
        on="business_id",
        how="left",
    )

    coxph_df = coxph_df.drop(
        columns=["month"] + get_model_drop_columns(),
        errors="ignore",
    ).copy()

    return coxph_df


def get_feature_columns(df: pd.DataFrame, exclude_columns: list[str]) -> list[str]:
    """Return predictor columns after excluding specified non-feature columns.

    Args:
        df: Input dataframe containing identifiers, outcomes, and predictors.
        exclude_columns: Columns to exclude from the returned feature list.

    Returns:
        A list of candidate feature column names.
    """
    return [column for column in df.columns if column not in exclude_columns]


def select_nonconstant_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    variance_threshold: float,
) -> FeatureSelectionResult:
    """Remove low-variance feature columns using a variance-threshold filter.

    Args:
        df: Dataframe containing candidate predictor columns.
        feature_columns: Names of columns to evaluate for variance filtering.
        variance_threshold: Minimum variance required for a feature to be kept.

    Returns:
        A FeatureSelectionResult containing kept and dropped feature names.

    Raises:
        ValueError: If no candidate feature columns are provided.
        ValueError: If no features remain after low-variance filtering.
    """
    if not feature_columns:
        raise ValueError("No candidate feature columns available for model fitting.")

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(df[feature_columns])

    kept_columns = df[feature_columns].columns[selector.get_support()].tolist()
    dropped_columns = [column for column in feature_columns if column not in kept_columns]

    if not kept_columns:
        raise ValueError("No features remain after low-variance filtering.")

    return FeatureSelectionResult(
        kept_columns=kept_columns,
        dropped_columns=dropped_columns,
    )


def scale_features(
    df: pd.DataFrame,
    kept_columns: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize retained predictor columns and return the scaled dataframe.

    Args:
        df: Dataframe containing the retained predictor columns.
        kept_columns: Names of feature columns to standardize.

    Returns:
        A tuple containing the scaled feature dataframe and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[kept_columns])

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=kept_columns,
        index=df.index,
    )
    return scaled_df, scaler


def prepare_time_varying_model_data(
    panel: pd.DataFrame,
    variance_threshold: float,
) -> PreparedModelData:
    """Prepare scaled modeling data and preprocessing artifacts for time-varying Cox fitting.

    Args:
        panel: Time-varying survival panel with business-month records.
        variance_threshold: Minimum variance required for a feature to be kept.

    Returns:
        A PreparedModelData object containing the final modeling dataframe,
        fitted scaler, and feature-selection result.

    Raises:
        ValueError: If no candidate features are available or none remain after
            variance filtering.
    """
    exclude_columns = ["business_id", "month", "start", "stop", "event"]
    feature_columns = get_feature_columns(panel, exclude_columns)

    selection = select_nonconstant_features(
        df=panel,
        feature_columns=feature_columns,
        variance_threshold=variance_threshold,
    )
    scaled_df, scaler = scale_features(panel, selection.kept_columns)

    modeling_df = pd.concat(
        [panel[["business_id", "month", "start", "stop", "event"]], scaled_df],
        axis=1,
    ).copy()

    return PreparedModelData(
        modeling_df=modeling_df,
        scaler=scaler,
        feature_selection=selection,
    )


def prepare_business_level_model_data(
    coxph_df: pd.DataFrame,
    variance_threshold: float,
) -> PreparedModelData:
    """Prepare scaled modeling data and preprocessing artifacts for standard CoxPH fitting.

    Args:
        coxph_df: One-row-per-business survival dataframe.
        variance_threshold: Minimum variance required for a feature to be kept.

    Returns:
        A PreparedModelData object containing the final modeling dataframe,
        fitted scaler, and feature-selection result.

    Raises:
        ValueError: If no candidate features are available or none remain after
            variance filtering.
    """
    exclude_columns = ["business_id", "duration_months", "event"]
    feature_columns = get_feature_columns(coxph_df, exclude_columns)

    selection = select_nonconstant_features(
        df=coxph_df,
        feature_columns=feature_columns,
        variance_threshold=variance_threshold,
    )
    scaled_df, scaler = scale_features(coxph_df, selection.kept_columns)

    modeling_df = pd.concat(
        [coxph_df[["business_id", "duration_months", "event"]], scaled_df],
        axis=1,
    ).copy()

    return PreparedModelData(
        modeling_df=modeling_df,
        scaler=scaler,
        feature_selection=selection,
    )


def fit_time_varying_cox_model(
    modeling_df: pd.DataFrame,
    penalizer: float,
) -> CoxTimeVaryingFitter:
    """Fit a penalized Cox time-varying survival model on the prepared dataset.

    Args:
        modeling_df: Prepared modeling dataframe containing id, interval,
            event, and scaled feature columns.
        penalizer: Penalty strength passed to CoxTimeVaryingFitter.

    Returns:
        A fitted CoxTimeVaryingFitter model instance.
    """
    model = CoxTimeVaryingFitter(penalizer=penalizer)
    fit_df = modeling_df.drop(columns=["month"], errors="ignore").copy()

    model.fit(
        fit_df,
        id_col="business_id",
        event_col="event",
        start_col="start",
        stop_col="stop",
        show_progress=True,
    )
    return model


def fit_standard_cox_model(
    modeling_df: pd.DataFrame,
    penalizer: float,
) -> CoxPHFitter:
    """Fit a penalized standard Cox proportional hazards model on the prepared dataset.

    Args:
        modeling_df: Prepared modeling dataframe containing durations,
            event status, and scaled feature columns.
        penalizer: Penalty strength passed to CoxPHFitter.

    Returns:
        A fitted CoxPHFitter model instance.
    """
    model = CoxPHFitter(penalizer=penalizer)
    fit_df = modeling_df.drop(columns=["business_id"], errors="ignore").copy()

    model.fit(
        fit_df,
        duration_col="duration_months",
        event_col="event",
        show_progress=True,
    )
    return model


def build_coefficient_summary(model: CoxPHFitter | CoxTimeVaryingFitter) -> pd.DataFrame:
    """Create a coefficient summary dataframe sorted by absolute coefficient size.

    Args:
        model: A fitted CoxPHFitter or CoxTimeVaryingFitter model.

    Returns:
        A dataframe derived from the model summary, augmented with feature names
        and absolute coefficient values, sorted in descending order.
    """
    summary = model.summary.copy()
    summary["feature"] = summary.index
    summary["abs_coef"] = summary["coef"].abs()
    summary = summary.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return summary


def save_time_varying_artifacts(
    model: CoxTimeVaryingFitter,
    prepared_data: PreparedModelData,
    output_dir: Path,
) -> dict[str, Path]:
    """Save model, preprocessing artifacts, and summary outputs for the time-varying Cox pipeline.

    Args:
        model: Fitted time-varying Cox model.
        prepared_data: Prepared modeling data and preprocessing artifacts.
        output_dir: Directory where artifacts should be saved.

    Returns:
        A dictionary mapping artifact names to their saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_coefficient_summary(model)

    return {
        "model": save_pickle_artifact(model, output_dir / "cox_time_varying_model.pkl"),
        "scaler": save_pickle_artifact(
            prepared_data.scaler,
            output_dir / "cox_time_varying_scaler.pkl",
        ),
        "kept_columns": save_pickle_artifact(
            prepared_data.feature_selection.kept_columns,
            output_dir / "cox_time_varying_kept_columns.pkl",
        ),
        "dropped_columns": save_pickle_artifact(
            prepared_data.feature_selection.dropped_columns,
            output_dir / "cox_time_varying_dropped_columns.pkl",
        ),
        "summary": save_dataframe_artifact(
            summary,
            output_dir / "cox_time_varying_summary.csv",
        ),
    }


def save_standard_cox_artifacts(
    model: CoxPHFitter,
    prepared_data: PreparedModelData,
    output_dir: Path,
) -> dict[str, Path]:
    """Save model, preprocessing artifacts, and summary outputs for the standard CoxPH pipeline.

    Args:
        model: Fitted standard CoxPH model.
        prepared_data: Prepared modeling data and preprocessing artifacts.
        output_dir: Directory where artifacts should be saved.

    Returns:
        A dictionary mapping artifact names to their saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_coefficient_summary(model)

    return {
        "model": save_pickle_artifact(model, output_dir / "coxph_model.pkl"),
        "scaler": save_pickle_artifact(
            prepared_data.scaler,
            output_dir / "coxph_scaler.pkl",
        ),
        "kept_columns": save_pickle_artifact(
            prepared_data.feature_selection.kept_columns,
            output_dir / "coxph_kept_columns.pkl",
        ),
        "dropped_columns": save_pickle_artifact(
            prepared_data.feature_selection.dropped_columns,
            output_dir / "coxph_dropped_columns.pkl",
        ),
        "summary": save_dataframe_artifact(summary, output_dir / "coxph_summary.csv"),
    }


def run_time_varying_pipeline(config: CoxConfig) -> dict[str, object]:
    """Run the full time-varying Cox modeling workflow from input data to saved artifacts.

    Args:
        config: Configuration for data loading, preprocessing, fitting, and saving.

    Returns:
        A dictionary containing dataset shapes, event counts, selected features,
        and saved artifact paths.

    Raises:
        ValueError: If the joined dataset is invalid or no usable features remain.
    """
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined)

    panel = build_time_varying_panel(
        joined=joined,
        study_end=config.study_end,
    )
    prepared_data = prepare_time_varying_model_data(
        panel=panel,
        variance_threshold=config.variance_threshold,
    )
    model = fit_time_varying_cox_model(
        modeling_df=prepared_data.modeling_df,
        penalizer=config.penalizer,
    )
    artifact_paths = save_time_varying_artifacts(
        model=model,
        prepared_data=prepared_data,
        output_dir=config.output_dir,
    )

    return {
        "joined_shape": joined.shape,
        "panel_shape": panel.shape,
        "modeling_shape": prepared_data.modeling_df.shape,
        "n_events": int(panel["event"].sum()),
        "n_businesses": int(panel["business_id"].nunique()),
        "kept_columns": prepared_data.feature_selection.kept_columns,
        "dropped_columns": prepared_data.feature_selection.dropped_columns,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
    }


def run_standard_cox_pipeline(config: CoxConfig) -> dict[str, object]:
    """Run the full standard CoxPH modeling workflow from input data to saved artifacts.

    Args:
        config: Configuration for data loading, preprocessing, fitting, and saving.

    Returns:
        A dictionary containing dataset shapes, event counts, selected features,
        and saved artifact paths.

    Raises:
        ValueError: If the joined dataset is invalid or no usable features remain.
    """
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined)

    coxph_df = build_business_level_dataset(
        joined=joined,
        study_end=config.study_end,
    )
    prepared_data = prepare_business_level_model_data(
        coxph_df=coxph_df,
        variance_threshold=config.variance_threshold,
    )
    model = fit_standard_cox_model(
        modeling_df=prepared_data.modeling_df,
        penalizer=config.penalizer,
    )
    artifact_paths = save_standard_cox_artifacts(
        model=model,
        prepared_data=prepared_data,
        output_dir=config.output_dir,
    )

    return {
        "joined_shape": joined.shape,
        "coxph_df_shape": coxph_df.shape,
        "modeling_shape": prepared_data.modeling_df.shape,
        "n_events": int(coxph_df["event"].sum()),
        "n_businesses": int(coxph_df["business_id"].nunique()),
        "kept_columns": prepared_data.feature_selection.kept_columns,
        "dropped_columns": prepared_data.feature_selection.dropped_columns,
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
    }


def run_full_pipeline(config: CoxConfig) -> dict[str, dict[str, object]]:
    """Run both Cox modeling workflows and return their results.

    Args:
        config: Base configuration used to derive per-model output directories.

    Returns:
        A dictionary with separate result payloads for the time-varying and
        standard Cox pipelines.
    """
    time_varying_output_dir = config.output_dir / "time_varying"
    standard_output_dir = config.output_dir / "standard"

    time_varying_config = CoxConfig(
        data_path=config.data_path,
        output_dir=time_varying_output_dir,
        study_end=config.study_end,
        variance_threshold=config.variance_threshold,
        penalizer=config.penalizer,
    )
    standard_config = CoxConfig(
        data_path=config.data_path,
        output_dir=standard_output_dir,
        study_end=config.study_end,
        variance_threshold=config.variance_threshold,
        penalizer=config.penalizer,
    )

    return {
        "time_varying": run_time_varying_pipeline(time_varying_config),
        "standard": run_standard_cox_pipeline(standard_config),
    }


def parse_args() -> CoxConfig:
    """Parse command-line arguments and construct a CoxConfig instance.

    Returns:
        A CoxConfig populated from command-line argument values.
    """
    parser = argparse.ArgumentParser(
        description="Fit time-varying and standard Cox survival models."
    )
    add_standard_modeling_args(parser)
    parser.add_argument(
        "--penalizer",
        type=float,
        default=DEFAULT_PENALIZER,
        help="Penalizer for Cox models",
    )

    args = parser.parse_args()

    return CoxConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        study_end=pd.Timestamp(args.study_end),
        variance_threshold=args.variance_threshold,
        penalizer=args.penalizer,
    )


def main() -> None:
    """Entry point for script execution."""
    config = parse_args()
    results = run_full_pipeline(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
