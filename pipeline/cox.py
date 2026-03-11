"""Train Cox Proportional Hazards models for business survival analysis."""

from __future__ import annotations

import argparse
import mlflow
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler



from pipeline.utils import (
    STUDY_END, VARIANCE_THRESHOLD, FeatureSelectionResult,
    load_joined_dataset, validate_joined_dataset, get_model_drop_columns,
    get_model_req_columns, calculate_duration_months, extract_baseline_features,
    save_pickle_artifact, save_dataframe_artifact, add_standard_modeling_args
)

PENALIZER = 0.1


@dataclass(frozen=True)
class CoxConfig:
    """Configuration for Cox survival modeling pipeline."""
    data_path: Path
    output_dir: Path
    study_end: pd.Timestamp = STUDY_END
    variance_threshold: float = VARIANCE_THRESHOLD
    penalizer: float = PENALIZER


@dataclass(frozen=True)
class PreparedModelData:
    """Container for data components ready for model fitting."""
    modeling_df: pd.DataFrame
    scaler: StandardScaler
    feature_selection: FeatureSelectionResult


def build_time_varying_panel(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Convert panel format to time-varying start/stop format for lifelines."""
    df = joined.copy()
    df = df.loc[df["month"] <= study_end].copy()

    df = df.sort_values(["business_id", "month"]).reset_index(drop=True)
    df["first_month"] = df.groupby("business_id")["month"].transform("min")

    df["stop"] = (
        (df["month"].dt.year - df["first_month"].dt.year) * 12
        + (df["month"].dt.month - df["first_month"].dt.month)
        + 1
    )
    df["start"] = df["stop"] - 1

    df["next_month"] = df.groupby("business_id")["month"].shift(-1)
    df["is_last_observation"] = df["next_month"].isna()

    df["event"] = 0
    closed_condition = df["is_last_observation"] & (df["open"] == 0)
    df.loc[closed_condition, "event"] = 1

    return df


def build_business_level_dataset(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build cross-sectional dataset for standard Cox modeling."""
    df = joined.copy()
    df = df.loc[df["month"] <= study_end].copy()

    business_survival = calculate_duration_months(df)

    last_obs_open = df.sort_values(["business_id", "month"]).groupby("business_id")["open"].last()
    business_survival["event"] = (
        business_survival["business_id"].map(last_obs_open).eq(0).astype(int)
    )

    business_features = extract_baseline_features(df).drop(columns=["month", "open"])

    coxph_df = business_survival.merge(
        business_features,
        on="business_id",
        how="inner",
    )

    return coxph_df


def get_feature_columns(
    df: pd.DataFrame,
    special_columns: list[str],
) -> list[str]:
    """Extract model feature columns by dropping special formatting columns."""
    drop_columns = get_model_drop_columns()
    return [
        column
        for column in df.columns
        if column not in drop_columns and column not in special_columns
    ]


def select_nonconstant_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    variance_threshold: float,
) -> FeatureSelectionResult:
    """Identify features that pass the variance threshold."""
    if not feature_columns:
        raise ValueError("No feature columns provided for variance filtering.")

    x_df = df[feature_columns].copy()

    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(x_df)

    kept_columns = x_df.columns[selector.get_support()].tolist()
    dropped_columns = [column for column in x_df.columns if column not in kept_columns]

    return FeatureSelectionResult(
        kept_columns=kept_columns,
        dropped_columns=dropped_columns,
    )


def scale_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale feature columns to zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_columns])

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=feature_columns,
        index=df.index,
    )

    return scaled_df, scaler


def prepare_time_varying_model_data(
    panel: pd.DataFrame,
    variance_threshold: float,
) -> PreparedModelData:
    """Filter features, scale, and assemble data for time-varying Cox."""
    special_columns = ["start", "stop", "event", "next_month", "is_last_observation", "first_month"]
    feature_columns = get_feature_columns(panel, special_columns)

    feature_selection = select_nonconstant_features(
        df=panel,
        feature_columns=feature_columns,
        variance_threshold=variance_threshold,
    )

    scaled_df, scaler = scale_features(
        df=panel,
        feature_columns=feature_selection.kept_columns,
    )

    modeling_df = pd.concat(
        [
            panel[["business_id", "start", "stop", "event"]],
            scaled_df,
        ],
        axis=1,
    )

    return PreparedModelData(
        modeling_df=modeling_df,
        scaler=scaler,
        feature_selection=feature_selection,
    )


def prepare_business_level_model_data(
    coxph_df: pd.DataFrame,
    variance_threshold: float,
) -> PreparedModelData:
    """Filter features, scale, and assemble data for standard Cox."""
    special_columns = ["duration_months", "event", "first_month", "last_month"]
    feature_columns = get_feature_columns(coxph_df, special_columns)

    feature_selection = select_nonconstant_features(
        df=coxph_df,
        feature_columns=feature_columns,
        variance_threshold=variance_threshold,
    )

    scaled_df, scaler = scale_features(
        df=coxph_df,
        feature_columns=feature_selection.kept_columns,
    )

    modeling_df = pd.concat(
        [
            coxph_df[["business_id", "duration_months", "event"]],
            scaled_df,
        ],
        axis=1,
    )

    return PreparedModelData(
        modeling_df=modeling_df,
        scaler=scaler,
        feature_selection=feature_selection,
    )


def fit_time_varying_cox_model(
    modeling_df: pd.DataFrame,
    penalizer: float,
) -> CoxTimeVaryingFitter:
    """Fit a Cox time-varying model using lifelines."""
    model = CoxTimeVaryingFitter(penalizer=penalizer)
    model.fit(
        modeling_df,
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
    """Fit a standard Cox proportional hazards model using lifelines."""
    model = CoxPHFitter(penalizer=penalizer)
    fit_df = modeling_df.drop(columns=["business_id"])

    fit_df["duration_months"] = fit_df["duration_months"].clip(lower=0.1)

    model.fit(
        fit_df,
        duration_col="duration_months",
        event_col="event",
        show_progress=True,
    )
    return model


def build_coefficient_summary(model: CoxPHFitter | CoxTimeVaryingFitter) -> pd.DataFrame:
    """Extract coefficients and return a sorted summary dataframe."""
    summary_df = model.summary.copy()
    summary_df["feature"] = summary_df.index
    summary_df["abs_coef"] = summary_df["coef"].abs()
    summary_df = summary_df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return summary_df


def save_time_varying_artifacts(
    model: CoxTimeVaryingFitter,
    prepared_data: PreparedModelData,
    output_dir: Path,
) -> dict[str, Path]:
    """Save time-varying model artifacts."""
    return {
        "model": save_pickle_artifact(
            model, output_dir / "cox_time_varying_model.pkl"
        ),
        "scaler": save_pickle_artifact(
            prepared_data.scaler, output_dir / "cox_time_varying_scaler.pkl"
        ),
        "kept_columns": save_pickle_artifact(
            prepared_data.feature_selection.kept_columns,
            output_dir / "cox_time_varying_kept_columns.pkl"
        ),
        "dropped_columns": save_pickle_artifact(
            prepared_data.feature_selection.dropped_columns,
            output_dir / "cox_time_varying_dropped_columns.pkl"
        ),
        "summary_csv": save_dataframe_artifact(
            build_coefficient_summary(model), output_dir / "cox_time_varying_summary.csv"
        ),
    }


def save_standard_artifacts(
    model: CoxPHFitter,
    prepared_data: PreparedModelData,
    output_dir: Path,
) -> dict[str, Path]:
    """Save standard model artifacts."""
    return {
        "model": save_pickle_artifact(
            model, output_dir / "coxph_model.pkl"
        ),
        "scaler": save_pickle_artifact(
            prepared_data.scaler, output_dir / "coxph_scaler.pkl"
        ),
        "kept_columns": save_pickle_artifact(
            prepared_data.feature_selection.kept_columns, output_dir / "coxph_kept_columns.pkl"
        ),
        "dropped_columns": save_pickle_artifact(
            prepared_data.feature_selection.dropped_columns,
            output_dir / "coxph_dropped_columns.pkl"
        ),
        "summary_csv": save_dataframe_artifact(
            build_coefficient_summary(model), output_dir / "coxph_summary.csv"
        ),
    }


def run_time_varying_cox_pipeline(config: CoxConfig) -> dict[str, object]:
    """Run full pipeline for time-varying Cox modeling."""
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined, get_model_req_columns())

    panel = build_time_varying_panel(joined, config.study_end)

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
        "pipeline": "time_varying",
        "modeling_shape": prepared_data.modeling_df.shape,
        "n_kept_columns": len(prepared_data.feature_selection.kept_columns),
        "n_dropped_columns": len(prepared_data.feature_selection.dropped_columns),
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }


def run_standard_cox_pipeline(config: CoxConfig) -> dict[str, object]:
    """Run full pipeline for standard Cox modeling."""
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined, get_model_req_columns())

    coxph_df = build_business_level_dataset(joined, config.study_end)

    prepared_data = prepare_business_level_model_data(
        coxph_df=coxph_df,
        variance_threshold=config.variance_threshold,
    )

    model = fit_standard_cox_model(
        modeling_df=prepared_data.modeling_df,
        penalizer=config.penalizer,
    )

    artifact_paths = save_standard_artifacts(
        model=model,
        prepared_data=prepared_data,
        output_dir=config.output_dir,
    )

    return {
        "pipeline": "standard",
        "modeling_shape": prepared_data.modeling_df.shape,
        "n_kept_columns": len(prepared_data.feature_selection.kept_columns),
        "n_dropped_columns": len(prepared_data.feature_selection.dropped_columns),
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }


def run_full_pipeline(config: CoxConfig) -> dict[str, object]:
    """Execute both time-varying and standard Cox pipelines."""
    tv_config = CoxConfig(
        data_path=config.data_path,
        output_dir=config.output_dir / "time_varying",
        study_end=config.study_end,
        variance_threshold=config.variance_threshold,
        penalizer=config.penalizer,
    )
    tv_results = run_time_varying_cox_pipeline(tv_config)

    std_config = CoxConfig(
        data_path=config.data_path,
        output_dir=config.output_dir / "standard",
        study_end=config.study_end,
        variance_threshold=config.variance_threshold,
        penalizer=config.penalizer,
    )
    std_results = run_standard_cox_pipeline(std_config)

    return {
        "time_varying": tv_results,
        "standard": std_results,
    }


def parse_args() -> CoxConfig:
    """Parse CLI arguments into a CoxConfig."""
    parser = argparse.ArgumentParser(description="Train Cox survival models for business survival.")
    add_standard_modeling_args(parser)
    parser.add_argument("--penalizer", type=float, default=PENALIZER)

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
    mlflow.set_experiment("business_survival_cox_models")
    with mlflow.start_run():

        mlflow.log_params({
            "penalizer": config.penalizer,
            "variance_threshold": config.variance_threshold,
            "study_end": str(config.study_end),
        })

        summary = run_full_pipeline(config)

        # Log all artifacts produced by the pipeline
        mlflow.log_artifacts(str(config.output_dir))

        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
