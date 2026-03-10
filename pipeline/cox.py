"""Fit Cox survival models using the preprocessed joined_dataset.csv panel.

Input:
- joined_dataset.csv

Processing steps:
- Load and validate the joined monthly business panel
- Build time-varying and business-level survival datasets
- Remove redundant and near-constant predictors
- Standardize retained predictors and fit Cox models
- Save model artifacts for downstream prediction

Outputs:
- Penalized Cox time-varying model artifacts
- Penalized standard CoxPH model artifacts
- Coefficient summary CSV files
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler


STUDY_END = pd.Timestamp("2026-03-01")
VARIANCE_THRESHOLD = 1e-8
DEFAULT_PENALIZER = 0.1


@dataclass(frozen=True)
class CoxConfig:
    """Configuration for Cox modeling pipeline."""

    data_path: Path
    output_dir: Path
    study_end: pd.Timestamp = STUDY_END
    variance_threshold: float = VARIANCE_THRESHOLD
    penalizer: float = DEFAULT_PENALIZER


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for retained and dropped feature names."""

    kept_columns: list[str]
    dropped_columns: list[str]


@dataclass(frozen=True)
class PreparedModelData:
    """Container for scaled modeling data and preprocessing artifacts."""

    modeling_df: pd.DataFrame
    scaler: StandardScaler
    feature_selection: FeatureSelectionResult


def load_joined_dataset(data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse date columns."""
    joined = pd.read_csv(data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate that required columns are present and well-formed."""
    required_columns = [
        "business_id",
        "month",
        "active_license_count",
        "total_311",
        "open",
        "months_since_first_license",
        "location_cluster",
        "location_cluster_lat",
        "location_cluster_lng",
        "business_latitude",
        "business_longitude",
        "business_category_sum",
        "complaint_sum",
    ]
    missing_columns = [column for column in required_columns if column not in joined.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if joined["month"].isna().any():
        raise ValueError("Some month values could not be parsed as datetimes.")

    if joined.duplicated(["business_id", "month"]).any():
        raise ValueError("Duplicate business_id-month rows found in joined dataset.")


def get_model_drop_columns() -> list[str]:
    """Return columns excluded from model fitting."""
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
    """Build time-varying business-month survival panel."""
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
    """Build one-row-per-business dataset for standard CoxPH modeling."""
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
    """Return candidate feature columns excluding identifier and outcome columns."""
    return [column for column in df.columns if column not in exclude_columns]


def select_nonconstant_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    variance_threshold: float,
) -> FeatureSelectionResult:
    """Remove near-constant columns using variance thresholding."""
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
    """Standardize retained predictors."""
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
    """Prepare scaled dataset for Cox time-varying model fitting."""
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
    """Prepare scaled dataset for standard CoxPH model fitting."""
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
    """Fit penalized Cox time-varying model."""
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
    """Fit penalized standard CoxPH model."""
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
    """Create sorted coefficient summary for inspection and export."""
    summary = model.summary.copy()
    summary["feature"] = summary.index
    summary["abs_coef"] = summary["coef"].abs()
    summary = summary.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    return summary


def save_pickle_artifact(obj: object, output_path: Path) -> Path:
    """Save object as pickle artifact."""
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return output_path


def save_dataframe_artifact(df: pd.DataFrame, output_path: Path) -> Path:
    """Save dataframe artifact as CSV."""
    df.to_csv(output_path, index=False)
    return output_path


def save_time_varying_artifacts(
    model: CoxTimeVaryingFitter,
    prepared_data: PreparedModelData,
    output_dir: Path,
) -> dict[str, Path]:
    """Save time-varying model artifacts."""
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
    """Save standard CoxPH model artifacts."""
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
    """Run full Cox time-varying modeling pipeline."""
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
    """Run full standard CoxPH modeling pipeline."""
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
    """Run both time-varying and standard Cox pipelines."""
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
    """Parse CLI arguments into a CoxConfig."""
    parser = argparse.ArgumentParser(
        description="Fit time-varying and standard Cox survival models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to joined_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save Cox artifacts",
    )
    parser.add_argument(
        "--study-end",
        type=str,
        default=str(STUDY_END.date()),
        help="Study end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help="Variance threshold for feature filtering",
    )
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
