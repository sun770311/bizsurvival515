"""Train a logistic regression model for 3-year business survival.

Input:
- joined_dataset.csv

Processing steps:
- Load and validate the joined business-month panel
- Build a business-level dataset using aggregated first-year features
- Remove leakage and near-constant predictors
- Balance the training data and fit logistic regression
- Save model artifacts and evaluation outputs

Outputs:
- Saved logistic regression pipeline
- Saved retained and dropped feature columns
- Saved coefficient summary CSV
- Saved balanced dataset and train/test splits
- Saved evaluation metrics JSON
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample


STUDY_END = pd.Timestamp("2026-03-01")
SURVIVAL_MONTHS = 36
AGGREGATION_MONTHS = 12
VARIANCE_THRESHOLD = 1e-8
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 5000


@dataclass(frozen=True)
class LogisticConfig:
    """Configuration for logistic survival modeling pipeline."""

    data_path: Path
    output_dir: Path
    study_end: pd.Timestamp = STUDY_END
    survival_months: int = SURVIVAL_MONTHS
    aggregation_months: int = AGGREGATION_MONTHS
    variance_threshold: float = VARIANCE_THRESHOLD
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    max_iter: int = MAX_ITER


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for retained and dropped feature names."""

    kept_columns: list[str]
    dropped_columns: list[str]


@dataclass(frozen=True)
class PreparedTrainingData:
    """Container for prepared datasets used in modeling."""

    training_df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    balanced_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_selection: FeatureSelectionResult


def load_joined_dataset(data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse month column."""
    joined = pd.read_csv(data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate required columns and uniqueness of panel rows."""
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


def restrict_to_study_window(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict panel rows to the study window."""
    df = joined.copy()
    df = df.loc[df["month"] <= study_end].copy()
    return df


def build_business_survival_summary(
    df: pd.DataFrame,
    survival_months: int,
) -> pd.DataFrame:
    """Build business-level survival summary and survival target."""
    first_month = df.groupby("business_id")["month"].min()
    last_month = df.groupby("business_id")["month"].max()

    duration_months = (
        (last_month.dt.year - first_month.dt.year) * 12
        + (last_month.dt.month - first_month.dt.month)
    )

    business_survival = pd.DataFrame(
        {
            "business_id": first_month.index,
            "first_month": first_month,
            "last_month": last_month,
            "duration_months": duration_months,
        }
    ).reset_index(drop=True)

    business_survival["survived_36m"] = (
        business_survival["duration_months"] >= survival_months
    ).astype(int)

    return business_survival


def filter_eligible_businesses(
    business_survival: pd.DataFrame,
    study_end: pd.Timestamp,
    survival_months: int,
    aggregation_months: int,
) -> pd.DataFrame:
    """Keep businesses with a full first-year window and enough horizon to score survival.

    This creates a first-year-profile modeling cohort:
    - business must have entered early enough to evaluate 36-month survival
    - business must have at least `aggregation_months` observed months available
      so first-year aggregation is well-defined
    """
    survival_cutoff = study_end - pd.DateOffset(months=survival_months)
    min_last_month_needed = (
        business_survival["first_month"] + pd.DateOffset(months=aggregation_months - 1)
    )

    eligible = business_survival.loc[
        (business_survival["first_month"] <= survival_cutoff)
        & (business_survival["last_month"] >= min_last_month_needed)
    ].copy()
    return eligible


def get_first_year_window(
    df: pd.DataFrame,
    aggregation_months: int,
) -> pd.DataFrame:
    """Return the first N observed monthly rows per business."""
    panel = df.sort_values(["business_id", "month"]).copy()
    panel["row_num_within_business"] = panel.groupby("business_id").cumcount()
    first_year = panel.loc[
        panel["row_num_within_business"] < aggregation_months
    ].copy()
    return first_year


def is_binary_series(series: pd.Series) -> bool:
    """Return True when non-null values are all binary 0/1."""
    non_null = series.dropna()
    if non_null.empty:
        return False
    unique_values = set(non_null.unique().tolist())
    return unique_values.issubset({0, 1})


def choose_aggregation(column_name: str, series: pd.Series) -> str:
    """Choose aggregation rule for a feature column."""
    static_first_columns = {
        "business_latitude",
        "business_longitude",
        "location_cluster_lat",
        "location_cluster_lng",
        "location_cluster",
    }

    sum_columns = {
        "complaint_sum",
        "total_311",
    }

    if column_name in {"business_id", "month", "row_num_within_business"}:
        raise ValueError(f"Aggregation should not be chosen for identifier column {column_name}.")

    if column_name in static_first_columns:
        return "first"

    if column_name.startswith("business_category_"):
        return "max"

    if column_name.startswith("complaint_type_"):
        return "sum"

    if column_name in sum_columns:
        return "sum"

    if column_name == "active_license_count":
        return "mean"

    if column_name == "open":
        return "max"

    if column_name == "months_since_first_license":
        return "max"

    if is_binary_series(series):
        return "max"

    return "last"


def aggregate_first_year_features(
    df: pd.DataFrame,
    aggregation_months: int,
) -> pd.DataFrame:
    """Aggregate the first N observed months into one row per business."""
    first_year = get_first_year_window(df, aggregation_months=aggregation_months)

    feature_columns = [
        column
        for column in first_year.columns
        if column not in {"business_id", "month", "row_num_within_business"}
    ]

    aggregation_map: dict[str, str] = {}
    for column in feature_columns:
        aggregation_map[column] = choose_aggregation(column, first_year[column])

    aggregated = (
        first_year.groupby("business_id", as_index=False)
        .agg(aggregation_map)
        .copy()
    )

    rename_map = {}
    for column, agg_name in aggregation_map.items():
        if agg_name == "sum":
            rename_map[column] = f"{column}_first12m_sum"
        elif agg_name == "mean":
            rename_map[column] = f"{column}_first12m_mean"
        elif agg_name == "max":
            rename_map[column] = f"{column}_first12m_max"
        elif agg_name == "last":
            rename_map[column] = f"{column}_first12m_last"
        elif agg_name == "first":
            rename_map[column] = f"{column}_first12m_first"

    aggregated = aggregated.rename(columns=rename_map)

    first_year_counts = (
        first_year.groupby("business_id")
        .size()
        .rename("observed_months_in_first_window")
        .reset_index()
    )

    aggregated = aggregated.merge(first_year_counts, on="business_id", how="left")
    return aggregated


def build_training_dataset(
    df: pd.DataFrame,
    study_end: pd.Timestamp,
    survival_months: int,
    aggregation_months: int,
) -> pd.DataFrame:
    """Build business-level training dataset with first-year features and target."""
    business_survival = build_business_survival_summary(df, survival_months)
    eligible = filter_eligible_businesses(
        business_survival=business_survival,
        study_end=study_end,
        survival_months=survival_months,
        aggregation_months=aggregation_months,
    )
    eligible_ids = set(eligible["business_id"])

    eligible_panel = df.loc[df["business_id"].isin(eligible_ids)].copy()
    aggregated_features = aggregate_first_year_features(
        eligible_panel,
        aggregation_months=aggregation_months,
    )

    training_df = aggregated_features.merge(
        eligible[
            [
                "business_id",
                "first_month",
                "last_month",
                "duration_months",
                "survived_36m",
            ]
        ],
        on="business_id",
        how="inner",
    )
    return training_df


def get_excluded_feature_columns() -> list[str]:
    """Return columns excluded from logistic feature matrix."""
    return [
        "business_id",
        "first_month",
        "last_month",
        "duration_months",
        "survived_36m",
        # leakage / target-adjacent columns intentionally excluded from fitting
        "months_since_first_license_first12m_max",
        "business_category_sum_first12m_max",
        "complaint_sum_first12m_sum",
        "total_311_first12m_sum",
        "open_first12m_max",
        "location_cluster_first12m_first",
    ]


def split_features_and_target(
    training_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split business-level dataset into predictors and target."""
    excluded_columns = get_excluded_feature_columns()
    feature_columns = [
        column for column in training_df.columns if column not in excluded_columns
    ]
    X = training_df[feature_columns].copy()
    y = training_df["survived_36m"].copy()
    return X, y


def select_nonconstant_features(
    X: pd.DataFrame,
    variance_threshold: float,
) -> tuple[pd.DataFrame, FeatureSelectionResult]:
    """Remove near-constant features using variance thresholding."""
    if X.empty:
        raise ValueError("Feature matrix is empty before variance filtering.")

    selector = VarianceThreshold(threshold=variance_threshold)
    X_reduced = selector.fit_transform(X)

    kept_columns = X.columns[selector.get_support()].tolist()
    dropped_columns = [column for column in X.columns if column not in kept_columns]

    if not kept_columns:
        raise ValueError("No features remain after low-variance filtering.")

    reduced_df = pd.DataFrame(
        X_reduced,
        columns=kept_columns,
        index=X.index,
    )

    return reduced_df, FeatureSelectionResult(
        kept_columns=kept_columns,
        dropped_columns=dropped_columns,
    )


def balance_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Oversample minority class to match majority class size."""
    full_df = X.copy()
    full_df["survived_36m"] = y.values

    class_counts = full_df["survived_36m"].value_counts()
    if class_counts.shape[0] < 2:
        raise ValueError("Target must contain both classes for balancing and training.")

    majority_label = class_counts.idxmax()
    minority_label = class_counts.idxmin()

    majority_df = full_df.loc[full_df["survived_36m"] == majority_label].copy()
    minority_df = full_df.loc[full_df["survived_36m"] == minority_label].copy()

    minority_upsampled = resample(
        minority_df,
        replace=True,
        n_samples=len(majority_df),
        random_state=random_state,
    )

    balanced_df = pd.concat([majority_df, minority_upsampled], axis=0)
    balanced_df = balanced_df.sample(
        frac=1.0,
        random_state=random_state,
    ).reset_index(drop=True)

    return balanced_df


def train_test_split_balanced(
    balanced_df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split balanced dataset into train and test partitions."""
    X_balanced = balanced_df.drop(columns=["survived_36m"])
    y_balanced = balanced_df["survived_36m"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=test_size,
        random_state=random_state,
        stratify=y_balanced,
    )
    return X_train, X_test, y_train, y_test


def prepare_training_data(config: LogisticConfig) -> PreparedTrainingData:
    """Prepare business-level training data for logistic regression."""
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined)
    df = restrict_to_study_window(joined, config.study_end)

    training_df = build_training_dataset(
        df=df,
        study_end=config.study_end,
        survival_months=config.survival_months,
        aggregation_months=config.aggregation_months,
    )
    X_raw, y = split_features_and_target(training_df)
    X, feature_selection = select_nonconstant_features(
        X=X_raw,
        variance_threshold=config.variance_threshold,
    )
    balanced_df = balance_dataset(
        X=X,
        y=y,
        random_state=config.random_state,
    )
    X_train, X_test, y_train, y_test = train_test_split_balanced(
        balanced_df=balanced_df,
        test_size=config.test_size,
        random_state=config.random_state,
    )

    return PreparedTrainingData(
        training_df=training_df,
        X=X,
        y=y,
        balanced_df=balanced_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_selection=feature_selection,
    )


def build_logistic_pipeline(max_iter: int, random_state: int) -> Pipeline:
    """Build sklearn pipeline for logistic regression modeling."""
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline


def fit_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Fit logistic regression pipeline."""
    pipeline = build_logistic_pipeline(
        max_iter=max_iter,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate logistic regression model on the test split."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    class_report = classification_report(y_test, y_pred, output_dict=True)

    survivor_probs = y_prob[y_test == 1]
    nonsurvivor_probs = y_prob[y_test == 0]

    mannwhitney_stat, mannwhitney_p = mannwhitneyu(
        survivor_probs,
        nonsurvivor_probs,
        alternative="greater",
    )

    return {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "mean_predicted_probability_survivors": float(np.mean(survivor_probs)),
        "mean_predicted_probability_nonsurvivors": float(np.mean(nonsurvivor_probs)),
        "mannwhitney_u_statistic": float(mannwhitney_stat),
        "mannwhitney_p_value": float(mannwhitney_p),
    }


def build_coefficient_summary(
    pipeline: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Build sorted coefficient summary from fitted logistic model."""
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": coefficients,
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def save_pickle_artifact(obj: object, output_path: Path) -> Path:
    """Save object as pickle artifact."""
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return output_path


def save_dataframe_artifact(df: pd.DataFrame, output_path: Path) -> Path:
    """Save dataframe artifact as CSV."""
    df.to_csv(output_path, index=False)
    return output_path


def save_json_artifact(payload: dict[str, object], output_path: Path) -> Path:
    """Save dictionary artifact as JSON."""
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
    return output_path


def save_model_artifacts(
    prepared_data: PreparedTrainingData,
    pipeline: Pipeline,
    metrics: dict[str, object],
    output_dir: Path,
) -> dict[str, Path]:
    """Save fitted model artifacts and evaluation outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficient_summary = build_coefficient_summary(
        pipeline=pipeline,
        feature_columns=prepared_data.X_train.columns.tolist(),
    )

    X_train_out = prepared_data.X_train.copy()
    X_train_out["survived_36m"] = prepared_data.y_train.values

    X_test_out = prepared_data.X_test.copy()
    X_test_out["survived_36m"] = prepared_data.y_test.values

    artifact_paths = {
        "model_pipeline": save_pickle_artifact(
            pipeline,
            output_dir / "logistic_pipeline.pkl",
        ),
        "kept_columns": save_pickle_artifact(
            prepared_data.feature_selection.kept_columns,
            output_dir / "logistic_kept_columns.pkl",
        ),
        "dropped_columns": save_pickle_artifact(
            prepared_data.feature_selection.dropped_columns,
            output_dir / "logistic_dropped_columns.pkl",
        ),
        "coefficient_summary": save_dataframe_artifact(
            coefficient_summary,
            output_dir / "logistic_coefficient_summary.csv",
        ),
        "balanced_dataset": save_dataframe_artifact(
            prepared_data.balanced_df,
            output_dir / "business_survival_balanced_dataset.csv",
        ),
        "train_split": save_dataframe_artifact(
            X_train_out,
            output_dir / "X_train_balanced_split.csv",
        ),
        "test_split": save_dataframe_artifact(
            X_test_out,
            output_dir / "X_test_balanced_split.csv",
        ),
        "evaluation_metrics": save_json_artifact(
            metrics,
            output_dir / "logistic_evaluation_metrics.json",
        ),
    }
    return artifact_paths


def run_logistic_pipeline(config: LogisticConfig) -> dict[str, object]:
    """Run full logistic regression training and artifact-saving pipeline."""
    prepared_data = prepare_training_data(config)
    pipeline = fit_logistic_model(
        X_train=prepared_data.X_train,
        y_train=prepared_data.y_train,
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    metrics = evaluate_model(
        pipeline=pipeline,
        X_test=prepared_data.X_test,
        y_test=prepared_data.y_test,
    )
    artifact_paths = save_model_artifacts(
        prepared_data=prepared_data,
        pipeline=pipeline,
        metrics=metrics,
        output_dir=config.output_dir,
    )

    return {
        "training_shape": prepared_data.training_df.shape,
        "feature_matrix_shape": prepared_data.X.shape,
        "balanced_shape": prepared_data.balanced_df.shape,
        "train_shape": prepared_data.X_train.shape,
        "test_shape": prepared_data.X_test.shape,
        "aggregation_months": config.aggregation_months,
        "n_kept_columns": len(prepared_data.feature_selection.kept_columns),
        "n_dropped_columns": len(prepared_data.feature_selection.dropped_columns),
        "target_distribution_original": prepared_data.y.value_counts().to_dict(),
        "target_distribution_balanced": (
            prepared_data.balanced_df["survived_36m"].value_counts().to_dict()
        ),
        "metrics": metrics,
        "artifact_paths": {key: str(path) for key, path in artifact_paths.items()},
    }


def parse_args() -> LogisticConfig:
    """Parse CLI arguments into a LogisticConfig."""
    parser = argparse.ArgumentParser(
        description="Train logistic regression model for 3-year business survival."
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
        help="Directory to save logistic model artifacts",
    )
    parser.add_argument(
        "--study-end",
        type=str,
        default=str(STUDY_END.date()),
        help="Study end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--survival-months",
        type=int,
        default=SURVIVAL_MONTHS,
        help="Survival horizon in months",
    )
    parser.add_argument(
        "--aggregation-months",
        type=int,
        default=AGGREGATION_MONTHS,
        help="Number of first observed months to aggregate per business",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help="Variance threshold for feature filtering",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help="Test split fraction",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
        help="Maximum logistic regression iterations",
    )

    args = parser.parse_args()

    return LogisticConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        study_end=pd.Timestamp(args.study_end),
        survival_months=args.survival_months,
        aggregation_months=args.aggregation_months,
        variance_threshold=args.variance_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )


def main() -> None:
    """Entry point for script execution."""
    config = parse_args()
    summary = run_logistic_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()