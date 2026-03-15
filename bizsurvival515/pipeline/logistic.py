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

import json
import argparse
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

from bizsurvival515.pipeline.utils import (
    STUDY_END,
    VARIANCE_THRESHOLD,
    FeatureSelectionResult,
    add_standard_modeling_args,
    load_joined_dataset,
    restrict_to_study_window,
    save_dataframe_artifact,
    save_json_artifact,
    save_pickle_artifact,
    validate_joined_dataset,
)


SURVIVAL_MONTHS = 36
AGGREGATION_MONTHS = 12
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 5000


@dataclass(frozen=True)
class LogisticModelSettings:
    """Tunable settings for logistic survival modeling."""

    study_end: pd.Timestamp = STUDY_END
    survival_months: int = SURVIVAL_MONTHS
    aggregation_months: int = AGGREGATION_MONTHS
    variance_threshold: float = VARIANCE_THRESHOLD
    test_size: float = TEST_SIZE
    random_state: int = RANDOM_STATE
    max_iter: int = MAX_ITER


@dataclass(frozen=True)
class LogisticConfig:
    """Configuration for logistic survival modeling pipeline."""

    data_path: Path
    output_dir: Path
    settings: LogisticModelSettings = LogisticModelSettings()


@dataclass(frozen=True)
class DatasetSplit:
    """Container for train/test feature and target splits."""

    features_train: pd.DataFrame
    features_test: pd.DataFrame
    target_train: pd.Series
    target_test: pd.Series


@dataclass(frozen=True)
class PreparedTrainingData:
    """Container for prepared datasets used in modeling."""

    training_df: pd.DataFrame
    feature_df: pd.DataFrame
    target: pd.Series
    balanced_df: pd.DataFrame
    split: DatasetSplit
    feature_selection: FeatureSelectionResult

    def __getattr__(self, name: str) -> object:
        """Provide backward-compatible attribute aliases."""
        legacy_mapping = {
            "X": self.feature_df,
            "y": self.target,
            "X_train": self.split.features_train,
            "X_test": self.split.features_test,
            "y_train": self.split.target_train,
            "y_test": self.split.target_test,
        }
        if name in legacy_mapping:
            return legacy_mapping[name]
        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute {name!r}"
        )


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
    """Keep businesses with a full first-year window and enough horizon.

    This creates a first-year-profile modeling cohort:
    - business must have entered early enough to evaluate 36-month survival
    - business must have at least `aggregation_months` observed months available
      so first-year aggregation is well-defined
    """
    survival_cutoff = study_end - pd.DateOffset(months=survival_months)
    min_last_month_needed = (
        business_survival["first_month"]
        + pd.DateOffset(months=aggregation_months - 1)
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
    invalid_columns = {"business_id", "month", "row_num_within_business"}

    if column_name in invalid_columns:
        raise ValueError(
            f"Aggregation should not be chosen for identifier column {column_name}."
        )

    aggregation = "last"

    if column_name in static_first_columns:
        aggregation = "first"
    elif column_name.startswith("business_category_"):
        aggregation = "max"
    elif column_name.startswith("complaint_type_"):
        aggregation = "sum"
    elif column_name in sum_columns:
        aggregation = "sum"
    elif column_name == "active_license_count":
        aggregation = "mean"
    elif column_name == "open":
        aggregation = "max"
    elif column_name == "months_since_first_license":
        aggregation = "max"
    elif is_binary_series(series):
        aggregation = "max"

    return aggregation


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

    aggregated = first_year.groupby("business_id", as_index=False).agg(aggregation_map)
    aggregated = aggregated.copy()

    rename_suffix_map = {
        "sum": "sum",
        "mean": "mean",
        "max": "max",
        "last": "last",
        "first": "first",
    }
    rename_map = {
        column: f"{column}_first12m_{rename_suffix_map[agg_name]}"
        for column, agg_name in aggregation_map.items()
    }

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
    feature_df = training_df[feature_columns].copy()
    target = training_df["survived_36m"].copy()
    return feature_df, target


def select_nonconstant_features(
    feature_df: pd.DataFrame,
    variance_threshold: float,
) -> tuple[pd.DataFrame, FeatureSelectionResult]:
    """Remove near-constant features using variance thresholding."""
    if feature_df.empty:
        raise ValueError("Feature matrix is empty before variance filtering.")

    selector = VarianceThreshold(threshold=variance_threshold)
    reduced_array = selector.fit_transform(feature_df)

    kept_columns = feature_df.columns[selector.get_support()].tolist()
    dropped_columns = [
        column for column in feature_df.columns if column not in kept_columns
    ]

    if not kept_columns:
        raise ValueError("No features remain after low-variance filtering.")

    reduced_df = pd.DataFrame(
        reduced_array,
        columns=kept_columns,
        index=feature_df.index,
    )

    return reduced_df, FeatureSelectionResult(
        kept_columns=kept_columns,
        dropped_columns=dropped_columns,
    )


def balance_dataset(
    feature_df: pd.DataFrame,
    target: pd.Series,
    random_state: int,
) -> pd.DataFrame:
    """Oversample minority class to match majority class size."""
    full_df = feature_df.copy()
    full_df["survived_36m"] = target.values

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
    balanced_features = balanced_df.drop(columns=["survived_36m"])
    balanced_target = balanced_df["survived_36m"]

    features_train, features_test, target_train, target_test = train_test_split(
        balanced_features,
        balanced_target,
        test_size=test_size,
        random_state=random_state,
        stratify=balanced_target,
    )
    return features_train, features_test, target_train, target_test


def prepare_training_data(config: LogisticConfig) -> PreparedTrainingData:
    """Prepare business-level training data for logistic regression."""
    joined = load_joined_dataset(config.data_path)
    validate_joined_dataset(joined)
    df = restrict_to_study_window(joined, config.settings.study_end)

    training_df = build_training_dataset(
        df=df,
        study_end=config.settings.study_end,
        survival_months=config.settings.survival_months,
        aggregation_months=config.settings.aggregation_months,
    )
    raw_feature_df, target = split_features_and_target(training_df)
    feature_df, feature_selection = select_nonconstant_features(
        feature_df=raw_feature_df,
        variance_threshold=config.settings.variance_threshold,
    )
    balanced_df = balance_dataset(
        feature_df=feature_df,
        target=target,
        random_state=config.settings.random_state,
    )
    (
        features_train,
        features_test,
        target_train,
        target_test,
    ) = train_test_split_balanced(
        balanced_df=balanced_df,
        test_size=config.settings.test_size,
        random_state=config.settings.random_state,
    )

    split = DatasetSplit(
        features_train=features_train,
        features_test=features_test,
        target_train=target_train,
        target_test=target_test,
    )

    return PreparedTrainingData(
        training_df=training_df,
        feature_df=feature_df,
        target=target,
        balanced_df=balanced_df,
        split=split,
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
    features_train: pd.DataFrame | None = None,
    target_train: pd.Series | None = None,
    max_iter: int = MAX_ITER,
    random_state: int = RANDOM_STATE,
    **legacy_kwargs: object,
) -> Pipeline:
    """Fit logistic regression pipeline."""
    if features_train is None:
        features_train = legacy_kwargs.pop("X_train", None)
    if target_train is None:
        target_train = legacy_kwargs.pop("y_train", None)

    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if features_train is None or target_train is None:
        raise ValueError("Training features and target must both be provided.")

    pipeline = build_logistic_pipeline(
        max_iter=max_iter,
        random_state=random_state,
    )
    pipeline.fit(features_train, target_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    features_test: pd.DataFrame | None = None,
    target_test: pd.Series | None = None,
    **legacy_kwargs: object,
) -> dict[str, object]:
    """Evaluate logistic regression model on the test split."""
    if features_test is None:
        features_test = legacy_kwargs.pop("X_test", None)
    if target_test is None:
        target_test = legacy_kwargs.pop("y_test", None)

    if legacy_kwargs:
        unexpected = ", ".join(sorted(legacy_kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    if features_test is None or target_test is None:
        raise ValueError("Test features and target must both be provided.")

    predicted_labels = pipeline.predict(features_test)
    predicted_probabilities = pipeline.predict_proba(features_test)[:, 1]

    accuracy = accuracy_score(target_test, predicted_labels)
    roc_auc = roc_auc_score(target_test, predicted_probabilities)
    conf_matrix = confusion_matrix(target_test, predicted_labels).tolist()
    class_report = classification_report(
        target_test,
        predicted_labels,
        output_dict=True,
    )

    survivor_probs = predicted_probabilities[target_test == 1]
    nonsurvivor_probs = predicted_probabilities[target_test == 0]

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
        "mean_predicted_probability_nonsurvivors": float(
            np.mean(nonsurvivor_probs)
        ),
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
    coef_df = coef_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)
    return coef_df


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
        feature_columns=prepared_data.split.features_train.columns.tolist(),
    )

    features_train_out = prepared_data.split.features_train.copy()
    features_train_out["survived_36m"] = prepared_data.split.target_train.values

    features_test_out = prepared_data.split.features_test.copy()
    features_test_out["survived_36m"] = prepared_data.split.target_test.values

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
            features_train_out,
            output_dir / "X_train_balanced_split.csv",
        ),
        "test_split": save_dataframe_artifact(
            features_test_out,
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
        features_train=prepared_data.split.features_train,
        target_train=prepared_data.split.target_train,
        max_iter=config.settings.max_iter,
        random_state=config.settings.random_state,
    )
    metrics = evaluate_model(
        pipeline=pipeline,
        features_test=prepared_data.split.features_test,
        target_test=prepared_data.split.target_test,
    )
    artifact_paths = save_model_artifacts(
        prepared_data=prepared_data,
        pipeline=pipeline,
        metrics=metrics,
        output_dir=config.output_dir,
    )

    return {
        "training_shape": prepared_data.training_df.shape,
        "feature_matrix_shape": prepared_data.feature_df.shape,
        "balanced_shape": prepared_data.balanced_df.shape,
        "train_shape": prepared_data.split.features_train.shape,
        "test_shape": prepared_data.split.features_test.shape,
        "aggregation_months": config.settings.aggregation_months,
        "n_kept_columns": len(prepared_data.feature_selection.kept_columns),
        "n_dropped_columns": len(prepared_data.feature_selection.dropped_columns),
        "target_distribution_original": prepared_data.target.value_counts().to_dict(),
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
    add_standard_modeling_args(parser)
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
        settings=LogisticModelSettings(
            study_end=pd.Timestamp(args.study_end),
            survival_months=args.survival_months,
            aggregation_months=args.aggregation_months,
            variance_threshold=args.variance_threshold,
            test_size=args.test_size,
            random_state=args.random_state,
            max_iter=args.max_iter,
        ),
    )


def main() -> None:
    """Entry point for script execution."""
    config = parse_args()
    summary = run_logistic_pipeline(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
