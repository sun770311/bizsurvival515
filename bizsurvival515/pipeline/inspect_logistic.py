"""Inspect trained logistic regression survival model."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class InspectConfig:
    """Configuration for logistic model inspection."""

    artifacts_dir: Path


def load_artifacts(
    config: InspectConfig,
) -> tuple[Any, list[str], dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load saved model artifacts."""
    base = config.artifacts_dir

    with (base / "logistic_pipeline.pkl").open("rb") as file_obj:
        pipeline = pickle.load(file_obj)

    with (base / "logistic_kept_columns.pkl").open("rb") as file_obj:
        kept_columns = pickle.load(file_obj)

    with (base / "logistic_evaluation_metrics.json").open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    coef_summary = pd.read_csv(base / "logistic_coefficient_summary.csv")
    train_split = pd.read_csv(base / "X_train_balanced_split.csv")

    return pipeline, kept_columns, metrics, coef_summary, train_split


def print_model_metrics(metrics: dict[str, Any]) -> None:
    """Print evaluation metrics."""
    print("\nMODEL METRICS\n")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def build_baseline_profile(
    kept_columns: list[str],
    train_split: pd.DataFrame,
) -> pd.DataFrame:
    """Create realistic baseline feature vector from training medians."""
    available_columns = [column for column in kept_columns if column in train_split.columns]
    missing_columns = [column for column in kept_columns if column not in train_split.columns]

    if missing_columns:
        raise ValueError(
            "Training split is missing kept feature columns: "
            f"{missing_columns}"
        )

    baseline = train_split[available_columns].copy()

    if "survived_36m" in baseline.columns:
        baseline = baseline.drop(columns=["survived_36m"])

    baseline_row = baseline.median(numeric_only=True).to_frame().T
    baseline_row = baseline_row.reindex(columns=kept_columns, fill_value=0.0)
    baseline_row.index = ["baseline"]
    return baseline_row


def build_hypothetical_profiles(
    kept_columns: list[str],
    train_split: pd.DataFrame,
) -> pd.DataFrame:
    """Construct hypothetical business profiles from a realistic baseline."""
    baseline = build_baseline_profile(kept_columns, train_split)

    profiles: list[pd.DataFrame] = [baseline.copy()]

    scenarios = [
        ("electronics_store", "business_category_electronics_store_first12m_max", 1.0),
        ("vape_shop", "business_category_electronic_cigarette_dealer_first12m_max", 1.0),
        ("bingo_operator", "business_category_bingo_game_operator_first12m_max", 1.0),
        ("many_licenses", "active_license_count_first12m_mean", 5.0),
        ("industrial_laundry", "business_category_industrial_laundry_first12m_max", 1.0),
        ("debt_collection", "business_category_debt_collection_agency_first12m_max", 1.0),
        (
            "home_improvement_contractor", 
            "business_category_home_improvement_contractor_first12m_max", 
            1.0),
    ]

    for profile_name, feature_name, feature_value in scenarios:
        profile = baseline.copy()
        profile.index = [profile_name]

        if feature_name in profile.columns:
            profile.loc[profile_name, feature_name] = feature_value

        profiles.append(profile)

    return pd.concat(profiles)


def predict_profiles(pipeline: Any, profiles: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for hypothetical profiles."""
    probabilities = pipeline.predict_proba(profiles)[:, 1]
    predictions = pipeline.predict(profiles)

    return pd.DataFrame(
        {
            "profile": profiles.index,
            "predicted_survival_probability": probabilities,
            "predicted_class": predictions,
        }
    )


def get_coefficient_direction(
    coef_summary: pd.DataFrame,
    feature_name: str,
) -> str:
    """Return expected direction implied by the coefficient sign."""
    matched = coef_summary.loc[coef_summary["feature"] == feature_name]

    if matched.empty:
        return "feature_missing"

    coefficient = float(matched["coefficient"].iloc[0])

    if coefficient > 0:
        return "above_baseline"
    if coefficient < 0:
        return "below_baseline"
    return "same_as_baseline"


def check_hypothetical_expectations(
    results: pd.DataFrame,
    coef_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Check whether hypothetical predictions align with baseline expectations."""
    baseline_probability = float(
        results.loc[
            results["profile"] == "baseline",
            "predicted_survival_probability",
        ].iloc[0]
    )

    expectation_rows = [
        {
            "profile": "electronics_store",
            "feature": "business_category_electronics_store_first12m_max",
        },
        {
            "profile": "vape_shop",
            "feature": "business_category_electronic_cigarette_dealer_first12m_max",
        },
        {
            "profile": "bingo_operator",
            "feature": "business_category_bingo_game_operator_first12m_max",
        },
        {
            "profile": "many_licenses",
            "feature": "active_license_count_first12m_mean",
        },
        {
            "profile": "industrial_laundry",
            "feature": "business_category_industrial_laundry_first12m_max",
        },
        {
            "profile": "debt_collection",
            "feature": "business_category_debt_collection_agency_first12m_max",
        },
        {
            "profile": "home_improvement_contractor",
            "feature": "business_category_home_improvement_contractor_first12m_max",
        },
    ]

    checks: list[dict[str, Any]] = []

    for row in expectation_rows:
        profile_name = row["profile"]
        feature_name = row["feature"]

        matched_profile = results.loc[results["profile"] == profile_name]
        if matched_profile.empty:
            checks.append(
                {
                    "profile": profile_name,
                    "feature": feature_name,
                    "expected_vs_baseline": "profile_missing",
                    "actual_vs_baseline": "profile_missing",
                    "matches_expectation": False,
                }
            )
            continue

        profile_probability = float(
            matched_profile["predicted_survival_probability"].iloc[0]
        )
        expected_direction = get_coefficient_direction(coef_summary, feature_name)

        if expected_direction == "feature_missing":
            checks.append(
                {
                    "profile": profile_name,
                    "feature": feature_name,
                    "expected_vs_baseline": "feature_missing",
                    "actual_vs_baseline": "not_checked",
                    "matches_expectation": False,
                }
            )
            continue

        if profile_probability > baseline_probability:
            actual_direction = "above_baseline"
        elif profile_probability < baseline_probability:
            actual_direction = "below_baseline"
        else:
            actual_direction = "same_as_baseline"

        matches_expectation = expected_direction == actual_direction

        checks.append(
            {
                "profile": profile_name,
                "feature": feature_name,
                "expected_vs_baseline": expected_direction,
                "actual_vs_baseline": actual_direction,
                "matches_expectation": matches_expectation,
            }
        )

    return pd.DataFrame(checks)


def print_expectation_results(expectation_results: pd.DataFrame) -> None:
    """Print whether hypothetical predictions match expectations."""
    print("\nHYPOTHETICAL EXPECTATION CHECKS\n")
    print(expectation_results.to_string(index=False))

    checked_rows = expectation_results.loc[
        expectation_results["expected_vs_baseline"] != "feature_missing"
    ].copy()

    all_match = bool(checked_rows["matches_expectation"].all()) if not checked_rows.empty else False
    print(f"\nall_checked_expectations_met: {all_match}")


def print_prediction_results(results: pd.DataFrame) -> None:
    """Print predictions."""
    ordered_results = results.sort_values(
        "predicted_survival_probability",
        ascending=False,
    )

    print("\nHYPOTHETICAL PROFILE PREDICTIONS\n")
    print(
        ordered_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.8f}",
        )
    )


def print_top_coefficients(coef_summary: pd.DataFrame, top_n: int = 10) -> None:
    """Print top positive and negative coefficients."""
    positive = coef_summary.sort_values("coefficient", ascending=False).head(top_n)
    negative = coef_summary.sort_values("coefficient", ascending=True).head(top_n)

    print("\nTOP POSITIVE COEFFICIENTS\n")
    print(
        positive[["feature", "coefficient"]].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nTOP NEGATIVE COEFFICIENTS\n")
    print(
        negative[["feature", "coefficient"]].to_string(
            index=False, float_format=lambda x: f"{x:.6f}"))


def run_inspection(config: InspectConfig) -> None:
    """Run full model inspection workflow."""
    pipeline, kept_columns, metrics, coef_summary, train_split = load_artifacts(config)

    print_model_metrics(metrics)
    print_top_coefficients(coef_summary)

    profiles = build_hypothetical_profiles(kept_columns, train_split)
    results = predict_profiles(pipeline, profiles)
    expectation_results = check_hypothetical_expectations(results, coef_summary)

    print_expectation_results(expectation_results)
    print_prediction_results(results)


def parse_args() -> InspectConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect saved logistic regression model artifacts."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing saved logistic artifacts",
    )
    args = parser.parse_args()
    return InspectConfig(artifacts_dir=args.artifacts_dir)


def main() -> None:
    """Entry point."""
    config = parse_args()
    run_inspection(config)


if __name__ == "__main__":
    main()
