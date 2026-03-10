"""Inspect trained logistic regression survival model."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class InspectConfig:
    """Configuration for logistic model inspection."""

    artifacts_dir: Path


def load_artifacts(
    config: InspectConfig,
) -> tuple[Any, list[str], dict[str, Any], pd.DataFrame]:
    """Load saved model artifacts."""
    base = config.artifacts_dir

    with (base / "logistic_pipeline.pkl").open("rb") as file_obj:
        pipeline = pickle.load(file_obj)

    with (base / "logistic_kept_columns.pkl").open("rb") as file_obj:
        kept_columns = pickle.load(file_obj)

    with (base / "logistic_evaluation_metrics.json").open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    coef_summary = pd.read_csv(base / "logistic_coefficient_summary.csv")

    return pipeline, kept_columns, metrics, coef_summary


def print_model_metrics(metrics: dict[str, Any]) -> None:
    """Print evaluation metrics."""
    print("\nMODEL METRICS\n")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def build_baseline_profile(kept_columns: list[str]) -> pd.DataFrame:
    """Create baseline feature vector."""
    return pd.DataFrame(
        np.zeros((1, len(kept_columns))),
        columns=kept_columns,
    )


def build_hypothetical_profiles(kept_columns: list[str]) -> pd.DataFrame:
    """Construct hypothetical business profiles."""
    baseline = build_baseline_profile(kept_columns)

    profiles: list[pd.DataFrame] = []

    base = baseline.copy()
    base.index = ["baseline"]
    profiles.append(base)

    electronics = baseline.copy()
    if "business_category_electronics_store" in electronics.columns:
        electronics.loc[0, "business_category_electronics_store"] = 1
    electronics.index = ["electronics_store"]
    profiles.append(electronics)

    vape = baseline.copy()
    if "business_category_electronic_cigarette_dealer" in vape.columns:
        vape.loc[0, "business_category_electronic_cigarette_dealer"] = 1
    vape.index = ["vape_shop"]
    profiles.append(vape)

    bingo = baseline.copy()
    if "business_category_bingo_game_operator" in bingo.columns:
        bingo.loc[0, "business_category_bingo_game_operator"] = 1
    bingo.index = ["bingo_operator"]
    profiles.append(bingo)

    multi_license = baseline.copy()
    if "active_license_count" in multi_license.columns:
        multi_license.loc[0, "active_license_count"] = 5
    multi_license.index = ["many_licenses"]
    profiles.append(multi_license)

    laundry = baseline.copy()
    if "business_category_laundries" in laundry.columns:
        laundry.loc[0, "business_category_laundries"] = 1
    laundry.index = ["laundries"]
    profiles.append(laundry)

    car_wash = baseline.copy()
    if "business_category_car_wash" in car_wash.columns:
        car_wash.loc[0, "business_category_car_wash"] = 1
    car_wash.index = ["car_wash"]
    profiles.append(car_wash)

    debt = baseline.copy()
    if "business_category_debt_collection_agency" in debt.columns:
        debt.loc[0, "business_category_debt_collection_agency"] = 1
    debt.index = ["debt_collection"]
    profiles.append(debt)

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
        {"profile": "electronics_store", "feature": "business_category_electronics_store"},
        {"profile": "vape_shop", "feature": "business_category_electronic_cigarette_dealer"},
        {"profile": "bingo_operator", "feature": "business_category_bingo_game_operator"},
        {"profile": "many_licenses", "feature": "active_license_count"},
        {"profile": "laundries", "feature": "business_category_laundries"},
        {"profile": "car_wash", "feature": "business_category_car_wash"},
        {"profile": "debt_collection", "feature": "business_category_debt_collection_agency"},
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

    all_match = bool(expectation_results["matches_expectation"].all())
    print(f"\nall_expectations_met: {all_match}")


def print_prediction_results(results: pd.DataFrame) -> None:
    """Print predictions."""
    ordered_results = results.sort_values(
        "predicted_survival_probability",
        ascending=False,
    )

    print("\nHYPOTHETICAL PROFILE PREDICTIONS\n")
    print(ordered_results.to_string(index=False))


def run_inspection(config: InspectConfig) -> None:
    """Run full model inspection workflow."""
    pipeline, kept_columns, metrics, coef_summary = load_artifacts(config)

    print_model_metrics(metrics)

    profiles = build_hypothetical_profiles(kept_columns)
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