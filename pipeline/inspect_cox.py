"""Inspect trained standard CoxPH model with hypothetical business profiles.

This module loads exported standard CoxPH model artifacts, creates a baseline
profile plus hypothetical business profiles, scores them, and checks whether
predicted risk directions align with coefficient signs.

Inputs:
- joined_dataset.csv
- coxph_model.pkl
- coxph_scaler.pkl
- coxph_kept_columns.pkl
- coxph_summary.csv

Outputs:
- Printed model diagnostics
- Printed hypothetical profile score table
- Printed directional expectation checks
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_TIMES = [12, 24, 36, 60, 120]
LICENSE_COUNT_OVERRIDE = 5
REQUIRED_HYPOTHETICAL_FEATURES = [
    "business_category_electronics_store",
    "business_category_electronic_cigarette_dealer",
    "business_category_bingo_game_operator",
    "active_license_count",
]


@dataclass(frozen=True)
class InspectCoxConfig:
    """Configuration for hypothetical-profile Cox inspection."""

    joined_data_path: Path
    artifact_dir: Path
    survival_times: list[int]
    active_license_override: int = LICENSE_COUNT_OVERRIDE


def load_pickle(path: Path) -> object:
    """Load a pickle artifact from disk."""
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_artifacts(artifact_dir: Path) -> dict[str, object]:
    """Load standard CoxPH artifacts required for profile scoring."""
    model = load_pickle(artifact_dir / "coxph_model.pkl")
    scaler = load_pickle(artifact_dir / "coxph_scaler.pkl")
    kept_columns = load_pickle(artifact_dir / "coxph_kept_columns.pkl")
    coef_summary = pd.read_csv(artifact_dir / "coxph_summary.csv")

    return {
        "model": model,
        "scaler": scaler,
        "kept_columns": kept_columns,
        "coef_summary": coef_summary,
    }


def load_joined_dataset(joined_data_path: Path) -> pd.DataFrame:
    """Load joined dataset used to construct a realistic baseline profile."""
    joined = pd.read_csv(joined_data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def build_baseline_profile(
    joined_df: pd.DataFrame,
    kept_feature_columns: list[str],
) -> pd.DataFrame:
    """Create a one-row baseline profile aligned to the model's kept columns."""
    template_row = joined_df.sort_values(["business_id", "month"]).iloc[[0]].copy()
    baseline = pd.DataFrame(0.0, index=[0], columns=kept_feature_columns)

    shared_columns = [
        column for column in kept_feature_columns if column in template_row.columns
    ]
    for column in shared_columns:
        baseline.loc[0, column] = template_row.iloc[0][column]

    return baseline


def zero_out_category_columns(profile: pd.DataFrame) -> pd.DataFrame:
    """Zero out all business category columns for controlled category testing."""
    updated = profile.copy()
    category_columns = [
        column
        for column in updated.columns
        if column.startswith("business_category_")
        and column != "business_category_sum"
    ]
    for column in category_columns:
        updated.loc[:, column] = 0.0
    return updated


def make_hypothetical_profiles(
    baseline_profile: pd.DataFrame,
    active_license_override: int,
) -> pd.DataFrame:
    """Construct baseline plus four hypothetical business profiles."""
    profiles: list[pd.DataFrame] = []

    baseline = baseline_profile.copy()
    baseline.index = ["baseline"]
    profiles.append(baseline)

    electronics_store = zero_out_category_columns(baseline_profile)
    if "business_category_electronics_store" in electronics_store.columns:
        electronics_store.loc[0, "business_category_electronics_store"] = 1.0
    if "active_license_count" in electronics_store.columns:
        electronics_store.loc[0, "active_license_count"] = 1.0
    electronics_store.index = ["electronics_store"]
    profiles.append(electronics_store)

    e_cigarette = zero_out_category_columns(baseline_profile)
    if "business_category_electronic_cigarette_dealer" in e_cigarette.columns:
        e_cigarette.loc[0, "business_category_electronic_cigarette_dealer"] = 1.0
    if "active_license_count" in e_cigarette.columns:
        e_cigarette.loc[0, "active_license_count"] = 1.0
    e_cigarette.index = ["electronic_cigarette_dealer"]
    profiles.append(e_cigarette)

    bingo_operator = zero_out_category_columns(baseline_profile)
    if "business_category_bingo_game_operator" in bingo_operator.columns:
        bingo_operator.loc[0, "business_category_bingo_game_operator"] = 1.0
    if "active_license_count" in bingo_operator.columns:
        bingo_operator.loc[0, "active_license_count"] = 1.0
    bingo_operator.index = ["bingo_game_operator"]
    profiles.append(bingo_operator)

    multi_license = baseline_profile.copy()
    if "active_license_count" in multi_license.columns:
        multi_license.loc[0, "active_license_count"] = float(active_license_override)
    multi_license.index = ["multi_license_business"]
    profiles.append(multi_license)

    return pd.concat(profiles, axis=0)


def score_profiles(
    profiles: pd.DataFrame,
    model: Any,
    scaler: Any,
    kept_columns: list[str],
    survival_times: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score hypothetical profiles with the exported standard CoxPH model."""
    feature_df = profiles[kept_columns].copy()
    scaled_array = scaler.transform(feature_df)

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=kept_columns,
        index=profiles.index,
    )

    partial_hazard = model.predict_partial_hazard(scaled_df)
    survival_df = model.predict_survival_function(
        scaled_df,
        times=survival_times,
    ).T

    results = pd.DataFrame(index=profiles.index)
    results["partial_hazard"] = partial_hazard.values.flatten()

    for month in survival_times:
        results[f"survival_prob_{month}m"] = survival_df[month].values

    return results, survival_df


def get_feature_direction(
    coef_summary: pd.DataFrame,
    feature_name: str,
) -> int | None:
    """Return expected direction from coefficient summary.

    Returns:
    - 1 for positive coefficient
    - -1 for negative coefficient
    - 0 for zero coefficient
    - None if feature is unavailable
    """
    matched = coef_summary.loc[coef_summary["feature"] == feature_name]
    if matched.empty:
        return None

    coefficient = float(matched["coef"].iloc[0])
    if coefficient > 0:
        return 1
    if coefficient < 0:
        return -1
    return 0


def compare_profile_to_baseline(
    results: pd.DataFrame,
    profile_name: str,
) -> str:
    """Compare a profile's risk to baseline."""
    baseline_hazard = float(results.loc["baseline", "partial_hazard"])
    profile_hazard = float(results.loc[profile_name, "partial_hazard"])

    if profile_hazard > baseline_hazard:
        return "higher_risk"
    if profile_hazard < baseline_hazard:
        return "lower_risk"
    return "same_risk"


def validate_feature_availability(
    kept_columns: list[str],
    required_features: list[str],
) -> None:
    """Ensure required hypothetical-test features are present in the model."""
    missing_features = [
        feature for feature in required_features if feature not in kept_columns
    ]
    if missing_features:
        raise ValueError(
            "Required hypothetical-test features missing from kept columns: "
            f"{missing_features}"
        )


def check_directional_expectations(
    results: pd.DataFrame,
    coef_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Check hypothetical profiles against coefficient directions."""
    expectation_rows = [
        {
            "profile": "electronics_store",
            "feature": "business_category_electronics_store",
        },
        {
            "profile": "electronic_cigarette_dealer",
            "feature": "business_category_electronic_cigarette_dealer",
        },
        {
            "profile": "bingo_game_operator",
            "feature": "business_category_bingo_game_operator",
        },
        {
            "profile": "multi_license_business",
            "feature": "active_license_count",
        },
    ]

    checks: list[dict[str, object]] = []

    for row in expectation_rows:
        profile_name = row["profile"]
        feature_name = row["feature"]

        coef_direction = get_feature_direction(coef_summary, feature_name)
        actual_relation = compare_profile_to_baseline(results, profile_name)

        if coef_direction == 1:
            expected_relation = "higher_risk"
        elif coef_direction == -1:
            expected_relation = "lower_risk"
        elif coef_direction == 0:
            expected_relation = "same_risk"
        else:
            expected_relation = "feature_missing"

        checks.append(
            {
                "profile": profile_name,
                "feature": feature_name,
                "expected_relation": expected_relation,
                "actual_relation": actual_relation,
                "matches_expectation": expected_relation == actual_relation,
            }
        )

    return pd.DataFrame(checks)


def run_directional_tests(config: InspectCoxConfig) -> pd.DataFrame:
    """Run hypothetical-profile directional tests and return score table."""
    artifacts = load_artifacts(config.artifact_dir)
    joined = load_joined_dataset(config.joined_data_path)

    kept_columns = artifacts["kept_columns"]

    validate_feature_availability(
        kept_columns=kept_columns,
        required_features=REQUIRED_HYPOTHETICAL_FEATURES,
    )

    baseline_profile = build_baseline_profile(joined, kept_columns)
    profiles = make_hypothetical_profiles(
        baseline_profile=baseline_profile,
        active_license_override=config.active_license_override,
    )

    results, _survival_df = score_profiles(
        profiles=profiles,
        model=artifacts["model"],
        scaler=artifacts["scaler"],
        kept_columns=kept_columns,
        survival_times=config.survival_times,
    )

    checks = check_directional_expectations(results, artifacts["coef_summary"])

    if not bool(checks["matches_expectation"].all()):
        raise AssertionError("One or more Cox directional expectations failed.")

    return results


def parse_args() -> InspectCoxConfig:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Inspect standard CoxPH model with hypothetical profiles."
    )
    parser.add_argument(
        "--joined-data",
        type=Path,
        required=True,
        help="Path to joined_dataset.csv",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Directory containing standard CoxPH artifacts",
    )
    parser.add_argument(
        "--survival-times",
        type=int,
        nargs="+",
        default=DEFAULT_TIMES,
        help="Survival times in months for scoring hypothetical profiles",
    )
    parser.add_argument(
        "--active-license-override",
        type=int,
        default=LICENSE_COUNT_OVERRIDE,
        help="Override active license count for multi-license hypothetical profile",
    )
    args = parser.parse_args()

    return InspectCoxConfig(
        joined_data_path=args.joined_data,
        artifact_dir=args.artifact_dir,
        survival_times=args.survival_times,
        active_license_override=args.active_license_override,
    )


def main() -> None:
    """Run hypothetical-profile directional testing."""
    config = parse_args()
    results = run_directional_tests(config)
    print("Directional hypothetical-profile tests passed.")
    print(results.round(6).to_string())


if __name__ == "__main__":
    main()
