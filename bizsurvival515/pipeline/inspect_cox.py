"""Inspect a trained standard CoxPH model using hypothetical business profiles.

This module loads exported standard Cox proportional hazards model artifacts,
constructs a realistic baseline business profile from the joined dataset,
creates several hypothetical profile variants, scores them with the trained
model, and checks whether the predicted risk directions match the signs of
the fitted model coefficients. It is intended as a lightweight diagnostic
tool for validating whether selected feature effects behave as expected in
model-based comparisons.

Inputs:
- joined_dataset.csv
- coxph_model.pkl
- coxph_scaler.pkl
- coxph_kept_columns.pkl
- coxph_summary.csv

Processing steps:
- Load trained standard CoxPH model artifacts
- Load the joined dataset used to derive a baseline business profile
- Build a one-row baseline profile aligned to the model's kept features
- Generate hypothetical business profiles by modifying selected features
- Standardize and score hypothetical profiles with the trained model
- Compare predicted profile risks against the baseline profile
- Check whether predicted risk directions align with coefficient signs

Outputs:
- Printed model diagnostics
- Printed hypothetical profile score table
- Printed directional expectation checks

Classes:
- InspectCoxConfig:
  Stores file paths and runtime options for hypothetical-profile inspection.

Functions:
- load_pickle:
  Load a pickle artifact from disk.
- load_artifacts:
  Load all exported standard CoxPH artifacts required for inspection.
- load_joined_dataset:
  Load the joined dataset used to derive a realistic baseline profile.
- build_baseline_profile:
  Construct a one-row baseline profile aligned with the model's kept features.
- zero_out_category_columns:
  Reset business-category indicator columns for controlled category testing.
- make_hypothetical_profiles:
  Build the baseline profile and several hypothetical business variants.
- score_profiles:
  Standardize and score hypothetical profiles with the exported CoxPH model.
- get_feature_direction:
  Return the expected risk direction implied by a coefficient in the summary.
- compare_profile_to_baseline:
  Compare a profile's predicted risk to the baseline profile's predicted risk.
- validate_feature_availability:
  Check that required hypothetical-test features are present in the model.
- check_directional_expectations:
  Compare observed risk changes to expected directions from model coefficients.
- run_directional_tests:
  Run the complete hypothetical-profile directional testing workflow.
- parse_args:
  Parse command-line arguments into an InspectCoxConfig instance.
- main:
  Execute hypothetical-profile testing from the command line and print results.
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
    """Store configuration values for hypothetical-profile Cox inspection.

    Attributes:
        joined_data_path: Path to the joined dataset used to build a baseline profile.
        artifact_dir: Directory containing exported standard CoxPH artifacts.
        survival_times: Survival times, in months, at which profiles are scored.
        active_license_override: Active license count assigned to the
            multi-license hypothetical profile.
    """

    joined_data_path: Path
    artifact_dir: Path
    survival_times: list[int]
    active_license_override: int = LICENSE_COUNT_OVERRIDE


def load_pickle(path: Path) -> object:
    """Load a pickled artifact from disk.

    Args:
        path: Path to the pickle file to load.

    Returns:
        The deserialized Python object stored in the pickle file.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled successfully.
        OSError: If an I/O error occurs while reading the file.
    """
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_artifacts(artifact_dir: Path) -> dict[str, object]:
    """Load the exported standard CoxPH artifacts required for profile inspection.

    Args:
        artifact_dir: Directory containing the saved CoxPH model, scaler,
            kept-column list, and coefficient summary.

    Returns:
        A dictionary containing the loaded model artifacts and coefficient summary.

    Raises:
        FileNotFoundError: If one or more required artifact files are missing.
        pickle.UnpicklingError: If a pickle artifact cannot be unpickled.
        pd.errors.EmptyDataError: If the coefficient summary CSV is empty.
        OSError: If an I/O error occurs while reading an artifact.
    """
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
    """Load the joined dataset used to construct a realistic baseline business profile.

    Args:
        joined_data_path: Path to the joined dataset CSV file.

    Returns:
        A dataframe containing the joined dataset, with the ``month`` column
        parsed as datetime where possible.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        OSError: If an I/O error occurs while reading the file.
    """
    joined = pd.read_csv(joined_data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def build_baseline_profile(
    joined_df: pd.DataFrame,
    kept_feature_columns: list[str],
) -> pd.DataFrame:
    """Build a one-row baseline profile aligned to the model's retained feature columns.

    Args:
        joined_df: Joined dataset used as the source for a realistic baseline row.
        kept_feature_columns: Feature columns retained by the trained CoxPH model.

    Returns:
        A one-row dataframe whose columns match the model's retained features,
        initialized from the first available business-month row where possible.

    Raises:
        IndexError: If the joined dataframe is empty.
    """
    template_row = joined_df.sort_values(["business_id", "month"]).iloc[[0]].copy()
    baseline = pd.DataFrame(0.0, index=[0], columns=kept_feature_columns)

    shared_columns = [
        column for column in kept_feature_columns if column in template_row.columns
    ]
    for column in shared_columns:
        baseline.loc[0, column] = template_row.iloc[0][column]

    return baseline


def zero_out_category_columns(profile: pd.DataFrame) -> pd.DataFrame:
    """Set business-category indicator columns to zero for controlled profile editing.

    Args:
        profile: One-row or multi-row profile dataframe.

    Returns:
        A copy of the input profile with all business-category indicator columns,
        except ``business_category_sum``, set to zero.
    """
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
    """Construct a baseline profile together with several hypothetical business variants.

    Args:
        baseline_profile: One-row baseline profile aligned to the model features.
        active_license_override: Active license count assigned to the
            multi-license hypothetical profile.

    Returns:
        A dataframe containing the baseline profile and the hypothetical profiles
        used for directional inspection.
    """
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
    """Standardize and score hypothetical profiles with the exported standard CoxPH model.

    Args:
        profiles: Dataframe containing hypothetical profiles to score.
        model: Fitted standard CoxPH model used for scoring.
        scaler: Fitted scaler used to standardize model input features.
        kept_columns: Feature columns expected by the trained model.
        survival_times: Survival times, in months, at which to evaluate survival probabilities.

    Returns:
        A tuple containing:
        - A dataframe of partial hazards and survival probabilities for each profile.
        - A dataframe of profile survival curves indexed by profile.

    Raises:
        KeyError: If one or more required kept columns are missing from ``profiles``.
        ValueError: If the scaler or model receives incompatible input dimensions.
    """
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
    """Return the expected risk direction implied by a feature's model coefficient.

    Args:
        coef_summary: Coefficient summary dataframe containing feature names and coefficients.
        feature_name: Name of the feature whose coefficient direction should be checked.

    Returns:
        ``1`` if the coefficient is positive, ``-1`` if it is negative,
        ``0`` if it is exactly zero, or ``None`` if the feature is not present
        in the coefficient summary.
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
    """Compare a profile's predicted partial hazard to the baseline profile.

    Args:
        results: Scored profile results containing ``partial_hazard`` values.
        profile_name: Name of the profile to compare against the baseline.

    Returns:
        ``"higher_risk"`` if the profile hazard exceeds the baseline hazard,
        ``"lower_risk"`` if it is below the baseline hazard, or ``"same_risk"``
        if the two hazards are equal.

    Raises:
        KeyError: If ``baseline`` or the requested profile is missing from ``results``.
    """
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
    """Validate that all required hypothetical-test features are available in the model.

    Args:
        kept_columns: Feature columns retained by the trained model.
        required_features: Features required to construct the hypothetical tests.

    Returns:
        None.

    Raises:
        ValueError: If one or more required features are missing from ``kept_columns``.
    """
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
    """Check whether observed profile risk changes match coefficient-based expectations.

    Args:
        results: Scored profile results containing predicted partial hazards.
        coef_summary: Coefficient summary dataframe from the trained CoxPH model.

    Returns:
        A dataframe listing each hypothetical profile, its associated feature,
        the expected risk relation, the actual risk relation, and whether the
        expectation was satisfied.
    """
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
    """Run the complete hypothetical-profile directional testing workflow.

    Args:
        config: Configuration specifying data paths, artifact paths,
            scoring times, and hypothetical-profile settings.

    Returns:
        A dataframe containing scored profile results for the baseline and
        hypothetical business profiles.

    Raises:
        ValueError: If required hypothetical-test features are missing.
        AssertionError: If one or more directional expectations fail.
        FileNotFoundError: If required input files are missing.
    """
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
    """Parse command-line arguments and construct an InspectCoxConfig instance.

    Args:
        None.

    Returns:
        An InspectCoxConfig populated from command-line argument values.
    """
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
