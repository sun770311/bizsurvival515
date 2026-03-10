"""Module to inspect and interpret Cox Proportional Hazards models."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class InspectCoxConfig:
    """Configuration for Cox model inspection."""
    artifacts_dir: Path


def load_artifacts(
    artifacts_dir: Path,
) -> dict[str, CoxPHFitter | StandardScaler | list[str] | pd.DataFrame]:
    """Load standard Cox artifacts from disk."""
    with (artifacts_dir / "coxph_model.pkl").open("rb") as file_obj:
        model = pickle.load(file_obj)

    with (artifacts_dir / "coxph_scaler.pkl").open("rb") as file_obj:
        scaler = pickle.load(file_obj)

    with (artifacts_dir / "coxph_kept_columns.pkl").open("rb") as file_obj:
        kept_columns = pickle.load(file_obj)

    coef_summary = pd.read_csv(artifacts_dir / "coxph_summary.csv")

    return {
        "model": model,
        "scaler": scaler,
        "kept_columns": kept_columns,
        "coef_summary": coef_summary,
    }


def build_baseline_profile(
    joined: pd.DataFrame,
    kept_columns: list[str],
) -> pd.DataFrame:
    """Create a baseline business profile matching the feature space."""
    recent_joined = joined.sort_values(["business_id", "month"]).groupby("business_id").last()

    profile_dict = {}
    for column in kept_columns:
        if column in recent_joined.columns:
            profile_dict[column] = float(recent_joined[column].median())
        else:
            profile_dict[column] = 0.0

    profile = pd.DataFrame([profile_dict])
    return profile


def zero_out_category_columns(profile: pd.DataFrame) -> pd.DataFrame:
    """Zero out specific business category features for a fresh profile."""
    updated = profile.copy()
    for column in updated.columns:
        if column.startswith("business_category_") and column != "business_category_sum":
            updated[column] = 0.0
    return updated


def make_hypothetical_profiles(
    baseline_profile: pd.DataFrame,
    active_license_override: int = 5,
) -> pd.DataFrame:
    """Generate various business profiles to inspect model behavior."""
    profiles = {"baseline": baseline_profile.copy()}

    def _create_category_profile(category_col: str) -> pd.DataFrame:
        prof = zero_out_category_columns(baseline_profile)
        if category_col in prof.columns:
            prof[category_col] = 1.0
        return prof

    profiles["electronics_store"] = _create_category_profile("business_category_electronics_store")
    profiles["electronic_cigarette_dealer"] = _create_category_profile(
        "business_category_electronic_cigarette_dealer"
    )
    profiles["bingo_game_operator"] = _create_category_profile(
        "business_category_bingo_game_operator"
    )

    multi_license = baseline_profile.copy()
    if "active_license_count" in multi_license.columns:
        multi_license["active_license_count"] = float(active_license_override)
    profiles["multi_license_business"] = multi_license

    return pd.concat(profiles.values(), keys=profiles.keys()).reset_index(level=1, drop=True)


def validate_feature_availability(
    kept_columns: list[str],
    required_features: list[str],
) -> None:
    """Ensure features required for hypothetical testing are in the model."""
    missing = [feature for feature in required_features if feature not in kept_columns]
    if missing:
        raise ValueError(f"Required hypothetical-test features missing from model: {missing}")


def get_feature_direction(coef_summary: pd.DataFrame, feature_name: str) -> int | None:
    """Determine the directional effect of a specific feature."""
    matches = coef_summary[coef_summary["feature"] == feature_name]
    if matches.empty:
        return None

    coef = matches.iloc[0]["coef"]
    if coef > 0:
        return 1
    if coef < 0:
        return -1
    return 0


def score_profiles(
    profiles: pd.DataFrame,
    model: CoxPHFitter,
    scaler: StandardScaler,
    kept_columns: list[str],
    survival_times: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score profiles using the scaler and Cox model."""
    scaled_profiles = scaler.transform(profiles[kept_columns])
    scaled_df = pd.DataFrame(
        scaled_profiles,
        columns=kept_columns,
        index=profiles.index,
    )

    partial_hazards = model.predict_partial_hazard(scaled_df)
    survival_curves = model.predict_survival_function(scaled_df, times=survival_times)

    results = pd.DataFrame(index=profiles.index)
    results["partial_hazard"] = partial_hazards

    for time_val in survival_times:
        results[f"survival_prob_{time_val}m"] = survival_curves.loc[time_val].values

    return results, survival_curves


def compare_profile_to_baseline(results: pd.DataFrame, profile_name: str) -> str:
    """Compare a profile's hazard to the baseline hazard."""
    baseline_hazard = results.loc["baseline", "partial_hazard"]
    profile_hazard = results.loc[profile_name, "partial_hazard"]

    if profile_hazard > baseline_hazard:
        return "higher_risk"
    if profile_hazard < baseline_hazard:
        return "lower_risk"
    return "same_risk"


def check_directional_expectations(
    results: pd.DataFrame,
    coef_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Check if profile score directions match the learned coefficients."""
    checks = []

    profile_feature_map = {
        "electronics_store": "business_category_electronics_store",
        "electronic_cigarette_dealer": "business_category_electronic_cigarette_dealer",
        "bingo_game_operator": "business_category_bingo_game_operator",
        "multi_license_business": "active_license_count",
    }

    for profile_name, feature_name in profile_feature_map.items():
        if profile_name not in results.index:
            continue

        actual_relation = compare_profile_to_baseline(results, profile_name)

        direction = get_feature_direction(coef_summary, feature_name)

        if direction == 1:
            expected_relation = "higher_risk"
        elif direction == -1:
            expected_relation = "lower_risk"
        elif direction == 0:
            expected_relation = "same_risk"
        else:
            expected_relation = "unknown"

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
