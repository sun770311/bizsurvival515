"""Module to inspect and interpret logistic regression models for business survival."""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class InspectConfig:
    """Configuration for model inspection."""
    artifacts_dir: Path


def load_artifacts(
    config: InspectConfig,
) -> tuple[Pipeline, list[str], dict[str, object], pd.DataFrame]:
    """Load model pipeline, metadata, and evaluation metrics from disk."""
    with (config.artifacts_dir / "logistic_pipeline.pkl").open("rb") as file_obj:
        pipeline = pickle.load(file_obj)

    with (config.artifacts_dir / "logistic_kept_columns.pkl").open("rb") as file_obj:
        kept_columns = pickle.load(file_obj)

    metrics_path = config.artifacts_dir / "logistic_evaluation_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)

    coef_summary = pd.read_csv(config.artifacts_dir / "logistic_coefficient_summary.csv")

    return pipeline, kept_columns, metrics, coef_summary


def build_baseline_profile(kept_columns: list[str]) -> pd.DataFrame:
    """Create a baseline business profile matching the exact feature space of the model."""
    profile = pd.DataFrame(0.0, index=[0], columns=kept_columns)
    return profile


def build_hypothetical_profiles(kept_columns: list[str]) -> pd.DataFrame:
    """Generate various hypothetical business profiles to inspect model behavior."""
    baseline = build_baseline_profile(kept_columns)

    profiles = {"baseline": baseline.copy()}

    def _create_category_profile(category_col: str) -> pd.DataFrame:
        prof = baseline.copy()
        if category_col in prof.columns:
            prof[category_col] = 1.0
        return prof

    profiles["electronics_store"] = _create_category_profile(
        "business_category_electronics_store"
    )
    profiles["vape_shop"] = _create_category_profile(
        "business_category_electronic_cigarette_dealer"
    )
    profiles["bingo_operator"] = _create_category_profile(
        "business_category_bingo_game_operator"
    )
    profiles["laundries"] = _create_category_profile(
        "business_category_laundries"
    )
    profiles["car_wash"] = _create_category_profile(
        "business_category_car_wash"
    )
    profiles["debt_collection"] = _create_category_profile(
        "business_category_debt_collection_agency"
    )

    many_licenses = baseline.copy()
    if "active_license_count" in many_licenses.columns:
        many_licenses["active_license_count"] = 5.0
    profiles["many_licenses"] = many_licenses

    profiles_df = pd.concat(profiles.values(), keys=profiles.keys())
    return profiles_df.reset_index(level=1, drop=True)


def predict_profiles(pipeline: Pipeline, profiles: pd.DataFrame) -> pd.DataFrame:
    """Predict survival probabilities and classes for a set of profiles."""
    probabilities = pipeline.predict_proba(profiles)[:, 1]
    predictions = pipeline.predict(profiles)

    results = pd.DataFrame(
        {
            "profile": profiles.index,
            "predicted_survival_probability": probabilities,
            "predicted_class": predictions,
        }
    )
    return results


def get_coefficient_direction(coef_summary: pd.DataFrame, feature_name: str) -> str:
    """Determine the expected direction of effect based on the learned coefficient."""
    matches = coef_summary[coef_summary["feature"] == feature_name]

    if matches.empty:
        return "feature_missing"

    coef = matches.iloc[0]["coefficient"]

    if coef > 0:
        return "above_baseline"
    if coef < 0:
        return "below_baseline"

    return "same_as_baseline"


def check_hypothetical_expectations(
    results: pd.DataFrame,
    coef_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Check if the predicted probabilities match the expectations."""
    baseline_prob = results.loc[results["profile"] == "baseline", "predicted_survival_probability"]
    baseline_prob_val = baseline_prob.iloc[0] if not baseline_prob.empty else 0.5

    checks = []

    profile_feature_map = {
        "electronics_store": "business_category_electronics_store",
        "vape_shop": "business_category_electronic_cigarette_dealer",
        "bingo_operator": "business_category_bingo_game_operator",
        "laundries": "business_category_laundries",
        "car_wash": "business_category_car_wash",
        "debt_collection": "business_category_debt_collection_agency",
        "many_licenses": "active_license_count",
    }

    for profile_name, feature_name in profile_feature_map.items():
        if profile_name not in results["profile"].values:
            continue

        profile_row = results[results["profile"] == profile_name]
        profile_prob = profile_row["predicted_survival_probability"].iloc[0]

        expected_direction = get_coefficient_direction(coef_summary, feature_name)

        if profile_prob > baseline_prob_val:
            actual_direction = "above_baseline"
        elif profile_prob < baseline_prob_val:
            actual_direction = "below_baseline"
        else:
            actual_direction = "same_as_baseline"

        if expected_direction == "feature_missing":
            matches = True
        else:
            matches = expected_direction == actual_direction

        checks.append(
            {
                "profile": profile_name,
                "feature": feature_name,
                "expected_vs_baseline": expected_direction,
                "actual_vs_baseline": actual_direction,
                "matches_expectation": matches,
            }
        )

    return pd.DataFrame(checks)
