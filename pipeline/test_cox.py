"""Directional tests for hypothetical CoxPH business profiles.

This script loads exported standard CoxPH model artifacts, creates a baseline
profile plus four hypothetical business profiles, scores them, and verifies
that predicted risk trajectories align directionally with coefficient signs.

Input:
- joined_dataset.csv
- cox_outputs/standard/coxph_model.pkl
- cox_outputs/standard/coxph_scaler.pkl
- cox_outputs/standard/coxph_kept_columns.pkl
- cox_outputs/standard/coxph_summary.csv

Test logic:
- Electronics store should be riskier than baseline
- Electronic cigarette dealer should be riskier than baseline
- Bingo game operator should be safer than baseline
- Higher active license count should be safer than baseline

This is a directional test only. It does not require exact rank ordering among
all hypothetical profiles.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_TIMES = [12, 24, 36, 60, 120]
LICENSE_COUNT_OVERRIDE = 5


@dataclass(frozen=True)
class HypothesisTestConfig:
    """Configuration for hypothetical-profile directional testing."""

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
        column for column in updated.columns
        if column.startswith("business_category_")
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
    model: object,
    scaler: object,
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


def assert_higher_risk_than_baseline(
    results: pd.DataFrame,
    profile_name: str,
) -> None:
    """Assert profile is directionally riskier than baseline."""
    baseline_hazard = float(results.loc["baseline", "partial_hazard"])
    profile_hazard = float(results.loc[profile_name, "partial_hazard"])
    if profile_hazard <= baseline_hazard:
        raise AssertionError(
            f"{profile_name} expected higher hazard than baseline, "
            f"but got {profile_hazard:.6f} <= {baseline_hazard:.6f}."
        )

    survival_columns = [column for column in results.columns if column.startswith("survival_prob_")]
    for column in survival_columns:
        baseline_survival = float(results.loc["baseline", column])
        profile_survival = float(results.loc[profile_name, column])
        if profile_survival >= baseline_survival:
            raise AssertionError(
                f"{profile_name} expected lower survival than baseline at {column}, "
                f"but got {profile_survival:.6f} >= {baseline_survival:.6f}."
            )


def assert_lower_risk_than_baseline(
    results: pd.DataFrame,
    profile_name: str,
) -> None:
    """Assert profile is directionally safer than baseline."""
    baseline_hazard = float(results.loc["baseline", "partial_hazard"])
    profile_hazard = float(results.loc[profile_name, "partial_hazard"])
    if profile_hazard >= baseline_hazard:
        raise AssertionError(
            f"{profile_name} expected lower hazard than baseline, "
            f"but got {profile_hazard:.6f} >= {baseline_hazard:.6f}."
        )

    survival_columns = [column for column in results.columns if column.startswith("survival_prob_")]
    for column in survival_columns:
        baseline_survival = float(results.loc["baseline", column])
        profile_survival = float(results.loc[profile_name, column])
        if profile_survival <= baseline_survival:
            raise AssertionError(
                f"{profile_name} expected higher survival than baseline at {column}, "
                f"but got {profile_survival:.6f} <= {baseline_survival:.6f}."
            )


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


def run_directional_tests(config: HypothesisTestConfig) -> pd.DataFrame:
    """Run hypothetical-profile directional tests and return score table."""
    artifacts = load_artifacts(config.artifact_dir)
    joined = load_joined_dataset(config.joined_data_path)

    kept_columns = artifacts["kept_columns"]
    coef_summary = artifacts["coef_summary"]

    validate_feature_availability(
        kept_columns=kept_columns,
        required_features=[
            "business_category_electronics_store",
            "business_category_electronic_cigarette_dealer",
            "business_category_bingo_game_operator",
            "active_license_count",
        ],
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

    electronics_direction = get_feature_direction(
        coef_summary,
        "business_category_electronics_store",
    )
    e_cigarette_direction = get_feature_direction(
        coef_summary,
        "business_category_electronic_cigarette_dealer",
    )
    bingo_direction = get_feature_direction(
        coef_summary,
        "business_category_bingo_game_operator",
    )
    license_direction = get_feature_direction(
        coef_summary,
        "active_license_count",
    )

    if electronics_direction != 1:
        raise AssertionError(
            "Expected business_category_electronics_store to have positive coefficient."
        )
    if e_cigarette_direction != 1:
        raise AssertionError(
            "Expected business_category_electronic_cigarette_dealer to have positive coefficient."
        )
    if bingo_direction != -1:
        raise AssertionError(
            "Expected business_category_bingo_game_operator to have negative coefficient."
        )
    if license_direction != -1:
        raise AssertionError(
            "Expected active_license_count to have negative coefficient."
        )

    assert_higher_risk_than_baseline(results, "electronics_store")
    assert_higher_risk_than_baseline(results, "electronic_cigarette_dealer")
    assert_lower_risk_than_baseline(results, "bingo_game_operator")
    assert_lower_risk_than_baseline(results, "multi_license_business")

    return results


def main() -> None:
    """Run hypothetical-profile directional testing."""
    config = HypothesisTestConfig(
        joined_data_path=Path("/content/drive/MyDrive/joined_dataset.csv"),
        artifact_dir=Path("/content/drive/MyDrive/cox_outputs/standard"),
        survival_times=DEFAULT_TIMES,
    )
    results = run_directional_tests(config)
    print("Directional hypothetical-profile tests passed.")
    print(results.round(6).to_string())


if __name__ == "__main__":
    main()