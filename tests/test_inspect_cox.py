"""Unit tests for Cox model inspection helpers."""

import unittest

import pandas as pd

from pipeline.inspect_cox import (
    REQUIRED_HYPOTHETICAL_FEATURES,
    build_baseline_profile,
    check_directional_expectations,
    compare_profile_to_baseline,
    get_feature_direction,
    load_artifacts,
    load_joined_dataset,
    make_hypothetical_profiles,
    score_profiles,
    validate_feature_availability,
    zero_out_category_columns,
)
from tests.pipeline_test_helpers import fitted_cox_output_dir
from tests.test_helpers import TEST_DATA_DIR, assert_month_column_parsed


class TestInspectCox(unittest.TestCase):
    """Test suite for inspecting trained standard Cox model artifacts."""

    def test_load_artifacts(self):
        """Load saved Cox artifacts and verify required entries exist."""
        with fitted_cox_output_dir() as artifact_dir:
            artifacts = load_artifacts(artifact_dir)

        self.assertIn("model", artifacts)
        self.assertIn("scaler", artifacts)
        self.assertIn("kept_columns", artifacts)
        self.assertIn("coef_summary", artifacts)
        self.assertIsInstance(artifacts["kept_columns"], list)
        self.assertGreater(len(artifacts["kept_columns"]), 0)
        self.assertIsInstance(artifacts["coef_summary"], pd.DataFrame)
        self.assertFalse(artifacts["coef_summary"].empty)

    def test_load_joined_dataset_parses_month(self):
        """Load the joined dataset and parse the month column as datetime."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        assert_month_column_parsed(self, joined)

    def test_build_baseline_profile_matches_kept_columns(self):
        """Build a one-row baseline profile aligned to kept feature columns."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        kept_columns = ["active_license_count", "business_category_electronics_store"]

        baseline = build_baseline_profile(joined, kept_columns)

        self.assertEqual(baseline.shape, (1, 2))
        self.assertEqual(baseline.columns.tolist(), kept_columns)

    def test_zero_out_category_columns_ignores_sum_column(self):
        """Zero out category columns while leaving summary columns unchanged."""
        profile = pd.DataFrame(
            {
                "business_category_electronics_store": [1.0],
                "business_category_bingo_game_operator": [1.0],
                "business_category_sum": [2.0],
                "active_license_count": [3.0],
            }
        )

        updated = zero_out_category_columns(profile)

        self.assertEqual(updated.loc[0, "business_category_electronics_store"], 0.0)
        self.assertEqual(updated.loc[0, "business_category_bingo_game_operator"], 0.0)
        self.assertEqual(updated.loc[0, "business_category_sum"], 2.0)
        self.assertEqual(updated.loc[0, "active_license_count"], 3.0)

    def test_make_hypothetical_profiles_has_expected_index(self):
        """Build the expected named hypothetical profiles from a baseline row."""
        baseline_profile = pd.DataFrame(
            {
                "active_license_count": [1.0],
                "business_category_electronics_store": [0.0],
                "business_category_electronic_cigarette_dealer": [0.0],
                "business_category_bingo_game_operator": [0.0],
            }
        )

        profiles = make_hypothetical_profiles(
            baseline_profile=baseline_profile,
            active_license_override=5,
        )

        self.assertEqual(
            set(profiles.index),
            {
                "baseline",
                "electronics_store",
                "electronic_cigarette_dealer",
                "bingo_game_operator",
                "multi_license_business",
            },
        )

    def test_validate_feature_availability_raises_for_missing_features(self):
        """Raise an error when required hypothetical test features are missing."""
        kept_columns = ["active_license_count", "business_category_electronics_store"]

        with self.assertRaisesRegex(
            ValueError,
            "Required hypothetical-test features missing",
        ):
            validate_feature_availability(
                kept_columns=kept_columns,
                required_features=REQUIRED_HYPOTHETICAL_FEATURES,
            )

    def test_get_feature_direction_positive(self):
        """Return 1 when the coefficient direction is positive."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.5],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), 1)

    def test_get_feature_direction_negative(self):
        """Return -1 when the coefficient direction is negative."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [-0.5],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), -1)

    def test_get_feature_direction_zero(self):
        """Return 0 when the coefficient is exactly zero."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.0],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), 0)

    def test_get_feature_direction_missing(self):
        """Return None when the requested feature is missing."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.5],
            }
        )
        self.assertIsNone(get_feature_direction(coef_summary, "missing"))

    def test_compare_profile_to_baseline(self):
        """Compare a profile hazard to baseline and label relative risk."""
        results = pd.DataFrame(
            {
                "partial_hazard": [1.0, 2.0, 0.5],
            },
            index=["baseline", "riskier", "safer"],
        )

        self.assertEqual(compare_profile_to_baseline(results, "riskier"), "higher_risk")
        self.assertEqual(compare_profile_to_baseline(results, "safer"), "lower_risk")

    def test_score_profiles_returns_expected_columns(self):
        """Score hypothetical profiles and return hazard and survival outputs."""
        with fitted_cox_output_dir() as artifact_dir:
            artifacts = load_artifacts(artifact_dir)
            joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")

            kept_columns = artifacts["kept_columns"]
            baseline = build_baseline_profile(joined, kept_columns)
            profiles = make_hypothetical_profiles(
                baseline_profile=baseline,
                active_license_override=5,
            )

            results, survival_df = score_profiles(
                profiles=profiles,
                model=artifacts["model"],
                scaler=artifacts["scaler"],
                kept_columns=kept_columns,
                survival_times=[12, 24, 36],
            )

        self.assertFalse(results.empty)
        self.assertIn("partial_hazard", results.columns)
        self.assertIn("survival_prob_12m", results.columns)
        self.assertIn("survival_prob_24m", results.columns)
        self.assertIn("survival_prob_36m", results.columns)
        self.assertFalse(survival_df.empty)

    def test_check_directional_expectations_returns_expected_columns(self):
        """Compare scored profiles to coefficient-based direction expectations."""
        results = pd.DataFrame(
            {
                "partial_hazard": [1.0, 2.0, 2.5, 0.5, 0.4],
            },
            index=[
                "baseline",
                "electronics_store",
                "electronic_cigarette_dealer",
                "bingo_game_operator",
                "multi_license_business",
            ],
        )

        coef_summary = pd.DataFrame(
            {
                "feature": [
                    "business_category_electronics_store",
                    "business_category_electronic_cigarette_dealer",
                    "business_category_bingo_game_operator",
                    "active_license_count",
                ],
                "coef": [1.0, 1.0, -1.0, -1.0],
            }
        )

        checks = check_directional_expectations(results, coef_summary)

        self.assertFalse(checks.empty)
        self.assertEqual(
            set(checks.columns),
            {
                "profile",
                "feature",
                "expected_relation",
                "actual_relation",
                "matches_expectation",
            },
        )
        self.assertEqual(len(checks), 4)
        self.assertTrue(bool(checks["matches_expectation"].all()))


if __name__ == "__main__":
    unittest.main()
