"""Tests for the inspect_cox module."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.cox import CoxConfig, run_standard_cox_pipeline
from pipeline.utils import load_joined_dataset
from pipeline.inspect_cox import (
    build_baseline_profile,
    check_directional_expectations,
    compare_profile_to_baseline,
    get_feature_direction,
    load_artifacts,
    make_hypothetical_profiles,
    score_profiles,
    validate_feature_availability,
    zero_out_category_columns,
)


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestInspectCox(unittest.TestCase):
    """Test suite for inspecting Cox models using hypothetical business profiles."""

    def create_cox_standard_artifacts_dir(self, tmp_path: Path) -> None:
        """Create a temporary directory and generate standard Cox artifacts for testing."""
        config = CoxConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=tmp_path,
        )
        run_standard_cox_pipeline(config)

    def test_load_artifacts(self):
        """Test that Cox model artifacts are successfully loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_cox_standard_artifacts_dir(tmp_path)
            artifacts = load_artifacts(tmp_path)

            self.assertIn("model", artifacts)
            self.assertIn("scaler", artifacts)
            self.assertIn("kept_columns", artifacts)
            self.assertIn("coef_summary", artifacts)
            self.assertIsInstance(artifacts["kept_columns"], list)
            self.assertGreater(len(artifacts["kept_columns"]), 0)
            self.assertIsInstance(artifacts["coef_summary"], pd.DataFrame)
            self.assertFalse(artifacts["coef_summary"].empty)

    def test_build_baseline_profile_matches_kept_columns(self):
        """Test that the generated baseline profile matches the model's kept columns."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        kept_columns = ["active_license_count", "business_category_electronics_store"]

        baseline = build_baseline_profile(joined, kept_columns)

        self.assertEqual(baseline.shape, (1, 2))
        self.assertEqual(baseline.columns.tolist(), kept_columns)

    def test_zero_out_category_columns_ignores_sum_column(self):
        """Test that zeroing out category columns ignores the category sum column."""
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
        """Test that generating hypothetical profiles produces the expected indices."""
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
        """Test that validation raises a ValueError when required features are missing."""
        kept_columns = ["active_license_count"]
        with self.assertRaisesRegex(ValueError, "Required hypothetical-test features missing"):
            validate_feature_availability(kept_columns, ["business_category_electronics_store"])

    def test_get_feature_direction_positive(self):
        """Test that getting the feature direction returns 1 for positive coefficients."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.5],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), 1)

    def test_get_feature_direction_negative(self):
        """Test that getting the feature direction returns -1 for negative coefficients."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [-0.5],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), -1)

    def test_get_feature_direction_zero(self):
        """Test that getting the feature direction returns 0 for zero coefficients."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.0],
            }
        )
        self.assertEqual(get_feature_direction(coef_summary, "x1"), 0)

    def test_get_feature_direction_missing(self):
        """Test that getting the feature direction returns None for missing features."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coef": [0.5],
            }
        )
        self.assertIsNone(get_feature_direction(coef_summary, "missing"))

    def test_compare_profile_to_baseline(self):
        """Test comparing a given profile's hazard to the baseline hazard."""
        results = pd.DataFrame(
            {
                "partial_hazard": [1.0, 2.0, 0.5],
            },
            index=["baseline", "riskier", "safer"],
        )

        self.assertEqual(compare_profile_to_baseline(results, "riskier"), "higher_risk")
        self.assertEqual(compare_profile_to_baseline(results, "safer"), "lower_risk")

    def test_score_profiles_returns_expected_columns(self):
        """Test that scoring profiles returns a dataframe with the expected columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_cox_standard_artifacts_dir(tmp_path)
            artifacts = load_artifacts(tmp_path)
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
        """Test that checking directional expectations returns expected results structure."""
        res_dict = {
            "baseline": 1.0,
            "electronics_store": 2.0,
            "electronic_cigarette_dealer": 2.5,
            "bingo_game_operator": 0.5,
            "multi_license_business": 0.4
        }
        results = pd.DataFrame(
            {"partial_hazard": list(res_dict.values())},
            index=list(res_dict.keys())
        )

        coef_data = [
            ("business_category_electronics_store", 1.0),
            ("business_category_electronic_cigarette_dealer", 1.0),
            ("business_category_bingo_game_operator", -1.0),
            ("active_license_count", -1.0)
        ]
        coef_summary = pd.DataFrame(coef_data, columns=["feature", "coef"])

        checks = check_directional_expectations(results, coef_summary)

        self.assertFalse(checks.empty)

        expected_cols = {
            "profile", "feature", "expected_relation",
            "actual_relation", "matches_expectation"
        }
        self.assertEqual(set(checks.columns), expected_cols)
        self.assertEqual(len(checks), 4)
        self.assertTrue(bool(checks["matches_expectation"].all()))


if __name__ == "__main__":
    unittest.main()
