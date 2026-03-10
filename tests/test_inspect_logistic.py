"""Tests for the inspect_logistic module."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.inspect_logistic import (
    InspectConfig,
    build_baseline_profile,
    build_hypothetical_profiles,
    check_hypothetical_expectations,
    get_coefficient_direction,
    load_artifacts,
    predict_profiles,
)
from pipeline.logistic import LogisticConfig, ModelingParams, run_logistic_pipeline


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestInspectLogistic(unittest.TestCase):
    """Test suite for inspecting logistic regression models with hypothetical profiles."""

    def create_logistic_artifacts_dir(self, tmp_path: Path) -> None:
        """Create a temporary directory and generate logistic model artifacts for testing."""
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=tmp_path,
            params=ModelingParams(),
        )
        run_logistic_pipeline(config)

    def test_load_artifacts(self):
        """Test that logistic model artifacts are successfully loaded from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_logistic_artifacts_dir(tmp_path)
            config = InspectConfig(artifacts_dir=tmp_path)

            pipeline, kept_columns, metrics, coef_summary = load_artifacts(config)

            self.assertIsNotNone(pipeline)
            self.assertIsInstance(kept_columns, list)
            self.assertGreater(len(kept_columns), 0)
            self.assertIsInstance(metrics, dict)
            self.assertIn("accuracy", metrics)
            self.assertIsInstance(coef_summary, pd.DataFrame)
            self.assertFalse(coef_summary.empty)
            self.assertIn("feature", coef_summary.columns)
            self.assertIn("coefficient", coef_summary.columns)

    def test_build_baseline_profile(self):
        """Test that a baseline profile of zeros is correctly built for kept columns."""
        kept_columns = ["a", "b", "c"]
        baseline = build_baseline_profile(kept_columns)

        self.assertEqual(baseline.shape, (1, 3))
        self.assertEqual(baseline.columns.tolist(), kept_columns)
        self.assertTrue((baseline.iloc[0] == 0).all())

    def test_build_hypothetical_profiles_contains_baseline(self):
        """Test that the generated hypothetical profiles contain the expected profiles."""
        kept_columns = [
            "active_license_count",
            "business_category_electronics_store",
            "business_category_electronic_cigarette_dealer",
            "business_category_bingo_game_operator",
            "business_category_laundries",
            "business_category_car_wash",
            "business_category_debt_collection_agency",
        ]

        profiles = build_hypothetical_profiles(kept_columns)

        expected_profiles = {
            "baseline",
            "electronics_store",
            "vape_shop",
            "bingo_operator",
            "many_licenses",
            "laundries",
            "car_wash",
            "debt_collection",
        }

        self.assertEqual(set(profiles.index), expected_profiles)
        self.assertEqual(profiles.shape[1], len(kept_columns))

    def test_build_hypothetical_profiles_sets_expected_feature_values(self):
        """Test that hypothetical profiles have the correct feature values set."""
        kept_columns = [
            "active_license_count",
            "business_category_electronics_store",
            "business_category_laundries",
        ]

        profiles = build_hypothetical_profiles(kept_columns)

        self.assertEqual(profiles.loc["baseline", "active_license_count"], 0)
        self.assertEqual(
            profiles.loc["electronics_store", "business_category_electronics_store"], 1
        )
        self.assertEqual(profiles.loc["many_licenses", "active_license_count"], 5)
        self.assertEqual(profiles.loc["laundries", "business_category_laundries"], 1)

    def test_predict_profiles_returns_valid_output(self):
        """Test that profile prediction returns a dataframe with valid probabilities."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_logistic_artifacts_dir(tmp_path)
            config = InspectConfig(artifacts_dir=tmp_path)
            pipeline, kept_columns, _metrics, _coef_summary = load_artifacts(config)

            profiles = build_hypothetical_profiles(kept_columns)
            results = predict_profiles(pipeline, profiles)

            self.assertFalse(results.empty)
            self.assertEqual(
                set(results.columns),
                {
                    "profile",
                    "predicted_survival_probability",
                    "predicted_class",
                },
            )
            self.assertEqual(results["profile"].nunique(), len(profiles.index))
            self.assertTrue(results["predicted_survival_probability"].between(0, 1).all())
            self.assertTrue(set(results["predicted_class"].unique()).issubset({0, 1}))

    def test_get_coefficient_direction_positive(self):
        """Test that positive coefficients return 'above_baseline'."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [0.75],
            }
        )
        self.assertEqual(get_coefficient_direction(coef_summary, "x1"), "above_baseline")

    def test_get_coefficient_direction_negative(self):
        """Test that negative coefficients return 'below_baseline'."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [-0.75],
            }
        )
        self.assertEqual(get_coefficient_direction(coef_summary, "x1"), "below_baseline")

    def test_get_coefficient_direction_zero(self):
        """Test that zero coefficients return 'same_as_baseline'."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [0.0],
            }
        )
        self.assertEqual(get_coefficient_direction(coef_summary, "x1"), "same_as_baseline")

    def test_get_coefficient_direction_missing(self):
        """Test that missing features return 'feature_missing'."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [0.5],
            }
        )
        self.assertEqual(
            get_coefficient_direction(coef_summary, "missing_feature"),
            "feature_missing",
        )

    def test_check_hypothetical_expectations_returns_expected_columns(self):
        """Test that hypothetical expectations checking returns the expected structure."""
        results = pd.DataFrame(
            {
                "profile": [
                    "baseline",
                    "electronics_store",
                    "vape_shop",
                    "bingo_operator",
                    "many_licenses",
                    "laundries",
                    "car_wash",
                    "debt_collection",
                ],
                "predicted_survival_probability": [
                    0.50, 0.60, 0.40, 0.55, 0.70, 0.45, 0.52, 0.48
                ],
                "predicted_class": [1, 1, 0, 1, 1, 0, 1, 0],
            }
        )

        coef_summary = pd.DataFrame(
            {
                "feature": [
                    "business_category_electronics_store",
                    "business_category_electronic_cigarette_dealer",
                    "business_category_bingo_game_operator",
                    "active_license_count",
                    "business_category_laundries",
                    "business_category_car_wash",
                    "business_category_debt_collection_agency",
                ],
                "coefficient": [1.0, -1.0, 0.5, 0.8, -0.3, 0.2, -0.4],
            }
        )

        expectation_results = check_hypothetical_expectations(results, coef_summary)

        self.assertFalse(expectation_results.empty)
        self.assertEqual(
            set(expectation_results.columns),
            {
                "profile",
                "feature",
                "expected_vs_baseline",
                "actual_vs_baseline",
                "matches_expectation",
            },
        )
        self.assertEqual(len(expectation_results), 7)

    def test_check_hypothetical_expectations_with_real_artifacts(self):
        """Test checking expectations using real artifacts generated from the pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            self.create_logistic_artifacts_dir(tmp_path)
            config = InspectConfig(artifacts_dir=tmp_path)
            pipeline, kept_columns, _metrics, coef_summary = load_artifacts(config)

            profiles = build_hypothetical_profiles(kept_columns)
            results = predict_profiles(pipeline, profiles)
            expectation_results = check_hypothetical_expectations(results, coef_summary)

            self.assertFalse(expectation_results.empty)
            self.assertIn("matches_expectation", expectation_results.columns)
            self.assertEqual(expectation_results["matches_expectation"].dtype, bool)
            self.assertEqual(len(expectation_results), 7)


if __name__ == "__main__":
    unittest.main()
