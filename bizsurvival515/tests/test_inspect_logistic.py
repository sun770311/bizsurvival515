"""Unit tests for logistic model inspection helpers."""

import unittest

import pandas as pd

from bizsurvival515.pipeline.inspect_logistic import (
    InspectConfig,
    build_baseline_profile,
    build_hypothetical_profiles,
    check_hypothetical_expectations,
    get_coefficient_direction,
    load_artifacts,
    predict_profiles,
)
from bizsurvival515.tests.pipeline_test_helpers import fitted_logistic_output_dir


class TestInspectLogistic(unittest.TestCase):
    """Test suite for inspecting trained logistic model artifacts."""

    def test_load_artifacts(self):
        """Load saved logistic artifacts and verify core object types."""
        with fitted_logistic_output_dir() as artifacts_dir:
            config = InspectConfig(artifacts_dir=artifacts_dir)

            pipeline, kept_columns, metrics, coef_summary, train_split = load_artifacts(
                config
            )

        self.assertIsNotNone(pipeline)
        self.assertIsInstance(kept_columns, list)
        self.assertGreater(len(kept_columns), 0)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIsInstance(coef_summary, pd.DataFrame)
        self.assertFalse(coef_summary.empty)
        self.assertIn("feature", coef_summary.columns)
        self.assertIn("coefficient", coef_summary.columns)
        self.assertIsInstance(train_split, pd.DataFrame)
        self.assertFalse(train_split.empty)

    def test_build_baseline_profile(self):
        """Build a one-row baseline profile from training medians."""
        kept_columns = ["a", "b", "c"]
        train_split = pd.DataFrame(
            {
                "a": [1.0, 3.0, 5.0],
                "b": [2.0, 4.0, 6.0],
                "c": [10.0, 20.0, 30.0],
            }
        )

        baseline = build_baseline_profile(kept_columns, train_split)

        self.assertEqual(baseline.shape, (1, 3))
        self.assertEqual(baseline.columns.tolist(), kept_columns)
        self.assertEqual(baseline.index.tolist(), ["baseline"])
        self.assertEqual(baseline.loc["baseline", "a"], 3.0)
        self.assertEqual(baseline.loc["baseline", "b"], 4.0)
        self.assertEqual(baseline.loc["baseline", "c"], 20.0)

    def test_build_hypothetical_profiles_contains_baseline(self):
        """Build the expected set of named hypothetical profiles."""
        kept_columns = [
            "active_license_count_first12m_mean",
            "business_category_electronics_store_first12m_max",
            "business_category_electronic_cigarette_dealer_first12m_max",
            "business_category_bingo_game_operator_first12m_max",
            "business_category_industrial_laundry_first12m_max",
            "business_category_debt_collection_agency_first12m_max",
            "business_category_home_improvement_contractor_first12m_max",
        ]

        train_split = pd.DataFrame([{column: 0.0 for column in kept_columns}])

        profiles = build_hypothetical_profiles(kept_columns, train_split)

        expected_profiles = {
            "baseline",
            "electronics_store",
            "vape_shop",
            "bingo_operator",
            "many_licenses",
            "industrial_laundry",
            "debt_collection",
            "home_improvement_contractor",
        }

        self.assertEqual(set(profiles.index), expected_profiles)
        self.assertEqual(profiles.shape[1], len(kept_columns))

    def test_build_hypothetical_profiles_sets_expected_feature_values(self):
        """Set activated features to the expected hypothetical profile values."""
        kept_columns = [
            "active_license_count_first12m_mean",
            "business_category_electronics_store_first12m_max",
            "business_category_industrial_laundry_first12m_max",
            "business_category_home_improvement_contractor_first12m_max",
        ]

        train_split = pd.DataFrame(
            {
                "active_license_count_first12m_mean": [2.0, 2.0, 2.0],
                "business_category_electronics_store_first12m_max": [0.0, 0.0, 0.0],
                "business_category_industrial_laundry_first12m_max": [0.0, 0.0, 0.0],
                "business_category_home_improvement_contractor_first12m_max": [
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

        profiles = build_hypothetical_profiles(kept_columns, train_split)

        self.assertEqual(
            profiles.loc["baseline", "active_license_count_first12m_mean"],
            2.0,
        )
        self.assertEqual(
            profiles.loc[
                "electronics_store",
                "business_category_electronics_store_first12m_max",
            ],
            1.0,
        )
        self.assertEqual(
            profiles.loc["many_licenses", "active_license_count_first12m_mean"],
            5.0,
        )
        self.assertEqual(
            profiles.loc[
                "industrial_laundry",
                "business_category_industrial_laundry_first12m_max",
            ],
            1.0,
        )
        self.assertEqual(
            profiles.loc[
                "home_improvement_contractor",
                "business_category_home_improvement_contractor_first12m_max",
            ],
            1.0,
        )

    def test_predict_profiles_returns_valid_output(self):
        """Predict classes and probabilities for hypothetical profiles."""
        with fitted_logistic_output_dir() as artifacts_dir:
            config = InspectConfig(artifacts_dir=artifacts_dir)
            pipeline, kept_columns, _metrics, _coef_summary, train_split = (
                load_artifacts(config)
            )

            profiles = build_hypothetical_profiles(kept_columns, train_split)
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
        """Return above-baseline when the coefficient is positive."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [0.75],
            }
        )
        self.assertEqual(get_coefficient_direction(coef_summary, "x1"), "above_baseline")

    def test_get_coefficient_direction_negative(self):
        """Return below-baseline when the coefficient is negative."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [-0.75],
            }
        )
        self.assertEqual(get_coefficient_direction(coef_summary, "x1"), "below_baseline")

    def test_get_coefficient_direction_zero(self):
        """Return same-as-baseline when the coefficient is zero."""
        coef_summary = pd.DataFrame(
            {
                "feature": ["x1"],
                "coefficient": [0.0],
            }
        )
        self.assertEqual(
            get_coefficient_direction(coef_summary, "x1"),
            "same_as_baseline",
        )

    def test_get_coefficient_direction_missing(self):
        """Return feature-missing when the requested feature is absent."""
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
        """Compare hypothetical profiles against coefficient-based expectations."""
        results = pd.DataFrame(
            {
                "profile": [
                    "baseline",
                    "electronics_store",
                    "vape_shop",
                    "bingo_operator",
                    "many_licenses",
                    "industrial_laundry",
                    "debt_collection",
                    "home_improvement_contractor",
                ],
                "predicted_survival_probability": [
                    0.50,
                    0.40,
                    0.65,
                    0.60,
                    0.70,
                    0.58,
                    0.45,
                    0.68,
                ],
                "predicted_class": [1, 0, 1, 1, 1, 1, 0, 1],
            }
        )

        coef_summary = pd.DataFrame(
            {
                "feature": [
                    "business_category_electronics_store_first12m_max",
                    "business_category_electronic_cigarette_dealer_first12m_max",
                    "business_category_bingo_game_operator_first12m_max",
                    "active_license_count_first12m_mean",
                    "business_category_industrial_laundry_first12m_max",
                    "business_category_debt_collection_agency_first12m_max",
                    "business_category_home_improvement_contractor_first12m_max",
                ],
                "coefficient": [-1.0, 1.0, 0.5, 0.8, 0.2, -0.4, 0.7],
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

    def test_check_hypothetical_expectations_with_feature_missing(self):
        """Mark expectation checks as missing when a feature is absent."""
        results = pd.DataFrame(
            {
                "profile": [
                    "baseline",
                    "electronics_store",
                    "vape_shop",
                    "bingo_operator",
                    "many_licenses",
                    "industrial_laundry",
                    "debt_collection",
                    "home_improvement_contractor",
                ],
                "predicted_survival_probability": [
                    0.50,
                    0.40,
                    0.65,
                    0.60,
                    0.70,
                    0.58,
                    0.45,
                    0.68,
                ],
                "predicted_class": [1, 0, 1, 1, 1, 1, 0, 1],
            }
        )

        coef_summary = pd.DataFrame(
            {
                "feature": [
                    "business_category_electronics_store_first12m_max",
                    "business_category_electronic_cigarette_dealer_first12m_max",
                    "business_category_bingo_game_operator_first12m_max",
                    "active_license_count_first12m_mean",
                    "business_category_debt_collection_agency_first12m_max",
                    "business_category_home_improvement_contractor_first12m_max",
                ],
                "coefficient": [-1.0, 1.0, 0.5, 0.8, -0.4, 0.7],
            }
        )

        expectation_results = check_hypothetical_expectations(results, coef_summary)

        industrial_row = expectation_results.loc[
            expectation_results["profile"] == "industrial_laundry"
        ].iloc[0]

        self.assertEqual(industrial_row["expected_vs_baseline"], "feature_missing")
        self.assertEqual(industrial_row["actual_vs_baseline"], "not_checked")
        self.assertFalse(bool(industrial_row["matches_expectation"]))

    def test_check_hypothetical_expectations_with_real_artifacts(self):
        """Run expectation checks end to end using real trained artifacts."""
        with fitted_logistic_output_dir() as artifacts_dir:
            config = InspectConfig(artifacts_dir=artifacts_dir)
            pipeline, kept_columns, _metrics, coef_summary, train_split = (
                load_artifacts(config)
            )

            profiles = build_hypothetical_profiles(kept_columns, train_split)
            results = predict_profiles(pipeline, profiles)
            expectation_results = check_hypothetical_expectations(
                results,
                coef_summary,
            )

        self.assertFalse(expectation_results.empty)
        self.assertIn("matches_expectation", expectation_results.columns)
        self.assertEqual(len(expectation_results), 7)
        self.assertTrue(
            expectation_results["matches_expectation"].isin([True, False]).all()
        )


if __name__ == "__main__":
    unittest.main()
