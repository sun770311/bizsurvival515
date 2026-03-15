"""Unit tests for the logistic regression training pipeline."""

from pathlib import Path
import unittest

import pandas as pd

from bizsurvival515.pipeline.logistic import (
    FeatureSelectionResult,
    LogisticConfig,
    aggregate_first_year_features,
    balance_dataset,
    build_business_survival_summary,
    build_coefficient_summary,
    build_training_dataset,
    choose_aggregation,
    evaluate_model,
    filter_eligible_businesses,
    fit_logistic_model,
    get_excluded_feature_columns,
    get_first_year_window,
    is_binary_series,
    load_joined_dataset,
    prepare_training_data,
    restrict_to_study_window,
    run_logistic_pipeline,
    select_nonconstant_features,
    split_features_and_target,
    train_test_split_balanced,
    validate_joined_dataset,
)
from bizsurvival515.tests.pipeline_test_helpers import fitted_logistic_output_dir
from bizsurvival515.tests.test_helpers import TEST_DATA_DIR, assert_month_column_parsed


class TestLogistic(unittest.TestCase):
    """Test suite for logistic pipeline helpers and end-to-end training."""

    def test_load_and_validate_joined_dataset(self):
        """Load the joined dataset, verify month parsing, and validate it."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        assert_month_column_parsed(self, joined)
        validate_joined_dataset(joined)

    def test_restrict_to_study_window_filters_future_rows(self):
        """Filter out rows whose month falls after the study end."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        self.assertTrue((restricted["month"] <= pd.Timestamp("2026-03-01")).all())

    def test_build_business_survival_summary_has_target_column(self):
        """Build a business-level survival summary with the expected target."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        summary = build_business_survival_summary(restricted, survival_months=36)

        self.assertIn("business_id", summary.columns)
        self.assertIn("duration_months", summary.columns)
        self.assertIn("survived_36m", summary.columns)
        self.assertTrue(set(summary["survived_36m"].unique()).issubset({0, 1}))

    def test_filter_eligible_businesses_respects_cutoff_and_window(self):
        """Keep only businesses eligible for first-window aggregation and outcome labeling."""
        business_survival = pd.DataFrame(
            {
                "business_id": ["A", "B", "C"],
                "first_month": pd.to_datetime(
                    ["2020-01-01", "2024-01-01", "2022-01-01"]
                ),
                "last_month": pd.to_datetime(
                    ["2025-01-01", "2026-01-01", "2022-06-01"]
                ),
                "duration_months": [60, 24, 5],
                "survived_36m": [1, 0, 0],
            }
        )

        eligible = filter_eligible_businesses(
            business_survival=business_survival,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
            aggregation_months=12,
        )

        self.assertEqual(eligible["business_id"].tolist(), ["A"])

    def test_get_first_year_window_keeps_first_twelve_rows_per_business(self):
        """Keep at most the first aggregation window of rows per business."""
        df = pd.DataFrame(
            {
                "business_id": ["A"] * 14 + ["B"] * 8,
                "month": pd.date_range("2020-01-01", periods=14, freq="MS").tolist()
                + pd.date_range("2021-01-01", periods=8, freq="MS").tolist(),
                "value": range(22),
            }
        )

        first_year = get_first_year_window(df, aggregation_months=12)
        counts = first_year.groupby("business_id").size().to_dict()

        self.assertEqual(counts["A"], 12)
        self.assertEqual(counts["B"], 8)

    def test_is_binary_series_detects_binary_values(self):
        """Identify whether a series contains only binary indicator values."""
        self.assertTrue(is_binary_series(pd.Series([0, 1, 1, 0, None])))
        self.assertFalse(is_binary_series(pd.Series([0, 1, 2, None])))

    def test_choose_aggregation_returns_expected_rules(self):
        """Choose aggregation rules based on feature naming and values."""
        self.assertEqual(
            choose_aggregation(
                "business_category_test",
                pd.Series([0, 1, 0]),
            ),
            "max",
        )
        self.assertEqual(
            choose_aggregation(
                "complaint_type_noise",
                pd.Series([0, 1, 2]),
            ),
            "sum",
        )
        self.assertEqual(
            choose_aggregation(
                "active_license_count",
                pd.Series([1, 2, 3]),
            ),
            "mean",
        )
        self.assertEqual(
            choose_aggregation(
                "business_latitude",
                pd.Series([40.1, 40.1, 40.1]),
            ),
            "first",
        )

    def test_aggregate_first_year_features_returns_one_row_per_business(self):
        """Aggregate first-window features down to one row per business."""
        df = pd.DataFrame(
            {
                "business_id": ["A", "A", "A", "B", "B"],
                "month": pd.to_datetime(
                    [
                        "2020-01-01",
                        "2020-02-01",
                        "2020-03-01",
                        "2021-01-01",
                        "2021-02-01",
                    ]
                ),
                "active_license_count": [1, 2, 3, 2, 4],
                "business_category_store": [1, 1, 1, 0, 1],
                "complaint_type_noise": [0, 1, 2, 0, 1],
                "business_latitude": [40.0, 40.0, 40.0, 41.0, 41.0],
                "open": [1, 1, 0, 1, 1],
                "months_since_first_license": [1, 2, 3, 1, 2],
            }
        )

        aggregated = aggregate_first_year_features(df, aggregation_months=12)

        self.assertEqual(aggregated["business_id"].nunique(), 2)
        self.assertEqual(len(aggregated), 2)
        self.assertIn("active_license_count_first12m_mean", aggregated.columns)
        self.assertIn("business_category_store_first12m_max", aggregated.columns)
        self.assertIn("complaint_type_noise_first12m_sum", aggregated.columns)
        self.assertIn("business_latitude_first12m_first", aggregated.columns)
        self.assertIn("observed_months_in_first_window", aggregated.columns)

    def test_build_training_dataset_returns_target(self):
        """Build the model training dataset with target and aggregation features."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))

        training_df = build_training_dataset(
            restricted,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
            aggregation_months=12,
        )

        self.assertFalse(training_df.empty)
        self.assertIn("survived_36m", training_df.columns)
        self.assertIn("observed_months_in_first_window", training_df.columns)

    def test_get_excluded_feature_columns_contains_leakage_columns(self):
        """Return known leakage and non-feature columns for exclusion."""
        excluded = get_excluded_feature_columns()
        self.assertIn("survived_36m", excluded)
        self.assertIn("duration_months", excluded)
        self.assertIn("open_first12m_max", excluded)

    def test_split_features_and_target_returns_valid_shapes(self):
        """Split the training dataframe into aligned feature and target outputs."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        training_df = build_training_dataset(
            restricted,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
            aggregation_months=12,
        )

        features, target = split_features_and_target(training_df)
        self.assertEqual(len(features), len(target))
        self.assertNotIn("survived_36m", features.columns)

    def test_select_nonconstant_features_drops_constant_columns(self):
        """Drop constant features and report them in the selection summary."""
        features = pd.DataFrame(
            {
                "signal": [0, 1, 0, 1],
                "constant": [5, 5, 5, 5],
            }
        )

        reduced, selection = select_nonconstant_features(
            features,
            variance_threshold=1e-8,
        )

        self.assertIn("signal", reduced.columns)
        self.assertNotIn("constant", reduced.columns)
        self.assertIsInstance(selection, FeatureSelectionResult)
        self.assertIn("constant", selection.dropped_columns)

    def test_balance_dataset_balances_classes(self):
        """Downsample or resample classes so both targets have equal counts."""
        features = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6]})
        target = pd.Series([1, 1, 1, 1, 0, 0])

        balanced = balance_dataset(features, target, random_state=42)
        counts = balanced["survived_36m"].value_counts()

        self.assertEqual(counts.nunique(), 1)

    def test_train_test_split_balanced_preserves_target_column_split(self):
        """Split a balanced dataframe into train/test feature and target sets."""
        balanced_df = pd.DataFrame(
            {
                "x1": range(10),
                "x2": range(10, 20),
                "survived_36m": [0, 1] * 5,
            }
        )

        x_train, x_test, y_train, y_test = train_test_split_balanced(
            balanced_df=balanced_df,
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(len(x_train) + len(x_test), len(balanced_df))
        self.assertEqual(len(y_train) + len(y_test), len(balanced_df))

    def test_prepare_training_data_returns_expected_parts(self):
        """Prepare end-to-end training artifacts and balanced splits."""
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )

        prepared = prepare_training_data(config)

        self.assertFalse(prepared.training_df.empty)
        self.assertFalse(prepared.X.empty)
        self.assertGreater(len(prepared.X_train), 0)
        self.assertGreater(len(prepared.X_test), 0)
        self.assertTrue(set(prepared.y.unique()).issubset({0, 1}))

    def test_fit_logistic_model_returns_pipeline(self):
        """Fit the logistic model pipeline and expose the trained model step."""
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)

        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.settings.max_iter,
            random_state=config.settings.random_state,
        )

        self.assertIn("model", pipeline.named_steps)

    def test_evaluate_model_returns_core_metrics(self):
        """Evaluate the trained model and return standard classification metrics."""
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)
        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.settings.max_iter,
            random_state=config.settings.random_state,
        )

        metrics = evaluate_model(
            pipeline=pipeline,
            X_test=prepared.X_test,
            y_test=prepared.y_test,
        )

        self.assertIn("accuracy", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)
        self.assertGreaterEqual(metrics["roc_auc"], 0.0)
        self.assertLessEqual(metrics["roc_auc"], 1.0)

    def test_build_coefficient_summary_matches_feature_count(self):
        """Build a coefficient table aligned with the retained feature columns."""
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)
        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.settings.max_iter,
            random_state=config.settings.random_state,
        )

        coef_df = build_coefficient_summary(
            pipeline=pipeline,
            feature_columns=prepared.X_train.columns.tolist(),
        )

        self.assertEqual(len(coef_df), prepared.X_train.shape[1])
        self.assertIn("feature", coef_df.columns)
        self.assertIn("coefficient", coef_df.columns)

    def test_run_logistic_pipeline_writes_artifacts(self):
        """Run the logistic pipeline end to end and write expected artifacts."""
        with fitted_logistic_output_dir() as tmp_path:
            summary = run_logistic_pipeline(
                LogisticConfig(
                    data_path=TEST_DATA_DIR / "joined_dataset.csv",
                    output_dir=tmp_path,
                )
            )

            self.assertIn("metrics", summary)
            self.assertIn("artifact_paths", summary)
            self.assertIn("aggregation_months", summary)
            self.assertEqual(summary["aggregation_months"], 12)

            expected_files = [
                "logistic_pipeline.pkl",
                "logistic_kept_columns.pkl",
                "logistic_dropped_columns.pkl",
                "logistic_coefficient_summary.csv",
                "business_survival_balanced_dataset.csv",
                "X_train_balanced_split.csv",
                "X_test_balanced_split.csv",
                "logistic_evaluation_metrics.json",
            ]

            for filename in expected_files:
                self.assertTrue((tmp_path / filename).exists())


if __name__ == "__main__":
    unittest.main()
