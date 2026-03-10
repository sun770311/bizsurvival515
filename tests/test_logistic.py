from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.logistic import (
    FeatureSelectionResult,
    LogisticConfig,
    balance_dataset,
    build_business_survival_summary,
    build_coefficient_summary,
    build_training_dataset,
    evaluate_model,
    filter_eligible_businesses,
    fit_logistic_model,
    get_excluded_feature_columns,
    load_joined_dataset,
    prepare_training_data,
    restrict_to_study_window,
    run_logistic_pipeline,
    select_nonconstant_features,
    split_features_and_target,
    train_test_split_balanced,
    validate_joined_dataset,
)


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestLogistic(unittest.TestCase):
    def test_load_joined_dataset_parses_month_column(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        self.assertIn("month", joined.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["month"]))

    def test_validate_joined_dataset_accepts_valid_data(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined)

    def test_validate_joined_dataset_rejects_duplicate_rows(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        duplicated = pd.concat([joined, joined.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate business_id-month rows"):
            validate_joined_dataset(duplicated)

    def test_restrict_to_study_window_filters_future_rows(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        self.assertTrue((restricted["month"] <= pd.Timestamp("2026-03-01")).all())

    def test_build_business_survival_summary_has_target_column(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        summary = build_business_survival_summary(restricted, survival_months=36)

        self.assertIn("business_id", summary.columns)
        self.assertIn("duration_months", summary.columns)
        self.assertIn("survived_36m", summary.columns)
        self.assertTrue(set(summary["survived_36m"].unique()).issubset({0, 1}))

    def test_filter_eligible_businesses_respects_cutoff(self):
        business_survival = pd.DataFrame(
            {
                "business_id": ["A", "B"],
                "first_month": pd.to_datetime(["2020-01-01", "2024-01-01"]),
                "last_month": pd.to_datetime(["2025-01-01", "2026-01-01"]),
                "duration_months": [60, 24],
                "survived_36m": [1, 0],
            }
        )

        eligible = filter_eligible_businesses(
            business_survival=business_survival,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
        )

        self.assertEqual(eligible["business_id"].tolist(), ["A"])

    def test_build_training_dataset_returns_target(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))

        training_df = build_training_dataset(
            restricted,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
        )

        self.assertFalse(training_df.empty)
        self.assertIn("survived_36m", training_df.columns)

    def test_get_excluded_feature_columns_contains_leakage_columns(self):
        excluded = get_excluded_feature_columns()
        self.assertIn("survived_36m", excluded)
        self.assertIn("duration_months", excluded)
        self.assertIn("open", excluded)

    def test_split_features_and_target_returns_valid_shapes(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        restricted = restrict_to_study_window(joined, pd.Timestamp("2026-03-01"))
        training_df = build_training_dataset(
            restricted,
            study_end=pd.Timestamp("2026-03-01"),
            survival_months=36,
        )

        X, y = split_features_and_target(training_df)
        self.assertEqual(len(X), len(y))
        self.assertNotIn("survived_36m", X.columns)

    def test_select_nonconstant_features_drops_constant_columns(self):
        X = pd.DataFrame(
            {
                "signal": [0, 1, 0, 1],
                "constant": [5, 5, 5, 5],
            }
        )

        reduced, selection = select_nonconstant_features(X, variance_threshold=1e-8)

        self.assertIn("signal", reduced.columns)
        self.assertNotIn("constant", reduced.columns)
        self.assertIsInstance(selection, FeatureSelectionResult)
        self.assertIn("constant", selection.dropped_columns)

    def test_balance_dataset_balances_classes(self):
        X = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6]})
        y = pd.Series([1, 1, 1, 1, 0, 0])

        balanced = balance_dataset(X, y, random_state=42)
        counts = balanced["survived_36m"].value_counts()

        self.assertEqual(counts.nunique(), 1)

    def test_train_test_split_balanced_preserves_target_column_split(self):
        balanced_df = pd.DataFrame(
            {
                "x1": range(10),
                "x2": range(10, 20),
                "survived_36m": [0, 1] * 5,
            }
        )

        X_train, X_test, y_train, y_test = train_test_split_balanced(
            balanced_df=balanced_df,
            test_size=0.2,
            random_state=42,
        )

        self.assertEqual(len(X_train) + len(X_test), len(balanced_df))
        self.assertEqual(len(y_train) + len(y_test), len(balanced_df))

    def test_prepare_training_data_returns_expected_parts(self):
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
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)

        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )

        self.assertIn("model", pipeline.named_steps)

    def test_evaluate_model_returns_core_metrics(self):
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)
        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.max_iter,
            random_state=config.random_state,
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
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=Path("unused"),
        )
        prepared = prepare_training_data(config)
        pipeline = fit_logistic_model(
            X_train=prepared.X_train,
            y_train=prepared.y_train,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )

        coef_df = build_coefficient_summary(
            pipeline=pipeline,
            feature_columns=prepared.X_train.columns.tolist(),
        )

        self.assertEqual(len(coef_df), prepared.X_train.shape[1])
        self.assertIn("feature", coef_df.columns)
        self.assertIn("coefficient", coef_df.columns)

    def test_run_logistic_pipeline_writes_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            config = LogisticConfig(
                data_path=TEST_DATA_DIR / "joined_dataset.csv",
                output_dir=tmp_path,
            )

            summary = run_logistic_pipeline(config)

            self.assertIn("metrics", summary)
            self.assertIn("artifact_paths", summary)

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