"""Unit tests for the Cox survival modeling pipeline."""

import unittest

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from bizsurvival515.pipeline.cox import (
    CoxConfig,
    FeatureSelectionResult,
    build_business_level_dataset,
    build_coefficient_summary,
    build_time_varying_panel,
    fit_standard_cox_model,
    get_feature_columns,
    get_model_drop_columns,
    load_joined_dataset,
    prepare_business_level_model_data,
    prepare_time_varying_model_data,
    run_standard_cox_pipeline,
    scale_features,
    select_nonconstant_features,
    validate_joined_dataset,
)
from bizsurvival515.tests.pipeline_test_helpers import fitted_cox_output_dir
from bizsurvival515.tests.test_helpers import (
    TEST_DATA_DIR,
    make_duplicate_business_month_row,
)


class TestCox(unittest.TestCase):
    """Test suite for Cox pipeline helpers and standard model training."""

    def test_validate_joined_dataset_accepts_valid_data(self):
        """Accept a valid joined dataset without raising validation errors."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined)

    def test_validate_joined_dataset_rejects_duplicate_rows(self):
        """Reject joined datasets with duplicate business_id-month rows."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        duplicated = make_duplicate_business_month_row(joined)

        with self.assertRaisesRegex(ValueError, "Duplicate business_id-month rows"):
            validate_joined_dataset(duplicated)

    def test_get_model_drop_columns_contains_expected_columns(self):
        """Return expected non-feature columns that should be dropped."""
        dropped = get_model_drop_columns()
        self.assertIn("open", dropped)
        self.assertIn("business_category_sum", dropped)
        self.assertIn("total_311", dropped)

    def test_build_time_varying_panel_has_required_columns(self):
        """Build a time-varying panel with interval and event columns."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        panel = build_time_varying_panel(joined, pd.Timestamp("2026-03-01"))

        self.assertFalse(panel.empty)
        self.assertIn("business_id", panel.columns)
        self.assertIn("start", panel.columns)
        self.assertIn("stop", panel.columns)
        self.assertIn("event", panel.columns)
        self.assertTrue((panel["stop"] >= panel["start"]).all())
        self.assertTrue(set(panel["event"].unique()).issubset({0, 1}))

    def test_prepare_time_varying_model_data_returns_expected_parts(self):
        """Prepare time-varying model data with selected nonconstant features."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        panel = build_time_varying_panel(joined, pd.Timestamp("2026-03-01"))

        prepared = prepare_time_varying_model_data(
            panel=panel,
            variance_threshold=1e-8,
        )

        self.assertFalse(prepared.modeling_df.empty)
        self.assertIn("event", prepared.modeling_df.columns)
        self.assertIn("start", prepared.modeling_df.columns)
        self.assertIn("stop", prepared.modeling_df.columns)
        self.assertGreater(len(prepared.feature_selection.kept_columns), 0)

    def test_build_business_level_dataset_has_required_columns(self):
        """Build the standard Cox business-level dataset with survival labels."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        coxph_df = build_business_level_dataset(joined, pd.Timestamp("2026-03-01"))

        self.assertFalse(coxph_df.empty)
        self.assertIn("business_id", coxph_df.columns)
        self.assertIn("duration_months", coxph_df.columns)
        self.assertIn("event", coxph_df.columns)
        self.assertTrue(set(coxph_df["event"].unique()).issubset({0, 1}))

    def test_get_feature_columns_excludes_requested_columns(self):
        """Exclude identifier and outcome columns from the feature list."""
        df = pd.DataFrame(
            {
                "business_id": ["A", "B"],
                "duration_months": [12, 24],
                "event": [0, 1],
                "x1": [1.0, 2.0],
                "x2": [3.0, 4.0],
            }
        )
        feature_columns = get_feature_columns(
            df,
            ["business_id", "duration_months", "event"],
        )
        self.assertEqual(feature_columns, ["x1", "x2"])

    def test_select_nonconstant_features_drops_constant_columns(self):
        """Drop constant features and retain informative numeric columns."""
        df = pd.DataFrame(
            {
                "signal": [0.0, 1.0, 0.0, 1.0],
                "constant": [5.0, 5.0, 5.0, 5.0],
            }
        )

        selection = select_nonconstant_features(
            df=df,
            feature_columns=["signal", "constant"],
            variance_threshold=1e-8,
        )

        self.assertIsInstance(selection, FeatureSelectionResult)
        self.assertEqual(selection.kept_columns, ["signal"])
        self.assertIn("constant", selection.dropped_columns)

    def test_scale_features_returns_scaled_df_and_scaler(self):
        """Scale selected feature columns and return the fitted scaler."""
        df = pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0],
                "x2": [4.0, 5.0, 6.0],
            }
        )

        scaled_df, scaler = scale_features(df, ["x1", "x2"])

        self.assertEqual(list(scaled_df.columns), ["x1", "x2"])
        self.assertEqual(scaled_df.shape, (3, 2))
        self.assertIsInstance(scaler, StandardScaler)

    def test_prepare_business_level_model_data_returns_expected_parts(self):
        """Prepare standard Cox model data with retained feature columns."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        coxph_df = build_business_level_dataset(joined, pd.Timestamp("2026-03-01"))

        prepared = prepare_business_level_model_data(
            coxph_df=coxph_df,
            variance_threshold=1e-8,
        )

        self.assertFalse(prepared.modeling_df.empty)
        self.assertIn("duration_months", prepared.modeling_df.columns)
        self.assertIn("event", prepared.modeling_df.columns)
        self.assertGreater(len(prepared.feature_selection.kept_columns), 0)

    def test_fit_standard_cox_model_returns_fitter(self):
        """Fit the standard Cox model and return a CoxPHFitter instance."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        coxph_df = build_business_level_dataset(joined, pd.Timestamp("2026-03-01"))
        prepared = prepare_business_level_model_data(
            coxph_df,
            variance_threshold=1e-8,
        )

        model = fit_standard_cox_model(
            modeling_df=prepared.modeling_df,
            penalizer=0.1,
        )

        self.assertIsInstance(model, CoxPHFitter)
        self.assertTrue(hasattr(model, "summary"))

    def test_build_coefficient_summary_returns_sorted_df(self):
        """Build a coefficient summary dataframe from a fitted Cox model."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        coxph_df = build_business_level_dataset(joined, pd.Timestamp("2026-03-01"))
        prepared = prepare_business_level_model_data(
            coxph_df,
            variance_threshold=1e-8,
        )
        model = fit_standard_cox_model(
            prepared.modeling_df,
            penalizer=0.1,
        )

        summary = build_coefficient_summary(model)

        self.assertFalse(summary.empty)
        self.assertIn("feature", summary.columns)
        self.assertIn("coef", summary.columns)
        self.assertIn("abs_coef", summary.columns)

    def test_run_standard_cox_pipeline_writes_artifacts(self):
        """Run the standard Cox pipeline and write expected output artifacts."""
        with fitted_cox_output_dir() as tmp_path:
            results = run_standard_cox_pipeline(
                CoxConfig(
                    data_path=TEST_DATA_DIR / "joined_dataset.csv",
                    output_dir=tmp_path,
                )
            )

            self.assertIn("artifact_paths", results)
            expected_files = [
                "coxph_model.pkl",
                "coxph_scaler.pkl",
                "coxph_kept_columns.pkl",
                "coxph_dropped_columns.pkl",
                "coxph_summary.csv",
            ]
            for filename in expected_files:
                self.assertTrue((tmp_path / filename).exists())


if __name__ == "__main__":
    unittest.main()
