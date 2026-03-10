"""Tests for the cox survival modeling pipeline module."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

from pipeline.cox import (
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


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestCox(unittest.TestCase):
    """Test suite for Cox survival modeling pipeline functions."""

    def test_load_joined_dataset_parses_month_column(self):
        """Test that loading the joined dataset parses the month column as datetime."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        self.assertIn("month", joined.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["month"]))

    def test_validate_joined_dataset_accepts_valid_data(self):
        """Test that a valid dataset passes validation without errors."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined)

    def test_validate_joined_dataset_rejects_duplicate_rows(self):
        """Test that validation fails when duplicate business_id-month rows exist."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        duplicated = pd.concat([joined, joined.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate business_id-month rows"):
            validate_joined_dataset(duplicated)

    def test_get_model_drop_columns_contains_expected_columns(self):
        """Test that the drop columns list contains expected leakage columns."""
        dropped = get_model_drop_columns()
        self.assertIn("open", dropped)
        self.assertIn("business_category_sum", dropped)
        self.assertIn("total_311", dropped)

    def test_build_time_varying_panel_has_required_columns(self):
        """Test that the time-varying panel includes start, stop, and event columns."""
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
        """Test that preparing time-varying data returns a correctly shaped dataframe."""
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
        """Test that the business-level dataset includes duration and event columns."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        coxph_df = build_business_level_dataset(joined, pd.Timestamp("2026-03-01"))

        self.assertFalse(coxph_df.empty)
        self.assertIn("business_id", coxph_df.columns)
        self.assertIn("duration_months", coxph_df.columns)
        self.assertIn("event", coxph_df.columns)
        self.assertTrue(set(coxph_df["event"].unique()).issubset({0, 1}))

    def test_get_feature_columns_excludes_requested_columns(self):
        """Test that feature column extraction properly excludes specified columns."""
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
        """Test that constant features are dropped by variance thresholding."""
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
        """Test that scaling returns a normalized dataframe and the fitted scaler."""
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
        """Test that preparing business-level data returns scaled features."""
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
        """Test that fitting a standard Cox model returns a CoxPHFitter instance."""
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
        """Test that building a coefficient summary returns a sorted dataframe."""
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
        """Test that running the standard Cox pipeline writes expected artifact files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            config = CoxConfig(
                data_path=TEST_DATA_DIR / "joined_dataset.csv",
                output_dir=tmp_path,
            )

            results = run_standard_cox_pipeline(config)

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
