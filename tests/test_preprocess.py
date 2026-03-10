"""Tests for the preprocessing pipeline module."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.preprocess import (
    PipelineConfig,
    build_joined_dataset,
    clean_licenses,
    clean_service_requests,
    make_unique_column_names,
    month_range,
    run_pipeline,
    sanitize_feature_name,
)


TEST_DATA_DIR = Path(__file__).parent / "data"


class TestPreprocess(unittest.TestCase):
    """Test suite for the preprocessing pipeline functions."""

    def test_sanitize_feature_name_basic(self):
        """Test that sanitize_feature_name formats basic strings correctly."""
        self.assertEqual(
            sanitize_feature_name("Home Improvement Contractor", "business_category"),
            "business_category_home_improvement_contractor",
        )

    def test_sanitize_feature_name_symbols(self):
        """Test that sanitize_feature_name replaces special characters appropriately."""
        self.assertEqual(
            sanitize_feature_name("Heat & Hot Water", "complaint_type"),
            "complaint_type_heat_and_hot_water",
        )

    def test_make_unique_column_names(self):
        """Test that make_unique_column_names appends numerical suffixes to duplicates."""
        names = ["a", "a", "b", "a"]
        self.assertEqual(make_unique_column_names(names), ["a", "a_1", "b", "a_2"])

    def test_month_range_inclusive(self):
        """Test that month_range returns an inclusive sequence of month-start dates."""
        result = month_range(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-01"))
        expected = pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"])
        pd.testing.assert_index_equal(result, expected)

    def test_clean_licenses_parses_dates_and_filters_invalid_rows(self):
        """Test that clean_licenses parses dates and filters out incomplete records."""
        raw = pd.DataFrame(
            {
                "License Number": ["1", "2"],
                "Business Unique ID": ["B1", "B2"],
                "Business Category": ["Food", None],
                "License Type": ["TypeA", "TypeB"],
                "License Status": ["Active", "Active"],
                "Initial Issuance Date": ["01/01/2025", "02/01/2025"],
                "Expiration Date": ["03/01/2025", "01/01/2025"],
                "Latitude": ["40.7", "40.8"],
                "Longitude": ["-73.9", "-73.8"],
            }
        )

        cleaned = clean_licenses(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Business Unique ID"], "B1")

    def test_clean_service_requests_parses_dates_and_filters_invalid_rows(self):
        """Test that clean_service_requests drops records with bad dates or values."""
        raw = pd.DataFrame(
            {
                "Unique Key": ["100", "101"],
                "Created Date": ["01/15/2025 01:00:00 PM", "bad-date"],
                "Problem (formerly Complaint Type)": ["Noise", ""],
                "Latitude": ["40.7", "40.8"],
                "Longitude": ["-73.9", "-73.8"],
            }
        )

        cleaned = clean_service_requests(raw)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Unique Key"], "100")
        self.assertEqual(cleaned.iloc[0]["month"], pd.Timestamp("2025-01-01"))

    def test_build_joined_dataset_from_sample_data(self):
        """Test the end-to-end dataset builder against a sample subset of data."""
        config = PipelineConfig(
            licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
            service_reqs_path=TEST_DATA_DIR / "service_reqs_sample.csv",
            output_path=TEST_DATA_DIR / "tmp_joined_dataset.csv",
            location_k=5,
        )

        joined = build_joined_dataset(config)

        self.assertFalse(joined.empty)
        self.assertIn("business_id", joined.columns)
        self.assertIn("month", joined.columns)
        self.assertIn("active_license_count", joined.columns)
        self.assertIn("open", joined.columns)
        self.assertIn("months_since_first_license", joined.columns)
        self.assertIn("location_cluster", joined.columns)
        self.assertIn("business_category_sum", joined.columns)
        self.assertIn("complaint_sum", joined.columns)

        self.assertTrue((joined["open"] == 1).all())
        self.assertTrue((joined["months_since_first_license"] >= 0).all())
        self.assertIn(joined["location_cluster"].dtype.kind, {"i", "u"})

    def test_business_category_sum_matches_category_columns(self):
        """Test that the category sum column aligns with individual category flags."""
        config = PipelineConfig(
            licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
            service_reqs_path=TEST_DATA_DIR / "service_reqs_sample.csv",
            output_path=TEST_DATA_DIR / "tmp_joined_dataset.csv",
            location_k=5,
        )

        joined = build_joined_dataset(config)

        category_cols = [
            c for c in joined.columns
            if c.startswith("business_category_") and c != "business_category_sum"
        ]
        if category_cols:
            expected_sum = joined[category_cols].sum(axis=1)
            pd.testing.assert_series_equal(
                joined["business_category_sum"],
                expected_sum,
                check_names=False,
            )

    def test_complaint_sum_matches_complaint_columns(self):
        """Test that the complaint sum column aligns with individual complaint flags."""
        config = PipelineConfig(
            licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
            service_reqs_path=TEST_DATA_DIR / "service_reqs_sample.csv",
            output_path=TEST_DATA_DIR / "tmp_joined_dataset.csv",
            location_k=5,
        )

        joined = build_joined_dataset(config)

        complaint_cols = [
            c for c in joined.columns
            if c.startswith("complaint_type_") and c != "complaint_sum"
        ]
        if complaint_cols:
            expected_sum = joined[complaint_cols].sum(axis=1)
            pd.testing.assert_series_equal(
                joined["complaint_sum"],
                expected_sum,
                check_names=False,
            )

    def test_run_pipeline_writes_output(self):
        """Test that the full pipeline writes the resulting dataset to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "joined_dataset.csv"

            config = PipelineConfig(
                licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
                service_reqs_path=TEST_DATA_DIR / "service_reqs_sample.csv",
                output_path=output_path,
                location_k=5,
            )

            returned_path = run_pipeline(config)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())

            written = pd.read_csv(output_path)
            self.assertFalse(written.empty)


if __name__ == "__main__":
    unittest.main()
