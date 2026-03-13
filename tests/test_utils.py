"""Tests for shared pipeline utilities."""

import unittest
from pathlib import Path

import pandas as pd

from pipeline.utils import (
    load_joined_dataset,
    validate_joined_dataset,
    get_model_drop_columns,
)

TEST_DATA_DIR = Path(__file__).parent / "data"


class TestUtils(unittest.TestCase):
    """Test suite for shared pipeline utilities."""

    def test_load_joined_dataset_parses_month_column(self):
        """Test that loading the joined dataset correctly parses the month column."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        self.assertIn("month", joined.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["month"]))

    def test_validate_joined_dataset_accepts_valid_data(self):
        """Test that a valid dataset passes validation without errors."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined, ["business_id", "month"])

    def test_validate_joined_dataset_rejects_duplicate_rows(self):
        """Test that validation fails when duplicate business_id-month rows exist."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        duplicated = pd.concat([joined, joined.iloc[[0]]], ignore_index=True)

        with self.assertRaisesRegex(ValueError, "Duplicate business_id-month rows"):
            validate_joined_dataset(duplicated, ["business_id", "month"])

    def test_get_model_drop_columns_contains_expected_columns(self):
        """Test that the drop columns list contains expected leakage columns."""
        dropped = get_model_drop_columns()
        self.assertIn("open", dropped)
        self.assertIn("business_category_sum", dropped)
        self.assertIn("total_311", dropped)


if __name__ == "__main__":
    unittest.main()
