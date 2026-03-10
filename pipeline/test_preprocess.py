"""Validation tests for the preprocessing pipeline output.

This script loads joined_dataset.csv and verifies that the final output
from preprocess.py satisfies key structural and data-quality requirements.

Input:
- joined_dataset.csv

Checks:
- Required columns exist
- Month values parse correctly
- No duplicate business_id-month rows
- Key numeric fields have valid ranges
- Latitude and longitude values are valid
- Aggregated sums match component columns
- Human-readable feature columns are used instead of numbered lookup columns
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PreprocessTestConfig:
    """Configuration for preprocess output validation."""

    joined_data_path: Path


def load_joined_dataset(joined_data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse the month column."""
    joined = pd.read_csv(joined_data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def get_required_columns() -> list[str]:
    """Return required output columns for joined_dataset.csv."""
    return [
        "business_id",
        "month",
        "active_license_count",
        "total_311",
        "open",
        "months_since_first_license",
        "location_cluster",
        "location_cluster_lat",
        "location_cluster_lng",
        "business_latitude",
        "business_longitude",
        "business_category_sum",
        "complaint_sum",
    ]


def assert_required_columns_present(joined: pd.DataFrame) -> None:
    """Assert that all required columns exist."""
    required_columns = get_required_columns()
    missing_required = [column for column in required_columns if column not in joined.columns]
    if missing_required:
        raise AssertionError(f"Missing required columns: {missing_required}")


def assert_month_parses(joined: pd.DataFrame) -> None:
    """Assert all month values parsed successfully."""
    if not joined["month"].notna().all():
        raise AssertionError("Some month values could not be parsed.")


def assert_no_duplicate_business_month_rows(joined: pd.DataFrame) -> None:
    """Assert there are no duplicate business_id-month rows."""
    if joined.duplicated(["business_id", "month"]).any():
        raise AssertionError("Duplicate business_id-month rows found.")


def assert_valid_core_ranges(joined: pd.DataFrame) -> None:
    """Assert core numeric columns have valid values."""
    if not (joined["active_license_count"] >= 1).all():
        raise AssertionError("active_license_count has values < 1.")

    if not joined["open"].isin([0, 1]).all():
        raise AssertionError("open contains values other than 0/1.")

    if not (joined["months_since_first_license"] >= 0).all():
        raise AssertionError("months_since_first_license has negative values.")

    if not (joined["location_cluster"] >= 0).all():
        raise AssertionError("location_cluster has negative values.")


def assert_valid_coordinates(joined: pd.DataFrame) -> None:
    """Assert latitude and longitude values fall within valid ranges."""
    if not joined["location_cluster_lat"].between(-90, 90).all():
        raise AssertionError("Invalid location_cluster_lat values.")

    if not joined["location_cluster_lng"].between(-180, 180).all():
        raise AssertionError("Invalid location_cluster_lng values.")

    if not joined["business_latitude"].between(-90, 90).all():
        raise AssertionError("Invalid business_latitude values.")

    if not joined["business_longitude"].between(-180, 180).all():
        raise AssertionError("Invalid business_longitude values.")


def assert_aggregate_sums_match(joined: pd.DataFrame) -> None:
    """Assert aggregate summary columns match their intended totals."""
    if not (joined["business_category_sum"] == joined["active_license_count"]).all():
        raise AssertionError(
            "business_category_sum does not match active_license_count for all rows."
        )

    if not (joined["complaint_sum"] == joined["total_311"]).all():
        raise AssertionError(
            "complaint_sum does not match total_311 for all rows."
        )


def find_numbered_lookup_style_columns(joined: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Find legacy numbered lookup-style feature columns."""
    bad_lookup_style_category_cols = [
        column
        for column in joined.columns
        if column.startswith("business_category_")
        and column[len("business_category_"):].isdigit()
    ]
    bad_lookup_style_complaint_cols = [
        column
        for column in joined.columns
        if column.startswith("complaint_type_")
        and column[len("complaint_type_"):].isdigit()
    ]
    return bad_lookup_style_category_cols, bad_lookup_style_complaint_cols


def assert_human_readable_feature_columns(joined: pd.DataFrame) -> None:
    """Assert feature columns are human-readable instead of lookup-indexed."""
    (
        bad_lookup_style_category_cols,
        bad_lookup_style_complaint_cols,
    ) = find_numbered_lookup_style_columns(joined)

    if bad_lookup_style_category_cols:
        raise AssertionError(
            "Found numbered business category columns: "
            f"{bad_lookup_style_category_cols[:10]}"
        )

    if bad_lookup_style_complaint_cols:
        raise AssertionError(
            "Found numbered complaint type columns: "
            f"{bad_lookup_style_complaint_cols[:10]}"
        )


def run_preprocess_tests(config: PreprocessTestConfig) -> None:
    """Run all validation checks for preprocess.py output."""
    joined = load_joined_dataset(config.joined_data_path)

    assert_required_columns_present(joined)
    assert_month_parses(joined)
    assert_no_duplicate_business_month_rows(joined)
    assert_valid_core_ranges(joined)
    assert_valid_coordinates(joined)
    assert_aggregate_sums_match(joined)
    assert_human_readable_feature_columns(joined)


def main() -> None:
    """Execute preprocess output validation."""
    config = PreprocessTestConfig(
        joined_data_path=Path("/content/drive/MyDrive/joined_dataset.csv"),
    )
    run_preprocess_tests(config)
    print("All preprocess output checks passed.")


if __name__ == "__main__":
    main()