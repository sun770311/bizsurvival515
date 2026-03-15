"""
Shared assertion helpers for unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TEST_DATA_DIR = Path(__file__).parent / "data"


def assert_nearest_cluster_case(
    test_case,
    assign_fn,
    cluster_df: pd.DataFrame,
) -> None:
    """Assert a standard nearest-cluster lookup case."""
    cluster_id, cluster_lat, cluster_lng = assign_fn(
        40.705,
        -74.005,
        cluster_df,
    )
    test_case.assertIsNotNone(cluster_id)
    test_case.assertAlmostEqual(cluster_lat, 40.70, places=2)
    test_case.assertAlmostEqual(cluster_lng, -74.00, places=2)


def assert_clamp_case(
    test_case,
    clamp_fn,
    lat_max: float,
    lng_min: float,
) -> None:
    """Assert that out-of-bounds coordinates are clamped correctly."""
    lat, lng = clamp_fn(50.0, -80.0)
    test_case.assertEqual(lat, lat_max)
    test_case.assertEqual(lng, lng_min)


def assert_month_column_parsed(
    test_case,
    df: pd.DataFrame,
    column_name: str = "month",
) -> None:
    """Assert that a dataframe contains a parsed datetime month column."""
    test_case.assertIn(column_name, df.columns)
    test_case.assertTrue(pd.api.types.is_datetime64_any_dtype(df[column_name]))


def make_duplicate_business_month_row(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with one duplicated first row appended."""
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)
