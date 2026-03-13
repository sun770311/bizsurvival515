"""
Shared assertion helpers for unit tests.
"""

from __future__ import annotations

import pandas as pd


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
