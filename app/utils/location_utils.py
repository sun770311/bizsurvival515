"""Shared location and cluster helper utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68

DEFAULT_SURVIVAL_MONTHS = [12, 36, 60, 120]


def clamp_to_nyc_bounds(latitude: float, longitude: float) -> tuple[float, float]:
    """Clamp a latitude/longitude pair to the NYC study area bounds."""
    lat = min(max(latitude, NYC_LAT_MIN), NYC_LAT_MAX)
    lng = min(max(longitude, NYC_LNG_MIN), NYC_LNG_MAX)
    return lat, lng


def build_cluster_reference_df(
    reference_df: pd.DataFrame,
    lat_column: str,
    lng_column: str,
    cluster_column: str = "location_cluster",
) -> pd.DataFrame:
    """Extract unique cluster centroid rows from a reference dataframe."""
    required = {lat_column, lng_column}
    if not required.issubset(reference_df.columns):
        return pd.DataFrame()

    cluster_cols = [
        column
        for column in [cluster_column, lat_column, lng_column]
        if column in reference_df.columns
    ]

    cluster_df = reference_df[cluster_cols].rename(
        columns={
            cluster_column: "location_cluster",
            lat_column: "location_cluster_lat",
            lng_column: "location_cluster_lng",
        }
    )

    return (
        cluster_df
        .dropna(subset=["location_cluster_lat", "location_cluster_lng"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def assign_nearest_cluster_info(
    latitude: float,
    longitude: float,
    cluster_df: pd.DataFrame,
) -> tuple[float | None, float, float]:
    """Find the nearest cluster centroid to a latitude/longitude pair."""
    if cluster_df.empty:
        return None, float(latitude), float(longitude)

    cluster_coords = cluster_df[
        ["location_cluster_lat", "location_cluster_lng"]
    ].to_numpy(dtype=float)
    point = np.array([latitude, longitude], dtype=float)

    distances = np.sqrt(((cluster_coords - point) ** 2).sum(axis=1))
    nearest_idx = int(np.argmin(distances))

    cluster_id: float | None = None
    if "location_cluster" in cluster_df.columns:
        raw_cluster = cluster_df.loc[nearest_idx, "location_cluster"]
        if pd.notna(raw_cluster):
            cluster_id = float(raw_cluster)

    return (
        cluster_id,
        float(cluster_df.loc[nearest_idx, "location_cluster_lat"]),
        float(cluster_df.loc[nearest_idx, "location_cluster_lng"]),
    )
