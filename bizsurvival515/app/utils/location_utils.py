"""Provide shared location and cluster helper utilities for NYC-based model inputs.

This module centralizes reusable helpers for working with business locations
and spatial cluster centroids in the NYC Business Survival app. It defines
shared geographic bounds for the NYC study area, provides utilities for
clamping coordinates to those bounds, extracts cluster-centroid reference
dataframes from model reference datasets, and assigns the nearest cluster
information to a given latitude/longitude pair.

Constants:
- NYC_LAT_MIN:
  Minimum latitude allowed within the NYC study area.
- NYC_LAT_MAX:
  Maximum latitude allowed within the NYC study area.
- NYC_LNG_MIN:
  Minimum longitude allowed within the NYC study area.
- NYC_LNG_MAX:
  Maximum longitude allowed within the NYC study area.
- DEFAULT_SURVIVAL_MONTHS:
  Default survival horizons used by app views and analyses.

Functions:
- clamp_to_nyc_bounds:
  Clamp a latitude/longitude pair to the NYC study-area bounds.
- build_cluster_reference_df:
  Extract a standardized cluster-centroid reference dataframe.
- assign_nearest_cluster_info:
  Find the nearest cluster centroid to a latitude/longitude pair.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68

DEFAULT_SURVIVAL_MONTHS = [12, 36, 60, 120]


def clamp_to_nyc_bounds(latitude: float, longitude: float) -> tuple[float, float]:
    """Clamp a latitude/longitude pair to the NYC study-area bounds.

    Args:
        latitude: Input latitude value.
        longitude: Input longitude value.

    Returns:
        A tuple containing latitude and longitude values clipped to the
        configured NYC bounds.
    """
    lat = min(max(latitude, NYC_LAT_MIN), NYC_LAT_MAX)
    lng = min(max(longitude, NYC_LNG_MIN), NYC_LNG_MAX)
    return lat, lng


def build_cluster_reference_df(
    reference_df: pd.DataFrame,
    lat_column: str,
    lng_column: str,
    cluster_column: str = "location_cluster",
) -> pd.DataFrame:
    """Extract a standardized cluster-centroid reference dataframe.

    Args:
        reference_df: Source dataframe containing cluster and coordinate columns.
        lat_column: Name of the latitude column in the source dataframe.
        lng_column: Name of the longitude column in the source dataframe.
        cluster_column: Name of the cluster-ID column in the source dataframe.

    Returns:
        A dataframe with standardized columns ``location_cluster``,
        ``location_cluster_lat``, and ``location_cluster_lng``, containing
        unique non-null cluster-centroid rows. Returns an empty dataframe if
        the required latitude and longitude columns are unavailable.
    """
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
    """Find the nearest cluster centroid to a latitude/longitude pair.

    Args:
        latitude: Input latitude value.
        longitude: Input longitude value.
        cluster_df: Standardized cluster-centroid dataframe containing
            location-cluster coordinates.

    Returns:
        A tuple containing:
        - The nearest cluster ID, or ``None`` if no cluster data is available.
        - The latitude of the nearest cluster centroid.
        - The longitude of the nearest cluster centroid.

        If ``cluster_df`` is empty, the function returns ``None`` together with
        the original latitude and longitude values.
    """
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
