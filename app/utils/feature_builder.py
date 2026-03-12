from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68

DEFAULT_SURVIVAL_MONTHS = [12, 36, 60, 120]

LOGISTIC_LAT_COL = "business_latitude_first12m_first"
LOGISTIC_LNG_COL = "business_longitude_first12m_first"
LOGISTIC_CLUSTER_COL = "location_cluster_first12m_first"
LOGISTIC_CLUSTER_LAT_COL = "location_cluster_lat_first12m_first"
LOGISTIC_CLUSTER_LNG_COL = "location_cluster_lng_first12m_first"
LOGISTIC_LICENSE_COL = "active_license_count_first12m_mean"


def prettify_feature_name(column_name: str) -> str:
    suffixes = [
        "_first12m_max",
        "_first12m_mean",
        "_first12m_sum",
        "_first12m_first",
        "_first12m_last",
    ]

    cleaned = column_name
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break

    for prefix in ("business_category_", "complaint_type_"):
        if cleaned.startswith(prefix):
            raw = cleaned[len(prefix):]
            return raw.replace("_", " ").title()

    return cleaned.replace("_", " ").title()


def category_feature_columns(kept_columns: list[str]) -> list[str]:
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("business_category_")
            and column.endswith("_first12m_max")
        ]
    )


def complaint_feature_columns(kept_columns: list[str]) -> list[str]:
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("complaint_type_")
            and column.endswith("_first12m_sum")
        ]
    )


def category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    category_cols = category_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in category_cols}


def complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    complaint_cols = complaint_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def _cluster_reference_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    required = {"location_cluster_lat", "location_cluster_lng"}
    if not required.issubset(reference_df.columns):
        return pd.DataFrame()

    cluster_cols = [
        column
        for column in [
            "location_cluster",
            "location_cluster_lat",
            "location_cluster_lng",
        ]
        if column in reference_df.columns
    ]

    return (
        reference_df[cluster_cols]
        .dropna(subset=["location_cluster_lat", "location_cluster_lng"])
        .drop_duplicates()
        .reset_index(drop=True)
    )


def assign_nearest_cluster_info(
    latitude: float,
    longitude: float,
    reference_df: pd.DataFrame,
) -> tuple[float | None, float, float]:
    cluster_df = _cluster_reference_df(reference_df)

    if cluster_df.empty:
        return None, float(latitude), float(longitude)

    cluster_coords = cluster_df[["location_cluster_lat", "location_cluster_lng"]].to_numpy(
        dtype=float
    )
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


def assign_nearest_cluster_centroid(
    latitude: float,
    longitude: float,
    reference_df: pd.DataFrame,
) -> tuple[float, float]:
    _cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        latitude,
        longitude,
        reference_df,
    )
    return cluster_lat, cluster_lng


def _rename_logistic_cluster_columns(reference_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        LOGISTIC_CLUSTER_COL: "location_cluster",
        LOGISTIC_CLUSTER_LAT_COL: "location_cluster_lat",
        LOGISTIC_CLUSTER_LNG_COL: "location_cluster_lng",
    }
    existing_map = {
        old_name: new_name
        for old_name, new_name in rename_map.items()
        if old_name in reference_df.columns
    }
    return reference_df.rename(columns=existing_map).copy()


def baseline_new_business_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_zero_profile(kept_columns)

    median_lat = (
        float(reference_df[LOGISTIC_LAT_COL].median())
        if LOGISTIC_LAT_COL in reference_df.columns
        else 40.7128
    )
    median_lng = (
        float(reference_df[LOGISTIC_LNG_COL].median())
        if LOGISTIC_LNG_COL in reference_df.columns
        else -74.0060
    )

    cluster_ref_df = _rename_logistic_cluster_columns(reference_df)
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        median_lat,
        median_lng,
        cluster_ref_df,
    )

    if LOGISTIC_LICENSE_COL in profile.columns:
        profile.loc[0, LOGISTIC_LICENSE_COL] = 1.0
    if LOGISTIC_LAT_COL in profile.columns:
        profile.loc[0, LOGISTIC_LAT_COL] = median_lat
    if LOGISTIC_LNG_COL in profile.columns:
        profile.loc[0, LOGISTIC_LNG_COL] = median_lng
    if LOGISTIC_CLUSTER_COL in profile.columns and cluster_id is not None:
        profile.loc[0, LOGISTIC_CLUSTER_COL] = cluster_id
    if LOGISTIC_CLUSTER_LAT_COL in profile.columns:
        profile.loc[0, LOGISTIC_CLUSTER_LAT_COL] = cluster_lat
    if LOGISTIC_CLUSTER_LNG_COL in profile.columns:
        profile.loc[0, LOGISTIC_CLUSTER_LNG_COL] = cluster_lng

    return profile


def build_standard_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    selected_category_columns: list[str],
    active_license_count: int,
    business_latitude: float,
    business_longitude: float,
    complaint_counts: dict[str, float] | None = None,
) -> pd.DataFrame:
    profile = build_zero_profile(kept_columns)

    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        business_latitude,
        business_longitude,
        reference_df,
    )

    numeric_inputs: dict[str, Any] = {
        "active_license_count": float(active_license_count),
        "business_latitude": float(business_latitude),
        "business_longitude": float(business_longitude),
        "location_cluster": float(cluster_id) if cluster_id is not None else 0.0,
        "location_cluster_lat": float(cluster_lat),
        "location_cluster_lng": float(cluster_lng),
    }

    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = float(value)

    for category_column in selected_category_columns:
        if category_column in profile.columns:
            profile.loc[0, category_column] = 1.0

    for complaint_column, count in (complaint_counts or {}).items():
        if complaint_column in profile.columns:
            profile.loc[0, complaint_column] = float(count)

    return profile


def build_logistic_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    selected_category_columns: list[str],
    active_license_count: int,
    business_latitude: float,
    business_longitude: float,
    complaint_counts: dict[str, float] | None = None,
) -> pd.DataFrame:
    profile = build_zero_profile(kept_columns)

    cluster_ref_df = _rename_logistic_cluster_columns(reference_df)
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        business_latitude,
        business_longitude,
        cluster_ref_df,
    )

    numeric_inputs: dict[str, Any] = {
        LOGISTIC_LICENSE_COL: float(active_license_count),
        LOGISTIC_LAT_COL: float(business_latitude),
        LOGISTIC_LNG_COL: float(business_longitude),
        LOGISTIC_CLUSTER_COL: float(cluster_id) if cluster_id is not None else 0.0,
        LOGISTIC_CLUSTER_LAT_COL: float(cluster_lat),
        LOGISTIC_CLUSTER_LNG_COL: float(cluster_lng),
    }

    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = float(value)

    for category_column in selected_category_columns:
        if category_column in profile.columns:
            profile.loc[0, category_column] = 1.0

    for complaint_column, count in (complaint_counts or {}).items():
        if complaint_column in profile.columns:
            profile.loc[0, complaint_column] = float(count)

    return profile


def clamp_to_nyc_bounds(latitude: float, longitude: float) -> tuple[float, float]:
    lat = min(max(latitude, NYC_LAT_MIN), NYC_LAT_MAX)
    lng = min(max(longitude, NYC_LNG_MIN), NYC_LNG_MAX)
    return lat, lng