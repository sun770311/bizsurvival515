"""
Utilities for constructing model input feature vectors used by the
business survival prediction models.

This module provides helper functions to build single-row feature
profiles for hypothetical businesses based on user inputs. These
profiles match the feature schema used during model training so that
they can be safely passed into trained prediction pipelines.

Supported functionality includes:
- Generating zero-initialized feature profiles
- Building logistic regression feature vectors for 3-year survival prediction
- Mapping display names to encoded feature columns
- Assigning nearest geographic cluster centroids

These utilities are primarily used by the Streamlit simulation pages
to convert user inputs (categories, complaint counts, location, etc.)
into model-ready feature matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .location_utils import (
    assign_nearest_cluster_info,
    build_cluster_reference_df,
)


LOGISTIC_LAT_COL = "business_latitude_first12m_first"
LOGISTIC_LNG_COL = "business_longitude_first12m_first"
LOGISTIC_CLUSTER_COL = "location_cluster_first12m_first"
LOGISTIC_CLUSTER_LAT_COL = "location_cluster_lat_first12m_first"
LOGISTIC_CLUSTER_LNG_COL = "location_cluster_lng_first12m_first"
LOGISTIC_LICENSE_COL = "active_license_count_first12m_mean"


@dataclass
class BusinessProfileInputs:
    """User-provided inputs for constructing a hypothetical business profile."""

    selected_category_columns: list[str]
    active_license_count: int
    business_latitude: float
    business_longitude: float
    complaint_counts: dict[str, float] = field(default_factory=dict)


def prettify_feature_name(column_name: str) -> str:
    """Convert a raw feature column name into a user-friendly display label."""
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
    """Return kept business-category indicator columns."""
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("business_category_")
            and column.endswith("_first12m_max")
        ]
    )


def complaint_feature_columns(kept_columns: list[str]) -> list[str]:
    """Return kept complaint-count feature columns."""
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("complaint_type_")
            and column.endswith("_first12m_sum")
        ]
    )


def category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map pretty category labels to raw category feature column names."""
    category_cols = category_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in category_cols}


def complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map pretty complaint labels to raw complaint feature column names."""
    complaint_cols = complaint_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    """Create a single-row profile initialized to zeros."""
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def assign_nearest_cluster_centroid(
    latitude: float,
    longitude: float,
    reference_df: pd.DataFrame,
) -> tuple[float, float]:
    """Return only the nearest cluster centroid latitude and longitude."""
    cluster_df = build_cluster_reference_df(
        reference_df=reference_df,
        lat_column="location_cluster_lat",
        lng_column="location_cluster_lng",
        cluster_column="location_cluster",
    )
    _cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        latitude,
        longitude,
        cluster_df,
    )
    return cluster_lat, cluster_lng


def _rename_logistic_cluster_columns(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Rename logistic-specific cluster columns to shared cluster field names."""
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
    """Build a default logistic profile using median location and one active license."""
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

    cluster_ref_df = build_cluster_reference_df(
        reference_df=_rename_logistic_cluster_columns(reference_df),
        lat_column="location_cluster_lat",
        lng_column="location_cluster_lng",
        cluster_column="location_cluster",
    )
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


def _populate_profile(
    profile: pd.DataFrame,
    numeric_inputs: dict[str, float],
    selected_category_columns: list[str],
    complaint_counts: dict[str, float],
) -> pd.DataFrame:
    """Fill numeric, category, and complaint values into a single-row profile."""
    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = float(value)

    for category_column in selected_category_columns:
        if category_column in profile.columns:
            profile.loc[0, category_column] = 1.0

    for complaint_column, count in complaint_counts.items():
        if complaint_column in profile.columns:
            profile.loc[0, complaint_column] = float(count)

    return profile


def build_logistic_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: BusinessProfileInputs,
) -> pd.DataFrame:
    """Build a single-row profile for the logistic regression model."""
    profile = build_zero_profile(kept_columns)

    cluster_ref_df = build_cluster_reference_df(
        reference_df=_rename_logistic_cluster_columns(reference_df),
        lat_column="location_cluster_lat",
        lng_column="location_cluster_lng",
        cluster_column="location_cluster",
    )
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        inputs.business_latitude,
        inputs.business_longitude,
        cluster_ref_df,
    )

    numeric_inputs = {
        LOGISTIC_LICENSE_COL: float(inputs.active_license_count),
        LOGISTIC_LAT_COL: float(inputs.business_latitude),
        LOGISTIC_LNG_COL: float(inputs.business_longitude),
        LOGISTIC_CLUSTER_COL: float(cluster_id) if cluster_id is not None else 0.0,
        LOGISTIC_CLUSTER_LAT_COL: float(cluster_lat),
        LOGISTIC_CLUSTER_LNG_COL: float(cluster_lng),
    }

    return _populate_profile(
        profile=profile,
        numeric_inputs=numeric_inputs,
        selected_category_columns=inputs.selected_category_columns,
        complaint_counts=inputs.complaint_counts,
    )
