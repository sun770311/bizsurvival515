"""Build logistic-model input profiles for business survival prediction.

Inputs:
- Kept logistic model feature-column lists
- Reference dataframes used for median defaults and cluster lookup
- User-provided profile inputs for category, complaint, license-count, and
  location settings

Processing steps:
- Identify retained category and complaint feature columns
- Convert raw feature-column names into display-friendly labels
- Build zero-valued profiles aligned to retained feature columns
- Derive baseline location and cluster defaults from reference data
- Assign nearest geographic cluster information for input coordinates
- Fill numeric, category, and complaint values into logistic model profiles

Outputs:
- Baseline logistic profile dataframes
- User-specified logistic profile dataframes
- Display-label mappings for retained category and complaint features

Classes:
- BusinessProfileInputs:
  Stores user-provided inputs for constructing a hypothetical business profile.

Functions:
- prettify_feature_name:
  Convert a raw feature-column name into a display-friendly label.
- category_feature_columns:
  Return retained business-category feature columns.
- complaint_feature_columns:
  Return retained complaint feature columns.
- category_display_to_column_map:
  Map display labels to raw business-category feature columns.
- complaint_display_to_column_map:
  Map display labels to raw complaint feature columns.
- build_zero_profile:
  Build a zero-valued profile aligned to retained feature columns.
- assign_nearest_cluster_centroid:
  Return the nearest cluster-centroid coordinates for a given location.
- _rename_logistic_cluster_columns:
  Rename logistic-specific cluster columns to shared cluster field names.
- baseline_new_business_profile:
  Build a baseline logistic profile using reference medians and one active license.
- _populate_profile:
  Fill numeric, category, and complaint values into a single-row profile.
- build_logistic_profile:
  Build a user-specified single-row profile for the logistic regression model.
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
    """Store user-provided inputs for constructing a hypothetical business profile.

    Attributes:
        selected_category_columns: Category feature columns set to active.
        active_license_count: Active license count for the business profile.
        business_latitude: Business latitude used for location-based features.
        business_longitude: Business longitude used for location-based features.
        complaint_counts: Mapping from complaint feature columns to complaint counts.
    """

    selected_category_columns: list[str]
    active_license_count: int
    business_latitude: float
    business_longitude: float
    complaint_counts: dict[str, float] = field(default_factory=dict)


def prettify_feature_name(column_name: str) -> str:
    """Convert a raw feature column name into a display-friendly label.

    Args:
        column_name: Raw encoded feature column name.

    Returns:
        A human-readable label with training-specific suffixes and standard
        prefixes removed where applicable.
    """
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
    """Return retained business-category indicator columns for the logistic model.

    Args:
        kept_columns: Feature columns retained by the logistic model.

    Returns:
        A sorted list of business-category feature columns ending in
        ``_first12m_max``.
    """
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("business_category_")
            and column.endswith("_first12m_max")
        ]
    )


def complaint_feature_columns(kept_columns: list[str]) -> list[str]:
    """Return retained complaint-count feature columns for the logistic model.

    Args:
        kept_columns: Feature columns retained by the logistic model.

    Returns:
        A sorted list of complaint-type feature columns ending in
        ``_first12m_sum``.
    """
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("complaint_type_")
            and column.endswith("_first12m_sum")
        ]
    )


def category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map display labels to raw business-category feature columns.

    Args:
        kept_columns: Feature columns retained by the logistic model.

    Returns:
        A dictionary mapping human-readable category labels to raw feature columns.
    """
    category_cols = category_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in category_cols}


def complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map display labels to raw complaint feature columns.

    Args:
        kept_columns: Feature columns retained by the logistic model.

    Returns:
        A dictionary mapping human-readable complaint labels to raw feature columns.
    """
    complaint_cols = complaint_feature_columns(kept_columns)
    return {prettify_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    """Build a zero-valued profile aligned to the retained logistic feature columns.

    Args:
        kept_columns: Feature columns retained by the logistic model.

    Returns:
        A one-row dataframe initialized to zero for all retained feature columns.
    """
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def assign_nearest_cluster_centroid(
    latitude: float,
    longitude: float,
    reference_df: pd.DataFrame,
) -> tuple[float, float]:
    """Return the nearest cluster-centroid latitude and longitude for a location.

    Args:
        latitude: Input latitude for nearest-cluster lookup.
        longitude: Input longitude for nearest-cluster lookup.
        reference_df: Reference dataframe containing cluster-center information.

    Returns:
        A tuple containing the nearest cluster-centroid latitude and longitude.
    """
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
    """Rename logistic-specific cluster columns to shared cluster field names.

    Args:
        reference_df: Reference dataframe containing logistic-specific cluster columns.

    Returns:
        A copy of the reference dataframe with available logistic cluster columns
        renamed to the shared cluster-field names expected by cluster utilities.
    """
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
    """Build a baseline logistic profile using median location and one active license.

    Args:
        kept_columns: Feature columns retained by the logistic model.
        reference_df: Reference dataframe used for median location and cluster defaults.

    Returns:
        A one-row baseline profile dataframe for logistic model prediction.
    """
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
    """Fill numeric, category, and complaint values into a single-row profile.

    Args:
        profile: One-row profile dataframe to update.
        numeric_inputs: Mapping from numeric feature columns to values.
        selected_category_columns: Category feature columns to activate.
        complaint_counts: Mapping from complaint feature columns to counts.

    Returns:
        The updated one-row profile dataframe.
    """
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
    """Build a single-row profile for the logistic regression model.

    Args:
        kept_columns: Feature columns retained by the logistic model.
        reference_df: Reference dataframe used for nearest-cluster assignment.
        inputs: User-provided business profile inputs.

    Returns:
        A one-row dataframe aligned to the retained logistic feature columns,
        ready for model prediction.
    """
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
