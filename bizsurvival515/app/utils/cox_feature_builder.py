"""Build input profiles and example timelines for standard and time-varying Cox models.

Inputs:
- Kept Cox model feature-column lists
- Reference dataframes used for cluster lookup and coordinate defaults
- User-provided profile inputs for category, complaint, license-count, and
  location settings
- Synthetic-generation settings for example time-varying business timelines

Processing steps:
- Identify category and complaint feature columns from kept-column lists
- Convert feature names into display-friendly labels
- Build zero-valued baseline profiles aligned to model feature columns
- Fill numeric, category, and complaint inputs into Cox profile rows
- Build standard and time-varying Cox input profiles
- Generate synthetic business timelines across multiple timepoints
- Summarize generated timelines in tabular form for display

Outputs:
- Standard and time-varying Cox profile dataframes
- Time-varying Cox profile sequences over time
- Synthetic example business timelines
- Summary tables for generated timelines

Classes:
- CoxProfileInputs:
  Stores user-provided inputs for constructing a Cox model profile.

Functions:
- prettify_cox_feature_name:
  Convert a Cox feature column name into a display-friendly label.
- cox_category_feature_columns:
  Return kept Cox feature columns corresponding to business categories.
- cox_complaint_feature_columns:
  Return kept Cox feature columns corresponding to complaint types.
- cox_category_display_to_column_map:
  Map display labels to Cox category feature columns.
- cox_complaint_display_to_column_map:
  Map display labels to Cox complaint feature columns.
- build_zero_profile:
  Build a zero-valued profile aligned to the kept feature columns.
- _cluster_reference_df:
  Build a cluster-reference dataframe from available location columns.
- get_reference_median_lat_lng:
  Return median reference coordinates from a reference dataframe.
- _sample_reference_coordinate:
  Sample a plausible reference coordinate for synthetic profile generation.
- _apply_numeric_inputs:
  Fill numeric and location-based inputs into a Cox profile.
- _apply_category_inputs:
  Fill selected category indicators into a Cox profile.
- _apply_complaint_inputs:
  Fill complaint counts into a Cox profile.
- _build_cox_profile:
  Shared helper for building a Cox profile from user inputs.
- baseline_standard_cox_profile:
  Build a baseline profile for the standard Cox model.
- baseline_time_varying_cox_profile:
  Build a baseline profile for the time-varying Cox model.
- build_standard_cox_profile:
  Build a user-specified profile for the standard Cox model.
- build_time_varying_cox_profile:
  Build a user-specified profile for the time-varying Cox model.
- build_time_varying_cox_profiles_over_time:
  Build a sequence of time-varying Cox profiles across multiple months.
- _mutate_active_license_count:
  Mutate the active-license-count state for synthetic timeline generation.
- _sample_initial_categories:
  Sample initial category assignments for a synthetic business.
- _mutate_categories:
  Mutate category assignments across timepoints.
- _sample_initial_complaint_counts:
  Sample initial complaint counts for a synthetic business.
- _mutate_complaint_counts:
  Mutate complaint counts across timepoints.
- _mutate_location:
  Mutate business coordinates across timepoints.
- _initial_timeline_state:
  Build the initial synthetic state for one business timeline.
- _advance_timeline_state:
  Mutate a synthetic business state in place for the next timepoint.
- _build_timepoint_record:
  Convert one synthetic state into a timepoint record.
- _generate_business_timeline:
  Generate one synthetic business timeline.
- generate_time_varying_example_timelines:
  Generate multiple example time-varying business timelines.
- summarize_generated_time_varying_timelines:
  Summarize generated timelines in a display-friendly dataframe.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .location_utils import (
    assign_nearest_cluster_info,
    build_cluster_reference_df,
    clamp_to_nyc_bounds,
)


@dataclass
class CoxProfileInputs:
    """Store user-provided inputs for constructing a Cox model profile.

    Attributes:
        selected_category_columns: Category feature columns set to active.
        active_license_count: Active license count for the business profile.
        business_latitude: Business latitude used for location-based features.
        business_longitude: Business longitude used for location-based features.
        complaint_counts: Optional mapping from complaint feature columns to counts.
    """

    selected_category_columns: list[str]
    active_license_count: int
    business_latitude: float
    business_longitude: float
    complaint_counts: dict[str, float] | None = None


def prettify_cox_feature_name(column_name: str) -> str:
    """Convert a Cox feature column name into a display-friendly label.

    Args:
        column_name: Raw Cox feature column name.

    Returns:
        A human-readable label with standard prefixes removed and words title-cased.
    """
    for prefix in ("business_category_", "complaint_type_"):
        if column_name.startswith(prefix):
            raw = column_name[len(prefix):]
            return raw.replace("_", " ").title()
    return column_name.replace("_", " ").title()


def cox_category_feature_columns(kept_columns: list[str]) -> list[str]:
    """Return kept Cox feature columns corresponding to business categories.

    Args:
        kept_columns: Feature columns retained by a Cox model.

    Returns:
        A sorted list of business-category feature columns, excluding
        ``business_category_sum``.
    """
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("business_category_")
            and column != "business_category_sum"
        ]
    )


def cox_complaint_feature_columns(kept_columns: list[str]) -> list[str]:
    """Return kept Cox feature columns corresponding to complaint types.

    Args:
        kept_columns: Feature columns retained by a Cox model.

    Returns:
        A sorted list of complaint-type feature columns.
    """
    return sorted(
        [column for column in kept_columns if column.startswith("complaint_type_")]
    )


def cox_category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map display labels to retained Cox category feature columns.

    Args:
        kept_columns: Feature columns retained by a Cox model.

    Returns:
        A dictionary mapping human-readable category labels to raw feature columns.
    """
    category_cols = cox_category_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in category_cols}


def cox_complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """Map display labels to retained Cox complaint feature columns.

    Args:
        kept_columns: Feature columns retained by a Cox model.

    Returns:
        A dictionary mapping human-readable complaint labels to raw feature columns.
    """
    complaint_cols = cox_complaint_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    """Build a zero-valued profile aligned to the retained Cox feature columns.

    Args:
        kept_columns: Feature columns retained by a Cox model.

    Returns:
        A one-row dataframe initialized to zero for all retained feature columns.
    """
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def _cluster_reference_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """Build a cluster-reference dataframe from available location columns.

    Args:
        reference_df: Reference dataframe containing raw or aggregated cluster-location columns.

    Returns:
        A dataframe suitable for nearest-cluster lookup, or an empty dataframe
        if the required location-cluster columns are unavailable.
    """
    raw_required = {"location_cluster_lat", "location_cluster_lng"}
    agg_required = {
        "location_cluster_lat_first12m_first",
        "location_cluster_lng_first12m_first",
    }

    if raw_required.issubset(reference_df.columns):
        return build_cluster_reference_df(
            reference_df=reference_df,
            lat_column="location_cluster_lat",
            lng_column="location_cluster_lng",
            cluster_column="location_cluster",
        )

    if agg_required.issubset(reference_df.columns):
        return build_cluster_reference_df(
            reference_df=reference_df,
            lat_column="location_cluster_lat_first12m_first",
            lng_column="location_cluster_lng_first12m_first",
            cluster_column="location_cluster_first12m_first",
        )

    return pd.DataFrame()


def get_reference_median_lat_lng(reference_df: pd.DataFrame) -> tuple[float, float]:
    """Return median reference coordinates from a reference dataframe.

    Args:
        reference_df: Reference dataframe containing raw or aggregated business coordinates.

    Returns:
        A tuple containing median latitude and longitude values, or default NYC
        coordinates if suitable columns are unavailable.
    """
    if (
        "business_latitude" in reference_df.columns
        and "business_longitude" in reference_df.columns
    ):
        return (
            float(reference_df["business_latitude"].median()),
            float(reference_df["business_longitude"].median()),
        )

    if (
        "business_latitude_first12m_first" in reference_df.columns
        and "business_longitude_first12m_first" in reference_df.columns
    ):
        return (
            float(reference_df["business_latitude_first12m_first"].median()),
            float(reference_df["business_longitude_first12m_first"].median()),
        )

    return 40.7128, -74.0060


def _sample_reference_coordinate(
    reference_df: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Sample a plausible reference coordinate for synthetic profile generation.

    Args:
        reference_df: Reference dataframe containing candidate business coordinates.
        rng: Random-number generator used for reproducible sampling.

    Returns:
        A tuple containing latitude and longitude values clamped to NYC bounds.
    """
    if (
        "business_latitude_first12m_first" in reference_df.columns
        and "business_longitude_first12m_first" in reference_df.columns
    ):
        coords = reference_df[
            ["business_latitude_first12m_first", "business_longitude_first12m_first"]
        ].dropna()

        if not coords.empty:
            sampled = coords.sample(n=1, random_state=int(rng.integers(0, 1_000_000)))
            lat = float(sampled["business_latitude_first12m_first"].iloc[0])
            lng = float(sampled["business_longitude_first12m_first"].iloc[0])
            return clamp_to_nyc_bounds(lat, lng)

    median_lat, median_lng = get_reference_median_lat_lng(reference_df)
    return clamp_to_nyc_bounds(median_lat, median_lng)


def _apply_numeric_inputs(
    profile: pd.DataFrame,
    reference_df: pd.DataFrame,
    active_license_count: int,
    business_latitude: float,
    business_longitude: float,
) -> None:
    """Fill numeric and location-derived inputs into a Cox profile in place.

    Args:
        profile: One-row profile dataframe to update.
        reference_df: Reference dataframe used for nearest-cluster assignment.
        active_license_count: Active license count to assign.
        business_latitude: Latitude to assign and use for cluster lookup.
        business_longitude: Longitude to assign and use for cluster lookup.

    Returns:
        None.
    """
    cluster_df = _cluster_reference_df(reference_df)
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        business_latitude,
        business_longitude,
        cluster_df,
    )

    numeric_inputs: dict[str, float] = {
        "active_license_count": float(active_license_count),
        "business_latitude": float(business_latitude),
        "business_longitude": float(business_longitude),
        "location_cluster": float(cluster_id) if cluster_id is not None else 0.0,
        "location_cluster_lat": float(cluster_lat),
        "location_cluster_lng": float(cluster_lng),
    }

    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = value


def _apply_category_inputs(
    profile: pd.DataFrame,
    selected_category_columns: list[str],
) -> None:
    """Fill selected category indicators into a Cox profile in place.

    Args:
        profile: One-row profile dataframe to update.
        selected_category_columns: Category feature columns to activate.

    Returns:
        None.
    """
    for category_column in selected_category_columns:
        if category_column in profile.columns:
            profile.loc[0, category_column] = 1.0


def _apply_complaint_inputs(
    profile: pd.DataFrame,
    complaint_counts: dict[str, float] | None,
) -> None:
    """Fill complaint counts into a Cox profile in place.

    Args:
        profile: One-row profile dataframe to update.
        complaint_counts: Optional mapping from complaint feature columns to counts.

    Returns:
        None.
    """
    for complaint_column, count in (complaint_counts or {}).items():
        if complaint_column in profile.columns:
            profile.loc[0, complaint_column] = float(count)


def _build_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: CoxProfileInputs,
) -> pd.DataFrame:
    """Build a Cox profile from retained columns, reference data, and user inputs.

    Args:
        kept_columns: Feature columns retained by the Cox model.
        reference_df: Reference dataframe used for location and cluster defaults.
        inputs: User-provided profile inputs.

    Returns:
        A one-row dataframe aligned to the retained feature columns.
    """
    profile = build_zero_profile(kept_columns)

    _apply_numeric_inputs(
        profile=profile,
        reference_df=reference_df,
        active_license_count=inputs.active_license_count,
        business_latitude=inputs.business_latitude,
        business_longitude=inputs.business_longitude,
    )
    _apply_category_inputs(profile, inputs.selected_category_columns)
    _apply_complaint_inputs(profile, inputs.complaint_counts)

    return profile


def baseline_standard_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a baseline profile for the standard Cox model.

    Args:
        kept_columns: Feature columns retained by the standard Cox model.
        reference_df: Reference dataframe used for baseline coordinate defaults.

    Returns:
        A one-row baseline profile dataframe for standard Cox scoring.
    """
    median_lat, median_lng = get_reference_median_lat_lng(reference_df)
    baseline_inputs = CoxProfileInputs(
        selected_category_columns=[],
        active_license_count=1,
        business_latitude=float(median_lat),
        business_longitude=float(median_lng),
        complaint_counts={},
    )
    return _build_cox_profile(kept_columns, reference_df, baseline_inputs)


def baseline_time_varying_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a baseline profile for the time-varying Cox model.

    Args:
        kept_columns: Feature columns retained by the time-varying Cox model.
        reference_df: Reference dataframe used for baseline coordinate defaults.

    Returns:
        A one-row baseline profile dataframe for time-varying Cox scoring.
    """
    median_lat, median_lng = get_reference_median_lat_lng(reference_df)
    baseline_inputs = CoxProfileInputs(
        selected_category_columns=[],
        active_license_count=1,
        business_latitude=float(median_lat),
        business_longitude=float(median_lng),
        complaint_counts={},
    )
    return _build_cox_profile(kept_columns, reference_df, baseline_inputs)


def build_standard_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: CoxProfileInputs,
) -> pd.DataFrame:
    """Build a user-specified profile for the standard Cox model.

    Args:
        kept_columns: Feature columns retained by the standard Cox model.
        reference_df: Reference dataframe used for cluster assignment.
        inputs: User-provided profile inputs.

    Returns:
        A one-row profile dataframe for standard Cox scoring.
    """
    return _build_cox_profile(kept_columns, reference_df, inputs)


def build_time_varying_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: CoxProfileInputs,
) -> pd.DataFrame:
    """Build a user-specified profile for the time-varying Cox model.

    Args:
        kept_columns: Feature columns retained by the time-varying Cox model.
        reference_df: Reference dataframe used for cluster assignment.
        inputs: User-provided profile inputs.

    Returns:
        A one-row profile dataframe for time-varying Cox scoring.
    """
    return _build_cox_profile(kept_columns, reference_df, inputs)


def build_time_varying_cox_profiles_over_time(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    timepoint_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a sequence of time-varying Cox profiles across multiple months.

    Args:
        kept_columns: Feature columns retained by the time-varying Cox model.
        reference_df: Reference dataframe used for cluster assignment.
        timepoint_specs: Timepoint specifications containing month and profile inputs.

    Returns:
        A dataframe of profile rows sorted by month, or an empty dataframe with
        the expected columns if no timepoints are provided.

    Raises:
        KeyError: If a required timepoint field is missing from a specification.
        ValueError: If a provided timepoint field cannot be coerced to the expected type.
    """
    rows: list[pd.DataFrame] = []

    for spec in timepoint_specs:
        month = int(spec["month"])
        inputs = CoxProfileInputs(
            selected_category_columns=list(spec["selected_category_columns"]),
            active_license_count=int(spec["active_license_count"]),
            business_latitude=float(spec["business_latitude"]),
            business_longitude=float(spec["business_longitude"]),
            complaint_counts=dict(spec.get("complaint_counts", {})),
        )

        profile = build_time_varying_cox_profile(
            kept_columns=kept_columns,
            reference_df=reference_df,
            inputs=inputs,
        ).copy()

        profile["month"] = month
        rows.append(profile)

    if not rows:
        return pd.DataFrame(columns=kept_columns + ["month"])

    return pd.concat(rows, ignore_index=True).sort_values("month").reset_index(drop=True)


def _mutate_active_license_count(
    previous_value: int,
    rng: np.random.Generator,
) -> int:
    """Mutate the active license count for synthetic timeline generation.

    Args:
        previous_value: Previous active-license-count value.
        rng: Random-number generator used for reproducible mutation.

    Returns:
        The mutated active-license-count value, clipped to a valid range.
    """
    change = int(rng.choice([-1, 0, 0, 0, 1]))
    return int(min(max(previous_value + change, 1), 5))


def _sample_initial_categories(
    category_columns: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Sample initial category assignments for a synthetic business.

    Args:
        category_columns: Available category feature columns.
        rng: Random-number generator used for reproducible sampling.

    Returns:
        A sorted list of selected category feature columns.
    """
    if not category_columns:
        return []

    n_categories = int(rng.choice([0, 1, 1, 1, 2]))
    if n_categories == 0:
        return []

    n_categories = min(n_categories, len(category_columns))
    sampled = rng.choice(category_columns, size=n_categories, replace=False)
    return sorted(sampled.tolist())


def _mutate_categories(
    previous_categories: list[str],
    category_columns: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """Mutate category assignments across synthetic timepoints.

    Args:
        previous_categories: Previously selected category feature columns.
        category_columns: Available category feature columns.
        rng: Random-number generator used for reproducible mutation.

    Returns:
        A sorted list of updated category feature columns.
    """
    categories = set(previous_categories)
    action = str(rng.choice(["keep", "keep", "keep", "add", "remove"]))

    if action == "add":
        available = [col for col in category_columns if col not in categories]
        if available and len(categories) < 2:
            categories.add(str(rng.choice(available)))
    elif action == "remove" and categories:
        categories.remove(str(rng.choice(list(categories))))

    return sorted(categories)


def _sample_initial_complaint_counts(
    complaint_columns: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Sample initial complaint counts for a synthetic business.

    Args:
        complaint_columns: Available complaint feature columns.
        rng: Random-number generator used for reproducible sampling.

    Returns:
        A mapping from selected complaint feature columns to initial counts.
    """
    counts: dict[str, float] = {}

    if not complaint_columns:
        return counts

    n_selected = int(rng.choice([0, 0, 1, 1, 2]))
    if n_selected == 0:
        return counts

    n_selected = min(n_selected, len(complaint_columns))
    selected = rng.choice(complaint_columns, size=n_selected, replace=False)

    for complaint_col in selected.tolist():
        counts[str(complaint_col)] = float(int(rng.choice([1, 1, 1, 2, 3])))

    return counts


def _mutate_complaint_counts(
    previous_counts: dict[str, float],
    complaint_columns: list[str],
    rng: np.random.Generator,
) -> dict[str, float]:
    """Mutate complaint counts across synthetic timepoints.

    Args:
        previous_counts: Previously assigned complaint counts.
        complaint_columns: Available complaint feature columns.
        rng: Random-number generator used for reproducible mutation.

    Returns:
        An updated mapping from complaint feature columns to counts.
    """
    counts = {str(key): float(value) for key, value in previous_counts.items()}

    for complaint_col in list(counts.keys()):
        change = int(rng.choice([-1, 0, 0, 1]))
        new_value = max(0, int(counts[complaint_col]) + change)
        if new_value == 0:
            counts.pop(complaint_col, None)
        else:
            counts[complaint_col] = float(new_value)

    maybe_add = bool(rng.choice([False, False, True]))
    if maybe_add:
        available = [col for col in complaint_columns if col not in counts]
        if available and len(counts) < 3:
            new_col = str(rng.choice(available))
            counts[new_col] = float(int(rng.choice([1, 1, 2])))

    return counts


def _mutate_location(
    latitude: float,
    longitude: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Mutate business coordinates for synthetic timeline generation.

    Args:
        latitude: Previous latitude value.
        longitude: Previous longitude value.
        rng: Random-number generator used for reproducible mutation.

    Returns:
        A tuple containing mutated latitude and longitude values clamped to NYC bounds.
    """
    lat = float(latitude) + float(rng.normal(0.0, 0.003))
    lng = float(longitude) + float(rng.normal(0.0, 0.003))
    return clamp_to_nyc_bounds(lat, lng)


def _initial_timeline_state(
    reference_df: pd.DataFrame,
    category_columns: list[str],
    complaint_columns: list[str],
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build the initial synthetic state for one business timeline.

    Args:
        reference_df: Reference dataframe used for coordinate defaults.
        category_columns: Available category feature columns.
        complaint_columns: Available complaint feature columns.
        rng: Random-number generator used for reproducible sampling.

    Returns:
        A dictionary describing the initial synthetic business state.
    """
    start_lat, start_lng = _sample_reference_coordinate(reference_df, rng)
    return {
        "active_license_count": int(rng.choice([1, 1, 1, 2, 2, 3])),
        "selected_categories": _sample_initial_categories(category_columns, rng),
        "complaint_counts": _sample_initial_complaint_counts(complaint_columns, rng),
        "current_lat": start_lat,
        "current_lng": start_lng,
    }


def _advance_timeline_state(
    state: dict[str, Any],
    category_columns: list[str],
    complaint_columns: list[str],
    rng: np.random.Generator,
) -> None:
    """Mutate a synthetic business state in place for the next timepoint.

    Args:
        state: Mutable synthetic business state to update.
        category_columns: Available category feature columns.
        complaint_columns: Available complaint feature columns.
        rng: Random-number generator used for reproducible mutation.

    Returns:
        None.
    """
    state["active_license_count"] = _mutate_active_license_count(
        int(state["active_license_count"]),
        rng,
    )
    state["selected_categories"] = _mutate_categories(
        list(state["selected_categories"]),
        category_columns,
        rng,
    )
    state["complaint_counts"] = _mutate_complaint_counts(
        dict(state["complaint_counts"]),
        complaint_columns,
        rng,
    )
    state["current_lat"], state["current_lng"] = _mutate_location(
        float(state["current_lat"]),
        float(state["current_lng"]),
        rng,
    )


def _build_timepoint_record(
    month: int,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Convert one synthetic state into a timepoint record.

    Args:
        month: Month value assigned to the timepoint.
        state: Synthetic business state at the given timepoint.

    Returns:
        A dictionary describing one timepoint in a synthetic business timeline.
    """
    return {
        "month": month,
        "selected_category_columns": list(state["selected_categories"]),
        "active_license_count": int(state["active_license_count"]),
        "business_latitude": float(state["current_lat"]),
        "business_longitude": float(state["current_lng"]),
        "complaint_counts": dict(state["complaint_counts"]),
    }


def _generate_business_timeline(
    reference_df: pd.DataFrame,
    category_columns: list[str],
    complaint_columns: list[str],
    num_timepoints: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Generate one synthetic business timeline across multiple timepoints.

    Args:
        reference_df: Reference dataframe used for coordinate defaults.
        category_columns: Available category feature columns.
        complaint_columns: Available complaint feature columns.
        num_timepoints: Number of timepoints to generate.
        rng: Random-number generator used for reproducible generation.

    Returns:
        A list of timepoint records for one synthetic business.
    """
    state = _initial_timeline_state(
        reference_df=reference_df,
        category_columns=category_columns,
        complaint_columns=complaint_columns,
        rng=rng,
    )

    timepoints: list[dict[str, Any]] = []

    for time_idx in range(num_timepoints):
        month = int(time_idx * 12)

        if time_idx > 0:
            _advance_timeline_state(
                state=state,
                category_columns=category_columns,
                complaint_columns=complaint_columns,
                rng=rng,
            )

        timepoints.append(_build_timepoint_record(month, state))

    return timepoints


def generate_time_varying_example_timelines(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    num_businesses: int,
    num_timepoints: int,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Generate example time-varying business timelines for app demonstration.

    Args:
        kept_columns: Feature columns retained by the time-varying Cox model.
        reference_df: Reference dataframe used for coordinate defaults.
        num_businesses: Number of synthetic businesses to generate.
        num_timepoints: Number of timepoints per business timeline.
        random_state: Random seed used for reproducible generation.

    Returns:
        A list of generated business timeline dictionaries.
    """
    rng = np.random.default_rng(random_state)
    category_columns = cox_category_feature_columns(kept_columns)
    complaint_columns = cox_complaint_feature_columns(kept_columns)

    generated: list[dict[str, Any]] = []

    for business_idx in range(num_businesses):
        timepoints = _generate_business_timeline(
            reference_df=reference_df,
            category_columns=category_columns,
            complaint_columns=complaint_columns,
            num_timepoints=num_timepoints,
            rng=rng,
        )
        generated.append(
            {
                "label": f"Business {business_idx + 1}",
                "timepoints": timepoints,
            }
        )

    return generated


def summarize_generated_time_varying_timelines(
    generated_timelines: list[dict[str, Any]],
) -> pd.DataFrame:
    """Summarize generated time-varying timelines in a display-friendly dataframe.

    Args:
        generated_timelines: Generated synthetic business timelines.

    Returns:
        A dataframe summarizing each business timeline by month, location,
        categories, and complaint counts.
    """
    rows: list[dict[str, Any]] = []

    for business in generated_timelines:
        business_label = str(business["label"])
        for timepoint in list(business["timepoints"]):
            categories = list(timepoint["selected_category_columns"])
            complaint_counts = dict(timepoint["complaint_counts"])

            category_text = (
                ", ".join(prettify_cox_feature_name(column) for column in categories)
                if categories
                else "None"
            )

            complaint_text = (
                ", ".join(
                    f"{prettify_cox_feature_name(column)} ({int(value)})"
                    for column, value in sorted(complaint_counts.items())
                )
                if complaint_counts
                else "None"
            )

            rows.append(
                {
                    "Business": business_label,
                    "Month": int(timepoint["month"]),
                    "Active license count": int(timepoint["active_license_count"]),
                    "Latitude": float(timepoint["business_latitude"]),
                    "Longitude": float(timepoint["business_longitude"]),
                    "Categories": category_text,
                    "Complaint counts": complaint_text,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Business",
                "Month",
                "Active license count",
                "Latitude",
                "Longitude",
                "Categories",
                "Complaint counts",
            ]
        )

    return pd.DataFrame(rows).sort_values(["Business", "Month"]).reset_index(drop=True)
