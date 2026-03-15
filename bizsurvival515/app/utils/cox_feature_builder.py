"""
Helper functions for building Cox models.
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
    """
    Container for user-provided Cox profile inputs.
    """

    selected_category_columns: list[str]
    active_license_count: int
    business_latitude: float
    business_longitude: float
    complaint_counts: dict[str, float] | None = None


def prettify_cox_feature_name(column_name: str) -> str:
    """
    Prettify a Cox feature name by removing the prefix and capitalizing the rest.
    """
    for prefix in ("business_category_", "complaint_type_"):
        if column_name.startswith(prefix):
            raw = column_name[len(prefix):]
            return raw.replace("_", " ").title()
    return column_name.replace("_", " ").title()


def cox_category_feature_columns(kept_columns: list[str]) -> list[str]:
    """
    Get the columns for the Cox category features.
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
    """
    Get the columns for the Cox complaint features.
    """
    return sorted(
        [column for column in kept_columns if column.startswith("complaint_type_")]
    )


def cox_category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """
    Map the Cox category feature names to the display names.
    """
    category_cols = cox_category_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in category_cols}


def cox_complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    """
    Map the Cox complaint feature names to the display names.
    """
    complaint_cols = cox_complaint_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    """
    Build a zero profile for the Cox model.
    """
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def _cluster_reference_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the reference dataframe for the Cox model.
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
    """
    Get the median latitude and longitude of the reference dataframe.
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
    """
    Sample a reference coordinate from the reference dataframe.
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
    """
    Fill the numeric Cox inputs into the profile in place.
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
    """
    Fill the selected category columns into the profile in place.
    """
    for category_column in selected_category_columns:
        if category_column in profile.columns:
            profile.loc[0, category_column] = 1.0


def _apply_complaint_inputs(
    profile: pd.DataFrame,
    complaint_counts: dict[str, float] | None,
) -> None:
    """
    Fill complaint counts into the profile in place.
    """
    for complaint_column, count in (complaint_counts or {}).items():
        if complaint_column in profile.columns:
            profile.loc[0, complaint_column] = float(count)


def _build_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: CoxProfileInputs,
) -> pd.DataFrame:
    """
    Shared helper for building a Cox profile.
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
    """
    Build a baseline standard Cox profile.
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
    """
    Build a baseline time-varying Cox profile.
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
    """
    Build a standard Cox profile.
    """
    return _build_cox_profile(kept_columns, reference_df, inputs)


def build_time_varying_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    inputs: CoxProfileInputs,
) -> pd.DataFrame:
    """
    Build a time-varying Cox profile.
    """
    return _build_cox_profile(kept_columns, reference_df, inputs)


def build_time_varying_cox_profiles_over_time(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    timepoint_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a time-varying Cox profile over time.
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
    """
    Mutate the active license count.
    """
    change = int(rng.choice([-1, 0, 0, 0, 1]))
    return int(min(max(previous_value + change, 1), 5))


def _sample_initial_categories(
    category_columns: list[str],
    rng: np.random.Generator,
) -> list[str]:
    """
    Sample the initial categories.
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
    """
    Mutate the categories.
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
    """
    Sample the initial complaint counts.
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
    """
    Mutate the complaint counts.
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
    """
    Mutate the location.
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
    """
    Build the initial synthetic business state.
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
    """
    Mutate a synthetic business state in place.
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
    """
    Convert a synthetic state into one timepoint record.
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
    """
    Generate one synthetic business timeline.
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
    """
    Generate time-varying example timelines.
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
    """
    Summarize the generated time-varying timelines.
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
