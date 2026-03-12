from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68

DEFAULT_SURVIVAL_MONTHS = [12, 36, 60, 120]


def prettify_cox_feature_name(column_name: str) -> str:
    for prefix in ("business_category_", "complaint_type_"):
        if column_name.startswith(prefix):
            raw = column_name[len(prefix):]
            return raw.replace("_", " ").title()
    return column_name.replace("_", " ").title()


def cox_category_feature_columns(kept_columns: list[str]) -> list[str]:
    return sorted(
        [
            column
            for column in kept_columns
            if column.startswith("business_category_")
            and column != "business_category_sum"
        ]
    )


def cox_complaint_feature_columns(kept_columns: list[str]) -> list[str]:
    return sorted(
        [column for column in kept_columns if column.startswith("complaint_type_")]
    )


def cox_category_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    category_cols = cox_category_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in category_cols}


def cox_complaint_display_to_column_map(kept_columns: list[str]) -> dict[str, str]:
    complaint_cols = cox_complaint_feature_columns(kept_columns)
    return {prettify_cox_feature_name(col): col for col in complaint_cols}


def build_zero_profile(kept_columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=[0], columns=kept_columns)


def _cluster_reference_df(reference_df: pd.DataFrame) -> pd.DataFrame:
    raw_required = {"location_cluster_lat", "location_cluster_lng"}
    agg_required = {
        "location_cluster_lat_first12m_first",
        "location_cluster_lng_first12m_first",
    }

    if raw_required.issubset(reference_df.columns):
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

    if agg_required.issubset(reference_df.columns):
        rename_map = {
            "location_cluster_first12m_first": "location_cluster",
            "location_cluster_lat_first12m_first": "location_cluster_lat",
            "location_cluster_lng_first12m_first": "location_cluster_lng",
        }
        available_cols = [col for col in rename_map if col in reference_df.columns]
        cluster_df = reference_df[available_cols].rename(columns=rename_map)
        return (
            cluster_df
            .dropna(subset=["location_cluster_lat", "location_cluster_lng"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return pd.DataFrame()


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


def get_reference_median_lat_lng(reference_df: pd.DataFrame) -> tuple[float, float]:
    if "business_latitude" in reference_df.columns and "business_longitude" in reference_df.columns:
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


def baseline_standard_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_zero_profile(kept_columns)

    median_lat, median_lng = get_reference_median_lat_lng(reference_df)
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        median_lat,
        median_lng,
        reference_df,
    )

    numeric_inputs: dict[str, Any] = {
        "active_license_count": 1.0,
        "business_latitude": float(median_lat),
        "business_longitude": float(median_lng),
        "location_cluster": float(cluster_id) if cluster_id is not None else 0.0,
        "location_cluster_lat": float(cluster_lat),
        "location_cluster_lng": float(cluster_lng),
    }

    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = float(value)

    return profile


def baseline_time_varying_cox_profile(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_zero_profile(kept_columns)

    median_lat, median_lng = get_reference_median_lat_lng(reference_df)
    cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
        median_lat,
        median_lng,
        reference_df,
    )

    numeric_inputs: dict[str, Any] = {
        "active_license_count": 1.0,
        "business_latitude": float(median_lat),
        "business_longitude": float(median_lng),
        "location_cluster": float(cluster_id) if cluster_id is not None else 0.0,
        "location_cluster_lat": float(cluster_lat),
        "location_cluster_lng": float(cluster_lng),
    }

    for column, value in numeric_inputs.items():
        if column in profile.columns:
            profile.loc[0, column] = float(value)

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


def build_time_varying_cox_profile(
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


def build_time_varying_cox_profiles_over_time(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    timepoint_specs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for spec in timepoint_specs:
        month = int(spec["month"])

        profile = build_time_varying_cox_profile(
            kept_columns=kept_columns,
            reference_df=reference_df,
            selected_category_columns=list(spec["selected_category_columns"]),
            active_license_count=int(spec["active_license_count"]),
            business_latitude=float(spec["business_latitude"]),
            business_longitude=float(spec["business_longitude"]),
            complaint_counts=dict(spec.get("complaint_counts", {})),
        ).copy()

        profile["month"] = month
        rows.append(profile)

    if not rows:
        return pd.DataFrame(columns=kept_columns + ["month"])

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values("month").reset_index(drop=True)
    return result


def _mutate_active_license_count(
    previous_value: int,
    rng: np.random.Generator,
) -> int:
    change = int(rng.choice([-1, 0, 0, 0, 1]))
    return int(min(max(previous_value + change, 1), 5))


def _sample_initial_categories(
    category_columns: list[str],
    rng: np.random.Generator,
) -> list[str]:
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
    lat = float(latitude) + float(rng.normal(0.0, 0.003))
    lng = float(longitude) + float(rng.normal(0.0, 0.003))
    return clamp_to_nyc_bounds(lat, lng)


def generate_time_varying_example_timelines(
    kept_columns: list[str],
    reference_df: pd.DataFrame,
    num_businesses: int,
    num_timepoints: int,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_state)

    category_columns = cox_category_feature_columns(kept_columns)
    complaint_columns = cox_complaint_feature_columns(kept_columns)

    generated: list[dict[str, Any]] = []

    for business_idx in range(num_businesses):
        start_lat, start_lng = _sample_reference_coordinate(reference_df, rng)
        active_license_count = int(rng.choice([1, 1, 1, 2, 2, 3]))
        selected_categories = _sample_initial_categories(category_columns, rng)
        complaint_counts = _sample_initial_complaint_counts(complaint_columns, rng)

        timepoints: list[dict[str, Any]] = []

        current_lat = start_lat
        current_lng = start_lng

        for time_idx in range(num_timepoints):
            month = int(time_idx * 12)

            if time_idx > 0:
                active_license_count = _mutate_active_license_count(active_license_count, rng)
                selected_categories = _mutate_categories(
                    selected_categories,
                    category_columns,
                    rng,
                )
                complaint_counts = _mutate_complaint_counts(
                    complaint_counts,
                    complaint_columns,
                    rng,
                )
                current_lat, current_lng = _mutate_location(current_lat, current_lng, rng)

            timepoints.append(
                {
                    "month": month,
                    "selected_category_columns": list(selected_categories),
                    "active_license_count": int(active_license_count),
                    "business_latitude": float(current_lat),
                    "business_longitude": float(current_lng),
                    "complaint_counts": dict(complaint_counts),
                }
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
    rows: list[dict[str, Any]] = []

    for business in generated_timelines:
        business_label = str(business["label"])
        for timepoint in list(business["timepoints"]):
            categories = list(timepoint["selected_category_columns"])
            complaint_counts = dict(timepoint["complaint_counts"])

            category_text = ", ".join(
                prettify_cox_feature_name(column) for column in categories
            ) if categories else "None"

            complaint_text = ", ".join(
                f"{prettify_cox_feature_name(column)} ({int(value)})"
                for column, value in sorted(complaint_counts.items())
            ) if complaint_counts else "None"

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


def clamp_to_nyc_bounds(latitude: float, longitude: float) -> tuple[float, float]:
    lat = min(max(latitude, NYC_LAT_MIN), NYC_LAT_MAX)
    lng = min(max(longitude, NYC_LNG_MIN), NYC_LNG_MAX)
    return lat, lng