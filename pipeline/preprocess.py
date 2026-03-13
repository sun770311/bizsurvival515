# preprocess.py
"""Preprocess NYC business license and 311 data into a monthly panel dataset.

Input (raw CSVs from NYC Open Data):
- licenses.csv
- service_reqs_sample.csv or service_reqs.csv

Processing steps:
- Clean and standardize records
- Expand licenses into a business-month panel
- Encode categories as human-readable features
- Spatially join 311 complaints to nearby businesses
- Add location clusters and export final dataset

Output:
- joined_dataset.csv
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree

from pipeline.utils import LOCATION_FEATURE_COLUMNS


LOCATION_K = 25
RADIUS_METERS = 50
EARTH_RADIUS_METERS = 6_371_000
RADIUS_RADIANS = RADIUS_METERS / EARTH_RADIUS_METERS

LICENSE_SOURCE_COLUMNS = [
    "License Number",
    "Business Unique ID",
    "Business Category",
    "License Type",
    "License Status",
    "Initial Issuance Date",
    "Expiration Date",
    "Latitude",
    "Longitude",
]

SERVICE_REQUEST_SOURCE_COLUMNS = [
    "Unique Key",
    "Created Date",
    "Problem (formerly Complaint Type)",
    "Latitude",
    "Longitude",
]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for preprocessing pipeline paths and parameters."""

    licenses_path: Path
    service_reqs_path: Path
    output_path: Path
    location_k: int = LOCATION_K
    radius_radians: float = RADIUS_RADIANS


def sanitize_feature_name(raw_name: str, prefix: str) -> str:
    """Convert a raw category/complaint label into a safe column name."""
    cleaned = str(raw_name).strip().lower()
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"['’]", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    if not cleaned:
        cleaned = "unknown"

    return f"{prefix}_{cleaned}"


def make_unique_column_names(names: Iterable[str]) -> list[str]:
    """Ensure generated column names are unique by appending suffixes."""
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        if name not in counts:
            counts[name] = 0
            unique_names.append(name)
            continue

        counts[name] += 1
        unique_names.append(f"{name}_{counts[name]}")

    return unique_names


def load_source_data(
    licenses_path: Path,
    service_reqs_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load required columns from source CSV files."""
    licenses = pd.read_csv(
        licenses_path,
        usecols=LICENSE_SOURCE_COLUMNS,
    )

    service_reqs = pd.read_csv(
        service_reqs_path,
        usecols=SERVICE_REQUEST_SOURCE_COLUMNS,
    )

    return licenses, service_reqs


def clean_licenses(licenses: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize license records."""
    cleaned = licenses.copy()

    string_columns = [
        "Business Unique ID",
        "License Number",
        "Business Category",
        "License Type",
        "License Status",
    ]
    for column in string_columns:
        cleaned[column] = cleaned[column].astype(str).str.strip()
        cleaned[column] = cleaned[column].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )

    cleaned["Initial Issuance Date"] = pd.to_datetime(
        cleaned["Initial Issuance Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )
    cleaned["Expiration Date"] = pd.to_datetime(
        cleaned["Expiration Date"],
        format="%m/%d/%Y",
        errors="coerce",
    )
    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=[
            "Business Unique ID",
            "License Number",
            "Initial Issuance Date",
            "Expiration Date",
            "Business Category",
        ]
    ).copy()

    cleaned = cleaned.loc[cleaned["Business Category"] != ""].copy()

    cleaned["issue_month"] = (
        cleaned["Initial Issuance Date"].dt.to_period("M").dt.to_timestamp()
    )
    cleaned["expire_month"] = (
        cleaned["Expiration Date"].dt.to_period("M").dt.to_timestamp()
    )

    cleaned = cleaned.loc[cleaned["expire_month"] >= cleaned["issue_month"]].copy()
    return cleaned


def clean_service_requests(service_reqs: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize 311 service request records."""
    cleaned = service_reqs.copy()

    cleaned["Unique Key"] = cleaned["Unique Key"].astype(str).str.strip()
    cleaned["Unique Key"] = cleaned["Unique Key"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )

    cleaned["Problem (formerly Complaint Type)"] = (
        cleaned["Problem (formerly Complaint Type)"].astype(str).str.strip()
    )
    cleaned["Problem (formerly Complaint Type)"] = (
        cleaned["Problem (formerly Complaint Type)"].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )
    )

    cleaned["Created Date"] = pd.to_datetime(
        cleaned["Created Date"],
        format="%m/%d/%Y %I:%M:%S %p",
        errors="coerce",
    )
    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    cleaned = cleaned.dropna(
        subset=[
            "Unique Key",
            "Created Date",
            "Latitude",
            "Longitude",
            "Problem (formerly Complaint Type)",
        ]
    ).copy()

    cleaned = cleaned.loc[
        cleaned["Problem (formerly Complaint Type)"] != ""
    ].copy()

    cleaned["month"] = cleaned["Created Date"].dt.to_period("M").dt.to_timestamp()
    return cleaned


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return monthly start dates from start to end, inclusive."""
    return pd.date_range(start=start, end=end, freq="MS")


def expand_licenses_to_months(licenses: pd.DataFrame) -> pd.DataFrame:
    """Expand each license row into one row per active month."""
    expanded = licenses.copy()
    expanded["month"] = expanded.apply(
        lambda row: month_range(row["issue_month"], row["expire_month"]),
        axis=1,
    )
    expanded = expanded.explode("month").reset_index(drop=True)
    return expanded


def build_license_panel(
    licenses: pd.DataFrame,
    licenses_expanded: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Build business-month license panel with human-readable category columns."""
    first_license_dates = (
        licenses.groupby("Business Unique ID")["Initial Issuance Date"]
        .min()
        .rename("first_license_date")
        .reset_index()
    )

    license_counts = (
        licenses_expanded.groupby(["Business Unique ID", "month"])["License Number"]
        .nunique()
        .rename("active_license_count")
        .reset_index()
    )

    license_cat_counts = (
        licenses_expanded.groupby(
            ["Business Unique ID", "month", "Business Category"]
        )["License Number"]
        .count()
        .reset_index(name="count")
    )

    if license_cat_counts.empty:
        license_panel = (
            license_counts.merge(
                first_license_dates,
                on="Business Unique ID",
                how="left",
            )
            .rename(columns={"Business Unique ID": "business_id"})
            .copy()
        )
        license_panel["months_since_first_license"] = (
            (license_panel["month"].dt.year - license_panel["first_license_date"].dt.year)
            * 12
            + (
                license_panel["month"].dt.month
                - license_panel["first_license_date"].dt.month
            )
        )
        license_panel["open"] = 1
        return license_panel, []

    raw_categories = sorted(license_cat_counts["Business Category"].dropna().unique())
    human_readable_columns = make_unique_column_names(
        sanitize_feature_name(category, "business_category")
        for category in raw_categories
    )
    category_column_map = dict(zip(raw_categories, human_readable_columns))

    license_cat_counts["category_column"] = license_cat_counts["Business Category"].map(
        category_column_map
    )

    license_cat_wide = license_cat_counts.pivot_table(
        index=["Business Unique ID", "month"],
        columns="category_column",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    category_columns = sorted(
        [
            column
            for column in license_cat_wide.columns
            if column not in {"Business Unique ID", "month"}
        ]
    )

    license_panel = (
        license_counts.merge(
            license_cat_wide,
            on=["Business Unique ID", "month"],
            how="left",
        )
        .merge(first_license_dates, on="Business Unique ID", how="left")
        .rename(columns={"Business Unique ID": "business_id"})
    )

    license_panel["months_since_first_license"] = (
        (license_panel["month"].dt.year - license_panel["first_license_date"].dt.year)
        * 12
        + (license_panel["month"].dt.month - license_panel["first_license_date"].dt.month)
    )
    license_panel["open"] = 1

    for column in category_columns:
        license_panel[column] = license_panel[column].fillna(0)

    return license_panel, category_columns


def compute_business_locations(
    licenses: pd.DataFrame,
) -> pd.DataFrame:
    """Compute one representative median coordinate pair per business."""
    business_locations = (
        licenses.dropna(subset=["Latitude", "Longitude"])
        .groupby("Business Unique ID")[["Latitude", "Longitude"]]
        .median()
        .reset_index()
        .rename(
            columns={
                "Business Unique ID": "business_id",
                "Latitude": "business_latitude",
                "Longitude": "business_longitude",
            }
        )
    )
    return business_locations


def assign_location_clusters(
    business_locations: pd.DataFrame,
    location_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign location clusters and attach cluster center coordinates."""
    clustered = business_locations.copy()

    if clustered.empty:
        clustered["location_cluster"] = pd.Series(dtype="int64")
        clustered["location_cluster_lat"] = pd.Series(dtype="float64")
        clustered["location_cluster_lng"] = pd.Series(dtype="float64")
        return clustered, clustered.copy()

    if len(clustered) >= location_k:
        kmeans = KMeans(n_clusters=location_k, random_state=42, n_init=20)
        clustered["location_cluster"] = kmeans.fit_predict(
            clustered[["business_latitude", "business_longitude"]]
        )
        cluster_centers = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=["location_cluster_lat", "location_cluster_lng"],
        )
        cluster_centers["location_cluster"] = cluster_centers.index
    else:
        clustered["location_cluster"] = 0
        center_lat = clustered["business_latitude"].mean()
        center_lng = clustered["business_longitude"].mean()
        cluster_centers = pd.DataFrame(
            {
                "location_cluster": [0],
                "location_cluster_lat": [center_lat],
                "location_cluster_lng": [center_lng],
            }
        )

    clustered = clustered.merge(
        cluster_centers,
        on="location_cluster",
        how="left",
    )

    return clustered, cluster_centers


def radius_join_requests_to_businesses(
    business_locations: pd.DataFrame,
    service_reqs: pd.DataFrame,
    radius_radians: float,
) -> pd.DataFrame:
    """Match 311 requests to nearby businesses within the configured radius."""
    biz = business_locations.dropna(
        subset=["business_latitude", "business_longitude"]
    ).copy()
    req = service_reqs.dropna(subset=["Latitude", "Longitude"]).copy()

    if biz.empty or req.empty:
        return pd.DataFrame(
            columns=["Unique Key", "month", "complaint_type", "business_id"]
        )

    biz["lat_rad"] = np.radians(biz["business_latitude"])
    biz["lon_rad"] = np.radians(biz["business_longitude"])
    req["lat_rad"] = np.radians(req["Latitude"])
    req["lon_rad"] = np.radians(req["Longitude"])

    biz_coords = np.c_[biz["lat_rad"].to_numpy(), biz["lon_rad"].to_numpy()]
    req_coords = np.c_[req["lat_rad"].to_numpy(), req["lon_rad"].to_numpy()]

    tree = BallTree(biz_coords, metric="haversine")
    indices_array = tree.query_radius(req_coords, r=radius_radians)

    pairs: list[dict[str, object]] = []
    for req_idx, biz_indices in enumerate(indices_array):
        if len(biz_indices) == 0:
            continue

        req_row = req.iloc[req_idx]
        for biz_idx in biz_indices:
            biz_row = biz.iloc[biz_idx]
            pairs.append(
                {
                    "Unique Key": req_row["Unique Key"],
                    "month": req_row["month"],
                    "complaint_type": req_row["Problem (formerly Complaint Type)"],
                    "business_id": biz_row["business_id"],
                }
            )

    return pd.DataFrame(pairs)


def build_complaint_panel(
    req_business: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate 311 complaints by business-month using human-readable columns."""
    if req_business.empty:
        complaint_panel = pd.DataFrame(columns=["business_id", "month", "total_311"])
        return complaint_panel, []

    total_311 = (
        req_business.groupby(["business_id", "month"])["Unique Key"]
        .count()
        .rename("total_311")
        .reset_index()
    )

    complaint_counts = (
        req_business.groupby(["business_id", "month", "complaint_type"])["Unique Key"]
        .count()
        .reset_index(name="count")
    )

    raw_complaint_types = sorted(complaint_counts["complaint_type"].dropna().unique())
    human_readable_columns = make_unique_column_names(
        sanitize_feature_name(complaint_type, "complaint_type")
        for complaint_type in raw_complaint_types
    )
    complaint_column_map = dict(zip(raw_complaint_types, human_readable_columns))

    complaint_counts["complaint_column"] = complaint_counts["complaint_type"].map(
        complaint_column_map
    )

    complaint_wide = complaint_counts.pivot_table(
        index=["business_id", "month"],
        columns="complaint_column",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()

    complaint_columns = sorted(
        [
            column
            for column in complaint_wide.columns
            if column not in {"business_id", "month"}
        ]
    )

    complaint_panel = total_311.merge(
        complaint_wide,
        on=["business_id", "month"],
        how="left",
    )

    for column in complaint_columns:
        complaint_panel[column] = complaint_panel[column].fillna(0)

    return complaint_panel, complaint_columns


def merge_final_dataset(
    license_panel: pd.DataFrame,
    complaint_panel: pd.DataFrame,
    business_locations: pd.DataFrame,
    category_columns: list[str],
    complaint_columns: list[str],
) -> pd.DataFrame:
    """Merge license, complaint, and location data into final dataset."""
    joined = license_panel.merge(
        complaint_panel,
        on=["business_id", "month"],
        how="left",
    )

    joined = joined.merge(
        business_locations[["business_id", *LOCATION_FEATURE_COLUMNS]],
        on="business_id",
        how="left",
    )

    if "total_311" not in joined.columns:
        joined["total_311"] = 0
    else:
        joined["total_311"] = joined["total_311"].fillna(0)

    for column in complaint_columns:
        joined[column] = joined[column].fillna(0)

    for column in category_columns:
        joined[column] = joined[column].fillna(0)

    joined["location_cluster"] = joined["location_cluster"].fillna(-1).astype(int)

    joined["business_category_sum"] = (
        joined[category_columns].sum(axis=1) if category_columns else 0
    )
    joined["complaint_sum"] = (
        joined[complaint_columns].sum(axis=1) if complaint_columns else 0
    )

    final_columns = (
        ["business_id", "month", "active_license_count"]
        + category_columns
        + ["total_311"]
        + complaint_columns
        + [
            "open",
            "months_since_first_license",
            *LOCATION_FEATURE_COLUMNS,
            "business_category_sum",
            "complaint_sum",
        ]
    )

    joined = joined[final_columns].sort_values(
        ["business_id", "month"]
    ).reset_index(drop=True)

    joined = joined[joined["location_cluster"] != -1].copy()
    joined = joined.dropna(
        subset=["location_cluster_lat", "location_cluster_lng"]
    ).copy()

    return joined


def build_joined_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return the final dataset."""
    licenses_raw, service_reqs_raw = load_source_data(
        config.licenses_path,
        config.service_reqs_path,
    )

    licenses = clean_licenses(licenses_raw)
    service_reqs = clean_service_requests(service_reqs_raw)
    licenses_expanded = expand_licenses_to_months(licenses)

    license_panel, category_columns = build_license_panel(
        licenses,
        licenses_expanded,
    )

    business_locations = compute_business_locations(licenses)
    business_locations, _cluster_centers = assign_location_clusters(
        business_locations,
        config.location_k,
    )

    req_business = radius_join_requests_to_businesses(
        business_locations,
        service_reqs,
        config.radius_radians,
    )

    complaint_panel, complaint_columns = build_complaint_panel(req_business)

    joined = merge_final_dataset(
        license_panel=license_panel,
        complaint_panel=complaint_panel,
        business_locations=business_locations,
        category_columns=category_columns,
        complaint_columns=complaint_columns,
    )

    return joined


def save_joined_dataset(joined: pd.DataFrame, output_path: Path) -> Path:
    """Save the final dataset to CSV and return the output path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output_path, index=False)
    return output_path


def run_pipeline(config: PipelineConfig) -> Path:
    """Execute preprocessing pipeline end-to-end and save output."""
    joined = build_joined_dataset(config)
    return save_joined_dataset(joined, config.output_path)


def parse_args() -> PipelineConfig:
    """Parse command-line arguments into a PipelineConfig."""
    parser = argparse.ArgumentParser(
        description="Build joined NYC business + 311 monthly panel dataset."
    )
    parser.add_argument(
        "--licenses",
        type=Path,
        required=True,
        help="Path to licenses.csv",
    )
    parser.add_argument(
        "--service-reqs",
        type=Path,
        required=True,
        help="Path to service_reqs.csv or service_reqs_sample.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write joined_dataset.csv",
    )
    parser.add_argument(
        "--location-k",
        type=int,
        default=LOCATION_K,
        help="Number of KMeans location clusters",
    )
    parser.add_argument(
        "--radius-meters",
        type=float,
        default=RADIUS_METERS,
        help="Radius in meters for joining 311 requests to nearby businesses",
    )

    args = parser.parse_args()
    radius_radians = args.radius_meters / EARTH_RADIUS_METERS

    return PipelineConfig(
        licenses_path=args.licenses,
        service_reqs_path=args.service_reqs,
        output_path=args.output,
        location_k=args.location_k,
        radius_radians=radius_radians,
    )


def main() -> None:
    """Entry point for script execution."""
    config = parse_args()
    output_path = run_pipeline(config)
    print(f"Saved joined dataset to {output_path}")


if __name__ == "__main__":
    main()
