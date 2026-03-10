"""Preprocess data for the business survival pipeline."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from pipeline.utils import save_dataframe_artifact


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration settings for the data processing pipeline."""

    licenses_path: Path
    service_reqs_path: Path
    output_path: Path
    location_k: int = 25
    radius_radians: float = 50.0 / 6_371_000


def sanitize_feature_name(name: str, prefix: str) -> str:
    """Sanitize strings to use as dataframe column features."""
    cleaned = str(name).lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return f"{prefix}_{cleaned}"


def make_unique_column_names(columns: list[str]) -> list[str]:
    """Ensure all column names are unique by appending numeric suffixes."""
    seen: dict[str, int] = {}
    unique_columns = []

    for col in columns:
        if col not in seen:
            seen[col] = 1
            unique_columns.append(col)
        else:
            new_col = f"{col}_{seen[col]}"
            seen[col] += 1
            unique_columns.append(new_col)

    return unique_columns


def month_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Generate a sequence of inclusive month-start timestamps."""
    return pd.date_range(
        start=start.replace(day=1),
        end=end.replace(day=1),
        freq="MS",
    )


def clean_licenses(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, filter, and clean raw license data."""
    cleaned = df.copy()

    cleaned["Business Unique ID"] = cleaned["Business Unique ID"].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["Business Unique ID"])

    cleaned["Initial Issuance Date"] = pd.to_datetime(
        cleaned["Initial Issuance Date"], format="%m/%d/%Y", errors="coerce"
    )
    cleaned["Expiration Date"] = pd.to_datetime(
        cleaned["Expiration Date"], format="%m/%d/%Y", errors="coerce"
    )

    cleaned = cleaned.dropna(subset=["Initial Issuance Date"])
    cleaned = cleaned[
        cleaned["Expiration Date"].isna()
        | (cleaned["Expiration Date"] >= cleaned["Initial Issuance Date"])
    ]

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    return cleaned


def clean_service_requests(df: pd.DataFrame) -> pd.DataFrame:
    """Parse, filter, and clean raw 311 service request data."""
    cleaned = df.copy()

    cleaned = cleaned.dropna(subset=["Unique Key"])
    cleaned["Unique Key"] = cleaned["Unique Key"].astype("string").str.strip()

    cleaned["Created Date"] = pd.to_datetime(
        cleaned["Created Date"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
    )
    cleaned = cleaned.dropna(subset=["Created Date"])
    cleaned["month"] = cleaned["Created Date"].dt.to_period("M").dt.to_timestamp()

    cleaned["Problem (formerly Complaint Type)"] = (
        cleaned["Problem (formerly Complaint Type)"].astype("string").str.strip()
    )
    cleaned = cleaned[cleaned["Problem (formerly Complaint Type)"] != ""]

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    return cleaned


def build_joined_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Build the final joined business panel dataset."""
    licenses_df = pd.read_csv(config.licenses_path, dtype={"Business Unique ID": "string"})
    reqs_df = pd.read_csv(config.service_reqs_path, dtype={"Unique Key": "string"})

    licenses = clean_licenses(licenses_df)
    reqs = clean_service_requests(reqs_df)

    business_months = []
    for _, row in licenses.iterrows():
        bus_id = row["Business Unique ID"]
        start_m = row["Initial Issuance Date"].replace(day=1)
        end_m = row["Expiration Date"]
        if pd.isna(end_m):
            end_m = pd.Timestamp.now()
        end_m = end_m.replace(day=1)

        category = "unknown"
        if pd.notna(row.get("Business Category")):
            category = row["Business Category"]

        for month in month_range(start_m, end_m):
            business_months.append({
                "business_id": bus_id,
                "month": month,
                "business_latitude": row["Latitude"],
                "business_longitude": row["Longitude"],
                "business_category": category,
            })

    panel = pd.DataFrame(business_months)

    if panel.empty:
        return pd.DataFrame()

    panel_agg = (
        panel.groupby(["business_id", "month"], as_index=False)
        .agg(
            active_license_count=("business_category", "count"),
            business_latitude=("business_latitude", "first"),
            business_longitude=("business_longitude", "first"),
            category_list=("business_category", list),
        )
    )

    first_license = panel_agg.groupby("business_id")["month"].transform("min")
    panel_agg["months_since_first_license"] = (
        (panel_agg["month"].dt.year - first_license.dt.year) * 12
        + (panel_agg["month"].dt.month - first_license.dt.month)
    )

    panel_agg["open"] = 1

    coords = panel_agg[["business_latitude", "business_longitude"]].dropna()
    if not coords.empty and len(coords) >= config.location_k:
        kmeans = KMeans(n_clusters=config.location_k, random_state=42, n_init=10)
        panel_agg.loc[coords.index, "location_cluster"] = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_
        panel_agg.loc[coords.index, "location_cluster_lat"] = centers[
            panel_agg.loc[coords.index, "location_cluster"].astype(int), 0
        ]
        panel_agg.loc[coords.index, "location_cluster_lng"] = centers[
            panel_agg.loc[coords.index, "location_cluster"].astype(int), 1
        ]
    else:
        panel_agg["location_cluster"] = 0
        panel_agg["location_cluster_lat"] = 0.0
        panel_agg["location_cluster_lng"] = 0.0

    category_dummies = panel_agg["category_list"].explode().to_frame()
    category_dummies["value"] = 1
    category_pivot = (
        category_dummies.pivot_table(
            index=category_dummies.index,
            columns="category_list",
            values="value",
            fill_value=0,
        )
    )
    category_pivot.columns = [
        sanitize_feature_name(c, "business_category") for c in category_pivot.columns
    ]

    panel_with_cats = pd.concat([panel_agg, category_pivot], axis=1)

    reqs_agg = (
        reqs.groupby(["month", "Problem (formerly Complaint Type)"])
        .size()
        .reset_index(name="complaint_count")
    )
    reqs_agg["complaint_type"] = reqs_agg["Problem (formerly Complaint Type)"].apply(
        lambda x: sanitize_feature_name(x, "complaint_type")
    )

    complaint_pivot = reqs_agg.pivot_table(
        index="month",
        columns="complaint_type",
        values="complaint_count",
        fill_value=0,
    ).reset_index()

    final_panel = panel_with_cats.merge(complaint_pivot, on="month", how="left")

    complaint_cols = [c for c in complaint_pivot.columns if c != "month"]
    for col in complaint_cols:
        final_panel[col] = final_panel[col].fillna(0)

    category_cols = [c for c in category_pivot.columns]
    final_panel["business_category_sum"] = final_panel[category_cols].sum(axis=1)
    final_panel["complaint_sum"] = final_panel[complaint_cols].sum(axis=1)
    final_panel["total_311"] = final_panel["complaint_sum"]

    final_panel.columns = make_unique_column_names(final_panel.columns.tolist())

    return final_panel.drop(columns=["category_list"])


def run_pipeline(config: PipelineConfig) -> Path:
    """Execute preprocessing pipeline and save to disk."""
    joined_dataset = build_joined_dataset(config)
    return save_dataframe_artifact(joined_dataset, config.output_path)


def main() -> None:
    """Entry point for script execution."""
    parser = argparse.ArgumentParser(description="Preprocess pipeline.")
    parser.add_argument("--licenses-path", type=Path, required=True)
    parser.add_argument("--service-reqs-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    config = PipelineConfig(
        licenses_path=args.licenses_path,
        service_reqs_path=args.service_reqs_path,
        output_path=args.output_path,
    )
    output_path = run_pipeline(config)
    print(f"Saved dataset to {output_path}")


if __name__ == "__main__":
    main()
