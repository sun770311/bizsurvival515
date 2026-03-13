"""Preprocess data for the business survival pipeline."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

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
    cleaned = cleaned.replace("&", "and")
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


def _build_base_panel(licenses: pd.DataFrame) -> pd.DataFrame:
    """Generate the base business-month panel from license lifetimes."""
    business_months = []
    for _, row in licenses.iterrows():
        bus_id = row["Business Unique ID"]
        start_m = row["Initial Issuance Date"].replace(day=1)
        end_m = row["Expiration Date"]
        if pd.isna(end_m):
            end_m = pd.Timestamp.now()
        end_m = end_m.replace(day=1)

        category = row.get("Business Category", "unknown")
        if pd.isna(category):
            category = "unknown"

        for month in month_range(start_m, end_m):
            business_months.append({
                "business_id": bus_id,
                "month": month,
                "business_latitude": row["Latitude"],
                "business_longitude": row["Longitude"],
                "business_category": category,
            })

    if not business_months:
        return pd.DataFrame()

    panel = pd.DataFrame(business_months)
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
    return panel_agg


def _add_location_clusters(panel: pd.DataFrame, location_k: int) -> pd.DataFrame:
    """Add KMeans location clusters to the panel."""
    panel["location_cluster"] = 0
    panel["location_cluster_lat"] = 0.0
    panel["location_cluster_lng"] = 0.0

    coords = panel[["business_latitude", "business_longitude"]].dropna()
    if not coords.empty and len(coords) >= location_k:
        kmeans = KMeans(n_clusters=location_k, random_state=42, n_init=10)
        panel.loc[coords.index, "location_cluster"] = kmeans.fit_predict(coords)
        centers = kmeans.cluster_centers_
        panel.loc[coords.index, "location_cluster_lat"] = centers[
            panel.loc[coords.index, "location_cluster"].astype(int), 0
        ]
        panel.loc[coords.index, "location_cluster_lng"] = centers[
            panel.loc[coords.index, "location_cluster"].astype(int), 1
        ]

    panel["location_cluster"] = panel["location_cluster"].astype(int)
    return panel


def _add_category_dummies(panel: pd.DataFrame) -> pd.DataFrame:
    """Pivot categories into dummy flags and attach to panel."""
    dummies = panel["category_list"].explode().to_frame()
    dummies["value"] = 1
    pivot = dummies.pivot_table(
        index=dummies.index, columns="category_list", values="value", fill_value=0
    )
    pivot.columns = [sanitize_feature_name(c, "business_category") for c in pivot.columns]

    combined = pd.concat([panel, pivot], axis=1)
    combined["business_category_sum"] = combined[list(pivot.columns)].sum(axis=1)
    return combined


def _process_complaints(reqs: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot 311 requests by month into dummy flags."""
    reqs_agg = (
        reqs.groupby(["month", "Problem (formerly Complaint Type)"])
        .size()
        .reset_index(name="complaint_count")
    )
    reqs_agg["complaint_type"] = reqs_agg["Problem (formerly Complaint Type)"].apply(
        lambda x: sanitize_feature_name(x, "complaint_type")
    )

    pivot = reqs_agg.pivot_table(
        index="month", columns="complaint_type", values="complaint_count", fill_value=0
    ).reset_index()

    cols = [c for c in pivot.columns if c != "month"]
    return pivot, cols


def build_joined_dataset(config: PipelineConfig) -> pd.DataFrame:
    """Build the final joined business panel dataset."""
    licenses = clean_licenses(
        pd.read_csv(
            config.licenses_path,
            dtype={"Business Unique ID": "string"},
            low_memory=False,
        )
    )
    reqs = clean_service_requests(
        pd.read_csv(
            config.service_reqs_path,
            dtype={"Unique Key": "string"},
            low_memory=False,
        )
    )

    panel_agg = _build_base_panel(licenses)
    if panel_agg.empty:
        return pd.DataFrame()

    panel_agg = _add_location_clusters(panel_agg, config.location_k)
    panel_with_cats = _add_category_dummies(panel_agg)

    complaint_pivot, complaint_cols = _process_complaints(reqs)
    final_panel = panel_with_cats.merge(complaint_pivot, on="month", how="left")
    final_panel = final_panel.copy()

    for col in complaint_cols:
        final_panel[col] = final_panel[col].fillna(0)

    final_panel["complaint_sum"] = final_panel[complaint_cols].sum(axis=1)
    final_panel["total_311"] = final_panel["complaint_sum"]

    final_panel.columns = make_unique_column_names(list(final_panel.columns))
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
