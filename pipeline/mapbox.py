"""Build GeoJSON business outputs for the Mapbox visualization pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

VALID_BOROUGHS = {
    "Bronx",
    "Brooklyn",
    "Manhattan",
    "Queens",
    "Staten Island",
}

DEFAULT_CUTOFF_DATE = pd.Timestamp("2026-03-01")

REQUIRED_JOINED_COLUMNS = {
    "business_id",
    "month",
}

REQUIRED_LICENSE_COLUMNS = {
    "Business Unique ID",
    "Latitude",
    "Longitude",
    "Borough",
}

LICENSE_RENAME_MAP = {
    "Business Unique ID": "business_id",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Borough": "borough",
    "Business Name": "business_name",
    "Industry": "industry",
    "Address Building": "address_building",
    "Address Street Name": "address_street_name",
    "Address City": "address_city",
    "Address State": "address_state",
    "Address Zip": "address_zip",
    "License Status": "license_status",
    "License Creation Date": "license_creation_date",
    "License Expiration Date": "license_expiration_date",
    "Full Address": "full_address",
}

COMPLAINT_COUNT_CANDIDATES = [
    "complaint_sum",
    "complaint_count",
    "complaints",
    "count",
]

COMPLAINT_TYPE_CANDIDATES = [
    "complaint_type",
    "Complaint Type",
]


@dataclass(frozen=True)
class GeoBounds:
    """Latitude and longitude bounds for NYC filtering."""

    lat_min: float = 40.49
    lat_max: float = 40.92
    lng_min: float = -74.27
    lng_max: float = -73.68


@dataclass(frozen=True)
class GeoJSONConfig:
    """Configuration for the GeoJSON generation pipeline."""

    joined_data_path: Path
    licenses_path: Path
    output_path: Path
    cutoff_date: pd.Timestamp = DEFAULT_CUTOFF_DATE
    bounds: GeoBounds = field(default_factory=GeoBounds)


def load_joined_dataset(path: Path) -> pd.DataFrame:
    """Load the joined business-month dataset."""
    joined = pd.read_csv(path)
    if "month" in joined.columns:
        joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    if "business_id" in joined.columns:
        joined["business_id"] = joined["business_id"].astype(str)
    return joined


def load_licenses_dataset(path: Path) -> pd.DataFrame:
    """Load the license metadata dataset."""
    return pd.read_csv(path)


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate that the joined dataset contains required columns."""
    required = {"business_id", "month"}
    missing = sorted(required - set(joined.columns))

    if missing:
        raise ValueError(f"Missing required joined columns: {missing}")

    complaint_columns = (
        set(COMPLAINT_COUNT_CANDIDATES) |
        set(COMPLAINT_TYPE_CANDIDATES)
    )

    if not complaint_columns.intersection(joined.columns):
        raise ValueError(
            "Missing required joined columns: complaint signal column"
        )


def validate_licenses_dataset(licenses: pd.DataFrame) -> None:
    """Validate that the license dataset contains required columns."""
    missing = sorted(REQUIRED_LICENSE_COLUMNS - set(licenses.columns))
    if missing:
        raise ValueError(f"Missing required license columns: {missing}")


def clean_joined_business_ids(joined: pd.DataFrame) -> pd.DataFrame:
    """Normalize business IDs in the joined dataset."""
    cleaned = joined.copy()
    cleaned["business_id"] = cleaned["business_id"].astype(str).str.strip()
    cleaned = cleaned[cleaned["business_id"] != ""].copy()
    return cleaned


def clean_license_fields(licenses: pd.DataFrame) -> pd.DataFrame:
    """Normalize relevant fields in the license dataset."""
    cleaned = licenses.copy()
    cleaned["Business Unique ID"] = cleaned["Business Unique ID"].astype(str).str.strip()

    if "Borough" in cleaned.columns:
        cleaned["Borough"] = cleaned["Borough"].astype(str).str.strip().str.title()

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    for column in ("License Creation Date", "License Expiration Date"):
        if column in cleaned.columns:
            cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce")

    return cleaned


def build_full_address(licenses: pd.DataFrame) -> pd.DataFrame:
    """Construct a full address field from available address components."""
    enriched = licenses.copy()
    address_parts = [
        "Address Building",
        "Address Street Name",
        "Address City",
        "Address State",
        "Address Zip",
    ]
    available_parts = [column for column in address_parts if column in enriched.columns]

    if not available_parts:
        enriched["Full Address"] = ""
        return enriched

    enriched["Full Address"] = (
        enriched[available_parts]
        .fillna("")
        .astype(str)
        .apply(
            lambda row: " ".join(part.strip() for part in row if part.strip()),
            axis=1,
        )
    )
    return enriched


def filter_valid_boroughs(licenses: pd.DataFrame) -> pd.DataFrame:
    """Keep only licenses belonging to the five NYC boroughs."""
    return licenses[licenses["Borough"].isin(VALID_BOROUGHS)].copy()


def filter_nyc_license_coordinates(
    licenses: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
) -> pd.DataFrame:
    """Keep only license rows whose coordinates fall within NYC bounds."""
    mask = (
        licenses["Latitude"].between(lat_min, lat_max)
        & licenses["Longitude"].between(lng_min, lng_max)
    )
    return licenses[mask].copy()


def _resolve_complaint_series(joined: pd.DataFrame) -> pd.Series:
    """Return a per-row complaint contribution series from available columns."""
    for column in COMPLAINT_COUNT_CANDIDATES:
        if column in joined.columns:
            return pd.to_numeric(joined[column], errors="coerce").fillna(0)

    for column in COMPLAINT_TYPE_CANDIDATES:
        if column in joined.columns:
            return joined[column].notna().astype(int)

    return pd.Series(0, index=joined.index, dtype="int64")


def build_business_summary(
    joined: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate business activity from the joined business-month dataset."""
    working = joined.copy()
    working["complaint_value"] = _resolve_complaint_series(working)

    summary = (
        working.groupby("business_id", as_index=False)
        .agg(
            last_month=("month", "max"),
            complaint_sum=("complaint_value", "sum"),
        )
        .copy()
    )
    summary["active"] = (summary["last_month"] >= cutoff_date).astype(int)
    return summary


def _serialize_value(value: Any) -> Any:
    """Convert pandas values into JSON-friendly Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _license_record_from_row(row: pd.Series) -> dict[str, Any]:
    """Convert a license row into a serialized record for GeoJSON properties."""
    record: dict[str, Any] = {}
    for source_column, target_column in LICENSE_RENAME_MAP.items():
        if source_column in row.index:
            record[target_column] = _serialize_value(row[source_column])
    return record


def build_business_license_metadata(licenses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate license metadata to one row per business."""
    grouped_rows: list[dict[str, Any]] = []

    for business_id, group in licenses.groupby("Business Unique ID", sort=True):
        ordered = group.sort_values("Business Unique ID", kind="stable").reset_index(
            drop=True
        )
        first_row = ordered.iloc[0]

        grouped_rows.append(
            {
                "business_id": str(business_id),
                "latitude": float(first_row["Latitude"]),
                "longitude": float(first_row["Longitude"]),
                "license_count": int(len(ordered)),
                "license_records": [
                    _license_record_from_row(record_row)
                    for _, record_row in ordered.iterrows()
                ],
            }
        )

    return pd.DataFrame(grouped_rows)


def merge_business_summary_with_license_metadata(
    summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Merge business activity summaries with coordinate/license metadata."""
    merged = summary.merge(metadata, on="business_id", how="inner")
    return merged.sort_values("business_id").reset_index(drop=True)


def build_feature(row: pd.Series) -> dict[str, Any] | None:
    """Build a single GeoJSON feature from one business row."""
    latitude = row.get("latitude")
    longitude = row.get("longitude")

    if pd.isna(latitude) or pd.isna(longitude):
        return None

    last_month = row.get("last_month")
    last_month_value = (
        last_month.strftime("%Y-%m-%d")
        if isinstance(last_month, pd.Timestamp) and not pd.isna(last_month)
        else None
    )

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(longitude), float(latitude)],
        },
        "properties": {
            "business_id": row.get("business_id"),
            "active": int(row.get("active", 0)),
            "last_month": last_month_value,
            "complaint_sum": float(row.get("complaint_sum", 0.0)),
            "license_count": int(row.get("license_count", 0)),
            "license_records": row.get("license_records", []),
        },
    }


def build_geojson_features(businesses: pd.DataFrame) -> list[dict[str, Any]]:
    """Build GeoJSON features for all business rows."""
    features: list[dict[str, Any]] = []

    for _, row in businesses.iterrows():
        feature = build_feature(row)
        if feature is not None:
            features.append(feature)

    return features


def build_geojson(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap a feature list in a GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def run_geojson_pipeline(config: GeoJSONConfig) -> Path:
    """Run the full GeoJSON generation pipeline and write the output file."""
    joined = load_joined_dataset(config.joined_data_path)
    licenses = load_licenses_dataset(config.licenses_path)

    validate_joined_dataset(joined)
    validate_licenses_dataset(licenses)

    joined = clean_joined_business_ids(joined)
    licenses = clean_license_fields(licenses)
    licenses = build_full_address(licenses)
    licenses = filter_valid_boroughs(licenses)
    licenses = filter_nyc_license_coordinates(
        licenses=licenses,
        lat_min=config.bounds.lat_min,
        lat_max=config.bounds.lat_max,
        lng_min=config.bounds.lng_min,
        lng_max=config.bounds.lng_max,
    )

    summary = build_business_summary(joined, cutoff_date=config.cutoff_date)
    metadata = build_business_license_metadata(licenses)
    businesses = merge_business_summary_with_license_metadata(summary, metadata)

    geojson = build_geojson(build_geojson_features(businesses))

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(geojson, file_obj, indent=2)

    return config.output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for GeoJSON generation."""
    parser = argparse.ArgumentParser(
        description="Build GeoJSON business outputs for the Mapbox app.",
    )
    parser.add_argument(
        "--joined-data-path",
        type=Path,
        required=True,
        help="Path to the joined business-month dataset CSV.",
    )
    parser.add_argument(
        "--licenses-path",
        type=Path,
        required=True,
        help="Path to the license metadata CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path where the GeoJSON output should be written.",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        default=str(DEFAULT_CUTOFF_DATE.date()),
        help="Cutoff date used to define active businesses.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the GeoJSON pipeline from the command line."""
    args = parse_args()
    config = GeoJSONConfig(
        joined_data_path=args.joined_data_path,
        licenses_path=args.licenses_path,
        output_path=args.output_path,
        cutoff_date=pd.Timestamp(args.cutoff_date),
    )
    run_geojson_pipeline(config)


if __name__ == "__main__":
    main()
