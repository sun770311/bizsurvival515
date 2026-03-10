"""Create a business-level Mapbox GeoJSON file from joined panel and license data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.utils import (
    CUTOFF_DATE, VALID_BOROUGHS, NYC_BBOX, BoundingBox,
    load_joined_dataset, validate_joined_dataset, save_json_artifact
)


@dataclass(frozen=True)
class GeoJSONConfig:
    """Configuration for business-level GeoJSON export."""
    joined_data_path: Path
    licenses_path: Path
    output_path: Path
    cutoff_date: pd.Timestamp = CUTOFF_DATE
    bbox: BoundingBox = NYC_BBOX


def load_licenses_dataset(licenses_path: Path) -> pd.DataFrame:
    """Load licenses dataset."""
    return pd.read_csv(licenses_path)


def validate_licenses_dataset(licenses: pd.DataFrame) -> None:
    """Validate required license columns."""
    required_columns = [
        "Business Unique ID",
        "Business Name",
        "Contact Phone",
        "Latitude",
        "Longitude",
    ]
    missing_columns = [
        column for column in required_columns if column not in licenses.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required license columns: {missing_columns}")


def clean_joined_business_ids(joined: pd.DataFrame) -> pd.DataFrame:
    """Clean joined business ID column."""
    cleaned = joined.copy()
    cleaned["business_id"] = cleaned["business_id"].astype("string").str.strip()
    cleaned["business_id"] = cleaned["business_id"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )
    return cleaned


def clean_license_fields(licenses: pd.DataFrame) -> pd.DataFrame:
    """Clean merge keys, display fields, and coordinates in license data."""
    cleaned = licenses.copy()

    cleaned["Business Unique ID"] = (
        cleaned["Business Unique ID"].astype("string").str.strip()
    )
    cleaned["Business Unique ID"] = cleaned["Business Unique ID"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )

    string_columns = [
        "Business Name", "Contact Phone", "Building Number", "Street1",
        "Street2", "Street3", "Apt/Suite", "City", "State", "ZIP Code",
        "Borough", "License Number", "License Type", "License Status",
        "Business Category",
    ]

    for column in string_columns:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip()
            cleaned[column] = cleaned[column].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA}
            )

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    return cleaned


def build_full_address(licenses: pd.DataFrame) -> pd.DataFrame:
    """Construct a human-readable full address from available license columns."""
    enriched = licenses.copy()

    address_parts = [
        "Building Number", "Street1", "Street2", "Street3", "Apt/Suite",
    ]
    existing_address_parts = [
        column for column in address_parts if column in enriched.columns
    ]

    if existing_address_parts:
        street_address = (
            enriched[existing_address_parts]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        street_address = pd.Series("", index=enriched.index, dtype="object")

    city_part = (
        enriched["City"].fillna("").astype(str).str.strip()
        if "City" in enriched.columns
        else pd.Series("", index=enriched.index, dtype="object")
    )
    state_part = (
        enriched["State"].fillna("").astype(str).str.strip()
        if "State" in enriched.columns
        else pd.Series("", index=enriched.index, dtype="object")
    )
    zip_part = (
        enriched["ZIP Code"].fillna("").astype(str).str.strip()
        if "ZIP Code" in enriched.columns
        else pd.Series("", index=enriched.index, dtype="object")
    )

    full_address = (
        street_address + ", " + city_part + " " + state_part + " " + zip_part
    ).str.replace(r"\s+", " ", regex=True).str.strip(" ,")

    enriched["full_address"] = full_address.replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )

    return enriched


def filter_valid_boroughs(licenses: pd.DataFrame) -> pd.DataFrame:
    """Keep only license rows in the five valid NYC boroughs."""
    if "Borough" not in licenses.columns:
        return licenses.copy()

    filtered = licenses.dropna(subset=["Borough"]).copy()
    filtered = filtered[filtered["Borough"].isin(VALID_BOROUGHS)].copy()
    return filtered


def filter_nyc_license_coordinates(
    licenses: pd.DataFrame,
    bbox: BoundingBox,
) -> pd.DataFrame:
    """Keep only license rows with valid NYC coordinates."""
    filtered = licenses.dropna(
        subset=["Business Unique ID", "Latitude", "Longitude"]
    ).copy()

    filtered = filtered[
        filtered["Latitude"].between(bbox.lat_min, bbox.lat_max)
        & filtered["Longitude"].between(bbox.lng_min, bbox.lng_max)
    ].copy()

    return filtered


def build_business_summary(
    joined: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate joined panel to one row per business_id."""
    summary = (
        joined.dropna(subset=["business_id"])
        .sort_values(["business_id", "month"])
        .groupby("business_id", as_index=False)
        .agg(
            complaint_sum=("complaint_sum", "sum"),
            last_month=("month", "max"),
            last_open=("open", "last"),
        )
    )

    summary["active"] = (summary["last_month"] >= cutoff_date).astype(int)
    return summary


def deduplicate_license_records(licenses: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate license rows for cleaner popup output."""
    candidate_columns = [
        "Business Unique ID", "Business Name", "full_address", "Borough",
        "Contact Phone", "License Number", "License Type", "License Status",
        "Business Category", "Latitude", "Longitude",
    ]
    existing_columns = [
        column for column in candidate_columns if column in licenses.columns
    ]

    deduped = (
        licenses[existing_columns]
        .drop_duplicates()
        .sort_values(["Business Unique ID", "Business Name", "License Number"])
        .reset_index(drop=True)
    )
    return deduped


def convert_value(value: Any) -> Any:
    """Convert pandas values into JSON-serializable Python values."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def build_license_list_for_business(group: pd.DataFrame) -> list[dict[str, Any]]:
    """Build popup-friendly license records for one business_id."""
    licenses_list: list[dict[str, Any]] = []

    for _, row in group.iterrows():
        licenses_list.append(
            {
                "business_name": convert_value(row.get("Business Name")),
                "address": convert_value(row.get("full_address")),
                "borough": convert_value(row.get("Borough")),
                "contact_phone": convert_value(row.get("Contact Phone")),
                "license_number": convert_value(row.get("License Number")),
                "license_type": convert_value(row.get("License Type")),
                "license_status": convert_value(row.get("License Status")),
                "business_category": convert_value(row.get("Business Category")),
            }
        )

    return licenses_list


def build_business_license_metadata(licenses: pd.DataFrame) -> pd.DataFrame:
    """Aggregate license metadata to one row per business_id."""
    deduped = deduplicate_license_records(licenses)
    metadata_rows: list[dict[str, Any]] = []

    for business_id, group in deduped.groupby("Business Unique ID", dropna=True):
        first_row = group.iloc[0]

        metadata_rows.append(
            {
                "business_id": business_id,
                "latitude": float(first_row["Latitude"]),
                "longitude": float(first_row["Longitude"]),
                "license_count": int(len(group)),
                "license_records": build_license_list_for_business(group),
            }
        )

    return pd.DataFrame(metadata_rows)


def merge_business_summary_with_license_metadata(
    business_summary: pd.DataFrame,
    license_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Merge business summary with business-level license metadata."""
    return business_summary.merge(
        license_metadata,
        on="business_id",
        how="left",
    )


def build_feature(row: pd.Series) -> dict[str, Any] | None:
    """Convert one business row into a GeoJSON feature."""
    latitude = row.get("latitude")
    longitude = row.get("longitude")

    if pd.isna(latitude) or pd.isna(longitude):
        return None

    license_records = row.get("license_records")
    if not isinstance(license_records, list):
        license_records = []

    return {
        "type": "Feature",
        "properties": {
            "business_id": convert_value(row.get("business_id")),
            "active": int(row["active"]) if not pd.isna(row.get("active")) else None,
            "last_month": convert_value(row.get("last_month")),
            "complaint_sum": convert_value(row.get("complaint_sum")),
            "license_count": convert_value(row.get("license_count")),
            "license_records": license_records,
        },
        "geometry": {
            "type": "Point",
            "coordinates": [float(longitude), float(latitude)],
        },
    }


def build_geojson_features(businesses: pd.DataFrame) -> list[dict[str, Any]]:
    """Build GeoJSON feature list from merged business records."""
    features: list[dict[str, Any]] = []
    for _, row in businesses.iterrows():
        feature = build_feature(row)
        if feature is not None:
            features.append(feature)
    return features


def run_geojson_pipeline(config: GeoJSONConfig) -> Path:
    """Run end-to-end business-level GeoJSON export pipeline."""
    joined = load_joined_dataset(config.joined_data_path)
    licenses = load_licenses_dataset(config.licenses_path)

    validate_joined_dataset(joined, ["business_id", "month", "complaint_sum", "open"])
    validate_licenses_dataset(licenses)

    joined = clean_joined_business_ids(joined)
    licenses = clean_license_fields(licenses)
    licenses = build_full_address(licenses)
    licenses = filter_valid_boroughs(licenses)
    licenses = filter_nyc_license_coordinates(
        licenses=licenses,
        bbox=config.bbox,
    )

    business_summary = build_business_summary(
        joined=joined,
        cutoff_date=config.cutoff_date,
    )
    license_metadata = build_business_license_metadata(licenses)
    businesses = merge_business_summary_with_license_metadata(
        business_summary=business_summary,
        license_metadata=license_metadata,
    )

    features = build_geojson_features(businesses)
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    return save_json_artifact(geojson, config.output_path)


def parse_args() -> GeoJSONConfig:
    """Parse CLI arguments into a GeoJSONConfig."""
    parser = argparse.ArgumentParser(
        description="Build business-level Mapbox GeoJSON from joined panel and licenses."
    )
    parser.add_argument("--joined-data", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff-date", type=str, default=str(CUTOFF_DATE.date()))

    args = parser.parse_args()

    return GeoJSONConfig(
        joined_data_path=args.joined_data,
        licenses_path=args.licenses,
        output_path=args.output,
        cutoff_date=pd.Timestamp(args.cutoff_date),
    )


def main() -> None:
    """Entry point for script execution."""
    config = parse_args()
    output_path = run_geojson_pipeline(config)
    print(f"Saved GeoJSON to {output_path}")


if __name__ == "__main__":
    main()
