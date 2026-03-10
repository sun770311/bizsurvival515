"""Create a business-level Mapbox GeoJSON file from joined panel and license data.

Input:
- joined_dataset.csv
- licenses.csv

Processing steps:
- Load and validate joined panel and license data
- Clean business IDs, contact fields, and coordinates
- Filter license records to valid NYC boroughs and coordinates
- Aggregate joined panel to one row per business_id
- Aggregate license metadata into a list per business_id
- Use the first valid license latitude/longitude per business_id

Output:
- businesses.geojson
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFF_DATE = pd.Timestamp("2026-03-01")

# Conservative NYC bounding box
NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68

VALID_BOROUGHS = {
    "Manhattan",
    "Brooklyn",
    "Queens",
    "Bronx",
    "Staten Island",
}


@dataclass(frozen=True)
class GeoJSONConfig:
    """Configuration for business-level GeoJSON export."""

    joined_data_path: Path
    licenses_path: Path
    output_path: Path
    cutoff_date: pd.Timestamp = CUTOFF_DATE
    lat_min: float = NYC_LAT_MIN
    lat_max: float = NYC_LAT_MAX
    lng_min: float = NYC_LNG_MIN
    lng_max: float = NYC_LNG_MAX


def load_joined_dataset(joined_data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse panel month."""
    joined = pd.read_csv(joined_data_path)
    joined["business_id"] = joined["business_id"].astype("string").str.strip()
    joined["business_id"] = joined["business_id"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def load_licenses_dataset(licenses_path: Path) -> pd.DataFrame:
    """Load licenses dataset."""
    return pd.read_csv(licenses_path)


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate required joined panel columns."""
    required_columns = [
        "business_id",
        "month",
        "complaint_sum",
        "open",
    ]
    missing_columns = [
        column for column in required_columns if column not in joined.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required joined columns: {missing_columns}")

    if joined["month"].isna().any():
        raise ValueError("Some joined month values could not be parsed as datetimes.")


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
        "Business Name",
        "Contact Phone",
        "Building Number",
        "Street1",
        "Street2",
        "Street3",
        "Apt/Suite",
        "City",
        "State",
        "ZIP Code",
        "Borough",
        "License Number",
        "License Type",
        "License Status",
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
        "Building Number",
        "Street1",
        "Street2",
        "Street3",
        "Apt/Suite",
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
        street_address
        + ", "
        + city_part
        + " "
        + state_part
        + " "
        + zip_part
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
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
) -> pd.DataFrame:
    """Keep only license rows with valid NYC coordinates."""
    filtered = licenses.dropna(
        subset=["Business Unique ID", "Latitude", "Longitude"]
    ).copy()

    filtered = filtered[
        filtered["Latitude"].between(lat_min, lat_max)
        & filtered["Longitude"].between(lng_min, lng_max)
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
        "Business Unique ID",
        "Business Name",
        "full_address",
        "Borough",
        "Contact Phone",
        "License Number",
        "License Type",
        "License Status",
        "Business Category",
        "Latitude",
        "Longitude",
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
    """Aggregate license metadata to one row per business_id.

    Uses the first valid NYC coordinate pair found for each business_id and keeps
    the full list of license-associated names/addresses/contact fields for popup use.
    """
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
            "active": (
                int(row["active"])
                if not pd.isna(row.get("active"))
                else None
            ),
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


def build_geojson(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap features in a GeoJSON FeatureCollection."""
    return {
        "type": "FeatureCollection",
        "features": features,
    }


def save_geojson(geojson: dict[str, Any], output_path: Path) -> Path:
    """Save GeoJSON to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(geojson, file_obj, indent=2, allow_nan=False)

    return output_path


def run_geojson_pipeline(config: GeoJSONConfig) -> Path:
    """Run end-to-end business-level GeoJSON export pipeline."""
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
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lng_min=config.lng_min,
        lng_max=config.lng_max,
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
    geojson = build_geojson(features)

    return save_geojson(geojson, config.output_path)


def main() -> None:
    """Entry point for script execution."""
    config = GeoJSONConfig(
        joined_data_path=Path("/content/drive/MyDrive/joined_dataset.csv"),
        licenses_path=Path("/content/drive/MyDrive/licenses.csv"),
        output_path=Path("/content/drive/MyDrive/businesses.geojson"),
    )
    run_geojson_pipeline(config)


if __name__ == "__main__":
    main()