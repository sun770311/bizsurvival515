"""Validation tests for business-level GeoJSON output from mapbox.py.

This script validates that businesses.geojson is structurally correct and
consistent with the source joined_dataset.csv and licenses.csv used to build it.

Inputs:
- businesses.geojson
- joined_dataset.csv
- licenses.csv

Checks:
- GeoJSON is a valid FeatureCollection
- Each feature is a Point with valid NYC coordinates
- Each feature has a business_id and license_records list
- All borough values in license_records are one of the five valid NYC boroughs
- Active/inactive status matches the joined panel relative to March 2026
- last_month in GeoJSON matches the joined panel business summary
- complaint_sum in GeoJSON matches the joined panel business summary
- GeoJSON contains at most one feature per business_id
- GeoJSON coordinates match the first valid filtered license coordinate per business_id
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CUTOFF_DATE = pd.Timestamp("2026-03-01")

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
class MapboxTestConfig:
    """Configuration for GeoJSON validation tests."""

    geojson_path: Path
    joined_data_path: Path
    licenses_path: Path
    cutoff_date: pd.Timestamp = CUTOFF_DATE
    lat_min: float = NYC_LAT_MIN
    lat_max: float = NYC_LAT_MAX
    lng_min: float = NYC_LNG_MIN
    lng_max: float = NYC_LNG_MAX


def load_geojson(geojson_path: Path) -> dict[str, Any]:
    """Load GeoJSON from disk."""
    with geojson_path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_joined_dataset(joined_data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse dates."""
    joined = pd.read_csv(joined_data_path)
    joined["business_id"] = joined["business_id"].astype("string").str.strip()
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def load_licenses_dataset(licenses_path: Path) -> pd.DataFrame:
    """Load licenses dataset."""
    return pd.read_csv(licenses_path)


def clean_license_fields(licenses: pd.DataFrame) -> pd.DataFrame:
    """Clean merge keys, boroughs, and coordinates like mapbox.py."""
    cleaned = licenses.copy()

    cleaned["Business Unique ID"] = (
        cleaned["Business Unique ID"].astype("string").str.strip()
    )
    cleaned["Business Unique ID"] = cleaned["Business Unique ID"].replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA}
    )

    if "Borough" in cleaned.columns:
        cleaned["Borough"] = cleaned["Borough"].astype("string").str.strip()
        cleaned["Borough"] = cleaned["Borough"].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )

    cleaned["Latitude"] = pd.to_numeric(cleaned["Latitude"], errors="coerce")
    cleaned["Longitude"] = pd.to_numeric(cleaned["Longitude"], errors="coerce")

    return cleaned


def filter_valid_boroughs(licenses: pd.DataFrame) -> pd.DataFrame:
    """Filter to the five valid NYC boroughs."""
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
    """Filter to valid NYC coordinate rows."""
    filtered = licenses.dropna(
        subset=["Business Unique ID", "Latitude", "Longitude"]
    ).copy()

    filtered = filtered[
        filtered["Latitude"].between(lat_min, lat_max)
        & filtered["Longitude"].between(lng_min, lng_max)
    ].copy()

    return filtered


def build_expected_business_summary(
    joined: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Rebuild expected business summary from joined panel."""
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


def build_expected_first_license_coordinates(licenses: pd.DataFrame) -> pd.DataFrame:
    """Rebuild first valid filtered license coordinate per business_id."""
    expected = (
        licenses.sort_values(["Business Unique ID"])
        .drop_duplicates(subset=["Business Unique ID"], keep="first")
        [["Business Unique ID", "Latitude", "Longitude"]]
        .rename(
            columns={
                "Business Unique ID": "business_id",
                "Latitude": "latitude",
                "Longitude": "longitude",
            }
        )
        .reset_index(drop=True)
    )
    return expected


def flatten_geojson_features(geojson: dict[str, Any]) -> pd.DataFrame:
    """Convert GeoJSON features into a flat dataframe for testing."""
    rows: list[dict[str, Any]] = []

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        geometry = feature.get("geometry", {})
        coordinates = geometry.get("coordinates", [None, None])

        longitude = coordinates[0] if len(coordinates) > 0 else None
        latitude = coordinates[1] if len(coordinates) > 1 else None

        rows.append(
            {
                "business_id": properties.get("business_id"),
                "active": properties.get("active"),
                "last_month": properties.get("last_month"),
                "complaint_sum": properties.get("complaint_sum"),
                "license_count": properties.get("license_count"),
                "license_records": properties.get("license_records"),
                "geometry_type": geometry.get("type"),
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    flattened = pd.DataFrame(rows)
    if not flattened.empty and "last_month" in flattened.columns:
        flattened["last_month"] = pd.to_datetime(flattened["last_month"], errors="coerce")
    return flattened


def assert_valid_geojson_root(geojson: dict[str, Any]) -> None:
    """Assert GeoJSON root object is a valid FeatureCollection."""
    if geojson.get("type") != "FeatureCollection":
        raise AssertionError("GeoJSON root type is not FeatureCollection.")

    if not isinstance(geojson.get("features"), list):
        raise AssertionError("GeoJSON features is not a list.")


def assert_unique_business_ids(features_df: pd.DataFrame) -> None:
    """Assert at most one feature per business_id."""
    if features_df["business_id"].isna().any():
        raise AssertionError("Some GeoJSON features are missing business_id.")

    duplicates = features_df["business_id"].duplicated()
    if duplicates.any():
        dup_ids = features_df.loc[duplicates, "business_id"].tolist()[:10]
        raise AssertionError(f"Duplicate business_id features found: {dup_ids}")


def assert_valid_geometry_and_coordinates(
    features_df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lng_min: float,
    lng_max: float,
) -> None:
    """Assert valid Point geometry and coordinates within NYC bounds."""
    if not (features_df["geometry_type"] == "Point").all():
        raise AssertionError("Some GeoJSON features do not have Point geometry.")

    if features_df["latitude"].isna().any() or features_df["longitude"].isna().any():
        raise AssertionError("Some GeoJSON features have missing coordinates.")

    if not features_df["latitude"].between(lat_min, lat_max).all():
        raise AssertionError("Some GeoJSON latitudes are outside NYC bounds.")

    if not features_df["longitude"].between(lng_min, lng_max).all():
        raise AssertionError("Some GeoJSON longitudes are outside NYC bounds.")


def assert_valid_license_records(features_df: pd.DataFrame) -> None:
    """Assert each feature has a valid license_records list and valid boroughs."""
    for _, row in features_df.iterrows():
        license_records = row["license_records"]

        if not isinstance(license_records, list):
            raise AssertionError(
                f"license_records is not a list for business_id={row['business_id']}."
            )

        for record in license_records:
            if not isinstance(record, dict):
                raise AssertionError(
                    f"license_records contains a non-dict record for "
                    f"business_id={row['business_id']}."
                )

            borough = record.get("borough")
            if borough is not None and borough not in VALID_BOROUGHS:
                raise AssertionError(
                    f"Invalid borough {borough!r} found in license_records for "
                    f"business_id={row['business_id']}."
                )


def assert_active_logic_and_last_month(
    features_df: pd.DataFrame,
    expected_summary: pd.DataFrame,
) -> None:
    """Assert GeoJSON active and last_month match joined panel business summary."""
    merged = features_df.merge(
        expected_summary[["business_id", "active", "last_month"]],
        on="business_id",
        how="left",
        suffixes=("_geojson", "_expected"),
    )

    if merged["active_expected"].isna().any():
        missing_ids = merged.loc[merged["active_expected"].isna(), "business_id"].tolist()[:10]
        raise AssertionError(
            f"Some GeoJSON business_ids are missing from joined summary: {missing_ids}"
        )

    active_match = merged["active_geojson"].astype(int) == merged["active_expected"].astype(int)
    if not active_match.all():
        bad_ids = merged.loc[~active_match, "business_id"].tolist()[:10]
        raise AssertionError(
            f"Active/inactive logic mismatch for business_ids: {bad_ids}"
        )

    last_month_match = merged["last_month_geojson"] == merged["last_month_expected"]
    if not last_month_match.all():
        bad_ids = merged.loc[~last_month_match, "business_id"].tolist()[:10]
        raise AssertionError(
            f"last_month mismatch for business_ids: {bad_ids}"
        )


def assert_complaint_sum_matches(
    features_df: pd.DataFrame,
    expected_summary: pd.DataFrame,
) -> None:
    """Assert GeoJSON complaint_sum matches joined panel business summary."""
    merged = features_df.merge(
        expected_summary[["business_id", "complaint_sum"]],
        on="business_id",
        how="left",
        suffixes=("_geojson", "_expected"),
    )

    if merged["complaint_sum_expected"].isna().any():
        missing_ids = merged.loc[
            merged["complaint_sum_expected"].isna(), "business_id"
        ].tolist()[:10]
        raise AssertionError(
            f"Some GeoJSON business_ids are missing complaint_sum in joined summary: "
            f"{missing_ids}"
        )

    complaint_match = (
        pd.to_numeric(merged["complaint_sum_geojson"], errors="coerce")
        == pd.to_numeric(merged["complaint_sum_expected"], errors="coerce")
    )
    if not complaint_match.all():
        bad_ids = merged.loc[~complaint_match, "business_id"].tolist()[:10]
        raise AssertionError(
            f"complaint_sum mismatch for business_ids: {bad_ids}"
        )


def assert_coordinates_match_expected_first_license(
    features_df: pd.DataFrame,
    expected_coords: pd.DataFrame,
) -> None:
    """Assert GeoJSON coordinates match first filtered license coordinates."""
    merged = features_df.merge(
        expected_coords,
        on="business_id",
        how="left",
        suffixes=("_geojson", "_expected"),
    )

    if merged["latitude_expected"].isna().any() or merged["longitude_expected"].isna().any():
        missing_ids = merged.loc[
            merged["latitude_expected"].isna() | merged["longitude_expected"].isna(),
            "business_id",
        ].tolist()[:10]
        raise AssertionError(
            f"Some GeoJSON business_ids are missing expected filtered license "
            f"coordinates: {missing_ids}"
        )

    lat_match = merged["latitude_geojson"].round(8) == merged["latitude_expected"].round(8)
    lng_match = merged["longitude_geojson"].round(8) == merged["longitude_expected"].round(8)

    if not lat_match.all() or not lng_match.all():
        bad_ids = merged.loc[~(lat_match & lng_match), "business_id"].tolist()[:10]
        raise AssertionError(
            f"Coordinate mismatch for business_ids: {bad_ids}"
        )


def run_mapbox_tests(config: MapboxTestConfig) -> None:
    """Run all GeoJSON output validation checks."""
    geojson = load_geojson(config.geojson_path)
    joined = load_joined_dataset(config.joined_data_path)
    licenses = load_licenses_dataset(config.licenses_path)

    licenses = clean_license_fields(licenses)
    licenses = filter_valid_boroughs(licenses)
    licenses = filter_nyc_license_coordinates(
        licenses=licenses,
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lng_min=config.lng_min,
        lng_max=config.lng_max,
    )

    expected_summary = build_expected_business_summary(
        joined=joined,
        cutoff_date=config.cutoff_date,
    )
    expected_coords = build_expected_first_license_coordinates(licenses)
    features_df = flatten_geojson_features(geojson)

    assert_valid_geojson_root(geojson)
    assert_unique_business_ids(features_df)
    assert_valid_geometry_and_coordinates(
        features_df=features_df,
        lat_min=config.lat_min,
        lat_max=config.lat_max,
        lng_min=config.lng_min,
        lng_max=config.lng_max,
    )
    assert_valid_license_records(features_df)
    assert_active_logic_and_last_month(
        features_df=features_df,
        expected_summary=expected_summary,
    )
    assert_complaint_sum_matches(
        features_df=features_df,
        expected_summary=expected_summary,
    )
    assert_coordinates_match_expected_first_license(
        features_df=features_df,
        expected_coords=expected_coords,
    )


def main() -> None:
    """Execute GeoJSON validation tests."""
    config = MapboxTestConfig(
        geojson_path=Path("/content/drive/MyDrive/businesses.geojson"),
        joined_data_path=Path("/content/drive/MyDrive/joined_dataset.csv"),
        licenses_path=Path("/content/drive/MyDrive/licenses.csv"),
    )
    run_mapbox_tests(config)
    print("All mapbox GeoJSON checks passed.")


if __name__ == "__main__":
    main()