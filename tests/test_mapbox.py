"""Tests for the Mapbox GeoJSON pipeline module."""

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from pipeline.utils import load_joined_dataset, CUTOFF_DATE, VALID_BOROUGHS, NYC_BBOX
from pipeline.mapbox import (
    GeoJSONConfig,
    build_business_license_metadata,
    build_business_summary,
    build_feature,
    build_full_address,
    clean_joined_business_ids,
    clean_license_fields,
    filter_valid_boroughs,
    load_licenses_dataset,
    merge_business_summary_with_license_metadata,
    run_geojson_pipeline,
    validate_licenses_dataset,
)

TEST_DATA_DIR = Path(__file__).parent / "data"


def flatten_geojson_features(geojson: dict) -> pd.DataFrame:
    """Flatten a GeoJSON dictionary into a pandas DataFrame for testing."""
    rows = []

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


def prepare_clean_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean input datasets for Mapbox testing."""
    joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
    licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")

    joined = clean_joined_business_ids(joined)

    licenses = clean_license_fields(licenses)
    licenses = build_full_address(licenses)
    licenses = filter_valid_boroughs(licenses)

    filtered = licenses.dropna(
        subset=["Business Unique ID", "Latitude", "Longitude"]
    ).copy()
    filtered = filtered[
        filtered["Latitude"].between(NYC_BBOX.lat_min, NYC_BBOX.lat_max)
        & filtered["Longitude"].between(NYC_BBOX.lng_min, NYC_BBOX.lng_max)
    ].copy()

    return joined, filtered


class TestMapbox(unittest.TestCase):
    """Test suite for Mapbox GeoJSON generation pipeline."""

    def test_validate_licenses_dataset_accepts_valid_data(self):
        """Test that validation succeeds on a properly formatted licenses dataset."""
        licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")
        validate_licenses_dataset(licenses)

    def test_validate_licenses_dataset_rejects_missing_required_columns(self):
        """Test that validation fails when the licenses dataset misses required columns."""
        bad = pd.DataFrame(
            {
                "Business Unique ID": ["A"],
                "Latitude": [40.7],
                "Longitude": [-73.9],
            }
        )

        with self.assertRaisesRegex(ValueError, "Missing required license columns"):
            validate_licenses_dataset(bad)

    def test_filter_valid_boroughs_keeps_only_five_boroughs(self):
        """Test that invalid boroughs are correctly filtered out."""
        licenses = pd.DataFrame(
            {
                "Business Unique ID": ["1", "2", "3"],
                "Borough": ["Queens", "Boston", "Bronx"],
                "Latitude": [40.7, 40.7, 40.8],
                "Longitude": [-73.9, -73.9, -73.8],
            }
        )

        filtered = filter_valid_boroughs(licenses)
        self.assertTrue(set(filtered["Borough"].unique()).issubset(VALID_BOROUGHS))
        self.assertNotIn("Boston", filtered["Borough"].tolist())

    def test_build_business_summary_has_expected_columns(self):
        """Test that the business summary aggregates expected panel metrics."""
        joined, _licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)

        self.assertFalse(summary.empty)
        self.assertIn("business_id", summary.columns)
        self.assertIn("complaint_sum", summary.columns)
        self.assertIn("last_month", summary.columns)
        self.assertIn("active", summary.columns)
        self.assertTrue(set(summary["active"].unique()).issubset({0, 1}))

    def test_build_business_license_metadata_has_expected_columns(self):
        """Test that license metadata aggregation forms expected structured lists."""
        _joined, licenses = prepare_clean_inputs()
        metadata = build_business_license_metadata(licenses)

        self.assertFalse(metadata.empty)
        self.assertIn("business_id", metadata.columns)
        self.assertIn("latitude", metadata.columns)
        self.assertIn("longitude", metadata.columns)
        self.assertIn("license_count", metadata.columns)
        self.assertIn("license_records", metadata.columns)

        is_list = metadata["license_records"].apply(lambda x: isinstance(x, list))
        self.assertTrue(is_list.all())

    def test_merge_business_summary_with_license_metadata_returns_business_rows(self):
        """Test that the summary and metadata dataframes map correctly to one another."""
        joined, licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
        metadata = build_business_license_metadata(licenses)

        merged = merge_business_summary_with_license_metadata(summary, metadata)

        self.assertFalse(merged.empty)
        self.assertIn("business_id", merged.columns)
        self.assertIn("license_records", merged.columns)

    def test_build_feature_returns_valid_geojson_feature(self):
        """Test that a single series row maps to a properly formed GeoJSON feature dict."""
        row = pd.Series(
            {
                "business_id": "B1",
                "active": 1,
                "last_month": pd.Timestamp("2026-02-01"),
                "complaint_sum": 5.0,
                "license_count": 2,
                "license_records": [],
                "latitude": 40.75,
                "longitude": -73.90,
            }
        )

        feature = build_feature(row)

        self.assertIsNotNone(feature)
        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "Point")
        self.assertEqual(feature["geometry"]["coordinates"], [-73.90, 40.75])
        self.assertEqual(feature["properties"]["business_id"], "B1")

    def test_run_geojson_pipeline_writes_valid_geojson(self):
        """Test that the end-to-end GeoJSON pipeline properly dumps out a valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "businesses.geojson"

            config = GeoJSONConfig(
                joined_data_path=TEST_DATA_DIR / "joined_dataset.csv",
                licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
                output_path=output_path,
            )

            returned_path = run_geojson_pipeline(config)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(output_path.exists())

            with output_path.open("r", encoding="utf-8") as file_obj:
                geojson = json.load(file_obj)

            self.assertEqual(geojson["type"], "FeatureCollection")
            self.assertIsInstance(geojson["features"], list)
            self.assertGreater(len(geojson["features"]), 0)

    def test_geojson_output_matches_active_status(self):
        """Test that the generated GeoJSON maps to the expected active status metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "businesses.geojson"
            config = GeoJSONConfig(
                joined_data_path=TEST_DATA_DIR / "joined_dataset.csv",
                licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
                output_path=output_path,
            )
            run_geojson_pipeline(config)

            with output_path.open("r", encoding="utf-8") as file_obj:
                geojson = json.load(file_obj)

            features_df = flatten_geojson_features(geojson)
            joined, _ = prepare_clean_inputs()

            expected_summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
            expected_subset = expected_summary[["business_id", "active", "last_month"]]
            
            merged = features_df.merge(
                expected_subset, on="business_id", how="left", suffixes=("_geojson", "_expected")
            )

            self.assertTrue(merged["active_expected"].notna().all())

            active_geojson = merged["active_geojson"].astype(int)
            active_expected = merged["active_expected"].astype(int)
            self.assertTrue((active_geojson == active_expected).all())
            self.assertTrue((merged["last_month_geojson"] == merged["last_month_expected"]).all())

    def test_geojson_output_matches_complaints(self):
        """Test that the generated GeoJSON maps to the expected complaint sum metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "businesses.geojson"
            config = GeoJSONConfig(
                joined_data_path=TEST_DATA_DIR / "joined_dataset.csv",
                licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
                output_path=output_path,
            )
            run_geojson_pipeline(config)

            with output_path.open("r", encoding="utf-8") as file_obj:
                geojson = json.load(file_obj)

            features_df = flatten_geojson_features(geojson)
            joined, _ = prepare_clean_inputs()

            expected_summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
            expected_subset = expected_summary[["business_id", "complaint_sum"]]
            
            merged = features_df.merge(
                expected_subset, on="business_id", how="left", suffixes=("_geojson", "_expected")
            )

            complaint_geojson = pd.to_numeric(merged["complaint_sum_geojson"], errors="coerce")
            complaint_expected = pd.to_numeric(merged["complaint_sum_expected"], errors="coerce")
            self.assertTrue((complaint_geojson == complaint_expected).all())

    def test_geojson_output_matches_coordinates(self):
        """Test that the generated GeoJSON maps to the expected coordinates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "businesses.geojson"
            config = GeoJSONConfig(
                joined_data_path=TEST_DATA_DIR / "joined_dataset.csv",
                licenses_path=TEST_DATA_DIR / "licenses_sample.csv",
                output_path=output_path,
            )
            run_geojson_pipeline(config)

            with output_path.open("r", encoding="utf-8") as file_obj:
                geojson = json.load(file_obj)

            features_df = flatten_geojson_features(geojson)
            _, licenses = prepare_clean_inputs()

            expected_coords = (
                licenses.sort_values(["Business Unique ID"])
                .drop_duplicates(subset=["Business Unique ID"], keep="first")
                [["Business Unique ID", "Latitude", "Longitude"]]
                .rename(
                    columns={
                        "Business Unique ID": "business_id",
                        "Latitude": "latitude_expected",
                        "Longitude": "longitude_expected",
                    }
                )
                .reset_index(drop=True)
            )

            merged_coords = features_df.merge(
                expected_coords,
                on="business_id",
                how="left",
            )

            self.assertTrue(merged_coords["latitude_expected"].notna().all())
            self.assertTrue(merged_coords["longitude_expected"].notna().all())

            lat_geojson = merged_coords["latitude"].round(8)
            lat_expected = merged_coords["latitude_expected"].round(8)
            self.assertTrue((lat_geojson == lat_expected).all())


if __name__ == "__main__":
    unittest.main()
