from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from pipeline.mapbox import (
    GeoJSONConfig,
    VALID_BOROUGHS,
    build_business_license_metadata,
    build_business_summary,
    build_feature,
    build_full_address,
    build_geojson,
    build_geojson_features,
    clean_joined_business_ids,
    clean_license_fields,
    filter_nyc_license_coordinates,
    filter_valid_boroughs,
    load_joined_dataset,
    load_licenses_dataset,
    merge_business_summary_with_license_metadata,
    run_geojson_pipeline,
    validate_joined_dataset,
    validate_licenses_dataset,
)


TEST_DATA_DIR = Path(__file__).parent / "data"
CUTOFF_DATE = pd.Timestamp("2026-03-01")
NYC_LAT_MIN = 40.49
NYC_LAT_MAX = 40.92
NYC_LNG_MIN = -74.27
NYC_LNG_MAX = -73.68


def flatten_geojson_features(geojson: dict) -> pd.DataFrame:
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


def prepare_clean_inputs():
    joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
    licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")

    validate_joined_dataset(joined)
    validate_licenses_dataset(licenses)

    joined = clean_joined_business_ids(joined)
    licenses = clean_license_fields(licenses)
    licenses = build_full_address(licenses)
    licenses = filter_valid_boroughs(licenses)
    licenses = filter_nyc_license_coordinates(
        licenses=licenses,
        lat_min=NYC_LAT_MIN,
        lat_max=NYC_LAT_MAX,
        lng_min=NYC_LNG_MIN,
        lng_max=NYC_LNG_MAX,
    )
    return joined, licenses


class TestMapbox(unittest.TestCase):
    def test_load_joined_dataset_parses_month_and_business_id(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        self.assertIn("business_id", joined.columns)
        self.assertIn("month", joined.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["month"]))

    def test_validate_joined_dataset_accepts_valid_data(self):
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined)

    def test_validate_joined_dataset_rejects_missing_required_columns(self):
        bad = pd.DataFrame(
            {
                "business_id": ["A"],
                "month": ["2025-01-01"],
            }
        )
        bad["month"] = pd.to_datetime(bad["month"])

        with self.assertRaisesRegex(ValueError, "Missing required joined columns"):
            validate_joined_dataset(bad)

    def test_validate_licenses_dataset_accepts_valid_data(self):
        licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")
        validate_licenses_dataset(licenses)

    def test_validate_licenses_dataset_rejects_missing_required_columns(self):
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

    def test_filter_nyc_license_coordinates_keeps_only_nyc_bounds(self):
        licenses = pd.DataFrame(
            {
                "Business Unique ID": ["1", "2"],
                "Latitude": [40.75, 39.0],
                "Longitude": [-73.9, -73.9],
            }
        )

        filtered = filter_nyc_license_coordinates(
            licenses=licenses,
            lat_min=NYC_LAT_MIN,
            lat_max=NYC_LAT_MAX,
            lng_min=NYC_LNG_MIN,
            lng_max=NYC_LNG_MAX,
        )

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["Business Unique ID"], "1")

    def test_build_business_summary_has_expected_columns(self):
        joined, _licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)

        self.assertFalse(summary.empty)
        self.assertIn("business_id", summary.columns)
        self.assertIn("complaint_sum", summary.columns)
        self.assertIn("last_month", summary.columns)
        self.assertIn("active", summary.columns)
        self.assertTrue(set(summary["active"].unique()).issubset({0, 1}))

    def test_build_business_license_metadata_has_expected_columns(self):
        _joined, licenses = prepare_clean_inputs()
        metadata = build_business_license_metadata(licenses)

        self.assertFalse(metadata.empty)
        self.assertIn("business_id", metadata.columns)
        self.assertIn("latitude", metadata.columns)
        self.assertIn("longitude", metadata.columns)
        self.assertIn("license_count", metadata.columns)
        self.assertIn("license_records", metadata.columns)
        self.assertTrue(metadata["license_records"].apply(lambda x: isinstance(x, list)).all())

    def test_merge_business_summary_with_license_metadata_returns_business_rows(self):
        joined, licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
        metadata = build_business_license_metadata(licenses)

        merged = merge_business_summary_with_license_metadata(summary, metadata)

        self.assertFalse(merged.empty)
        self.assertIn("business_id", merged.columns)
        self.assertIn("license_records", merged.columns)

    def test_build_feature_returns_valid_geojson_feature(self):
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

    def test_build_geojson_features_and_collection(self):
        businesses = pd.DataFrame(
            [
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
            ]
        )

        features = build_geojson_features(businesses)
        geojson = build_geojson(features)

        self.assertEqual(len(features), 1)
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertEqual(len(geojson["features"]), 1)

    def test_run_geojson_pipeline_writes_valid_geojson(self):
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

    def test_geojson_output_matches_expected_business_logic(self):
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
            joined, licenses = prepare_clean_inputs()

            expected_summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
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

            self.assertFalse(features_df.empty)
            self.assertTrue(features_df["business_id"].notna().all())
            self.assertFalse(features_df["business_id"].duplicated().any())
            self.assertTrue((features_df["geometry_type"] == "Point").all())
            self.assertTrue(features_df["latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX).all())
            self.assertTrue(features_df["longitude"].between(NYC_LNG_MIN, NYC_LNG_MAX).all())
            self.assertTrue(features_df["license_records"].apply(lambda x: isinstance(x, list)).all())

            for records in features_df["license_records"]:
                for record in records:
                    borough = record.get("borough")
                    self.assertTrue(borough is None or borough in VALID_BOROUGHS)

            merged = features_df.merge(
                expected_summary[["business_id", "active", "last_month", "complaint_sum"]],
                on="business_id",
                how="left",
                suffixes=("_geojson", "_expected"),
            )

            self.assertTrue(merged["active_expected"].notna().all())
            self.assertTrue(
                (merged["active_geojson"].astype(int) == merged["active_expected"].astype(int)).all()
            )
            self.assertTrue((merged["last_month_geojson"] == merged["last_month_expected"]).all())
            self.assertTrue(
                (
                    pd.to_numeric(merged["complaint_sum_geojson"], errors="coerce")
                    == pd.to_numeric(merged["complaint_sum_expected"], errors="coerce")
                ).all()
            )

            merged_coords = features_df.merge(
                expected_coords,
                on="business_id",
                how="left",
            )

            self.assertTrue(merged_coords["latitude_expected"].notna().all())
            self.assertTrue(merged_coords["longitude_expected"].notna().all())
            self.assertTrue(
                (merged_coords["latitude"].round(8) == merged_coords["latitude_expected"].round(8)).all()
            )
            self.assertTrue(
                (merged_coords["longitude"].round(8) == merged_coords["longitude_expected"].round(8)).all()
            )


if __name__ == "__main__":
    unittest.main()