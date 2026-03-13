"""Unit tests for the GeoJSON mapbox pipeline."""

from pathlib import Path
import json
import tempfile
import unittest

import pandas as pd

from pipeline.mapbox import (
    GeoBounds,
    GeoJSONConfig,
    VALID_BOROUGHS,
    build_business_license_metadata,
    build_business_summary,
    build_feature,
    build_geojson,
    build_geojson_features,
    filter_nyc_license_coordinates,
    filter_valid_boroughs,
    load_joined_dataset,
    load_licenses_dataset,
    merge_business_summary_with_license_metadata,
    prepare_geojson_inputs,
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
    """Flatten GeoJSON features into a tabular dataframe for assertions."""
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
        flattened["last_month"] = pd.to_datetime(
            flattened["last_month"],
            errors="coerce",
        )
    return flattened


def prepare_clean_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, validate, and clean joined and license inputs for tests."""
    joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
    licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")
    return prepare_geojson_inputs(
        joined=joined,
        licenses=licenses,
        bounds=GeoBounds(
            lat_min=NYC_LAT_MIN,
            lat_max=NYC_LAT_MAX,
            lng_min=NYC_LNG_MIN,
            lng_max=NYC_LNG_MAX,
        ),
    )


def build_expected_coords(licenses: pd.DataFrame) -> pd.DataFrame:
    """Build the expected business-to-coordinate mapping from licenses."""
    return (
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


def assert_summary_matches_geojson(
    test_case: unittest.TestCase,
    features_df: pd.DataFrame,
    expected_summary: pd.DataFrame,
) -> None:
    """Assert that GeoJSON properties match the expected business summary."""
    merged = features_df.merge(
        expected_summary[["business_id", "active", "last_month", "complaint_sum"]],
        on="business_id",
        how="left",
        suffixes=("_geojson", "_expected"),
    )

    test_case.assertTrue(merged["active_expected"].notna().all())
    test_case.assertTrue(
        (
            merged["active_geojson"].astype(int)
            == merged["active_expected"].astype(int)
        ).all()
    )
    test_case.assertTrue(
        (merged["last_month_geojson"] == merged["last_month_expected"]).all()
    )
    test_case.assertTrue(
        (
            pd.to_numeric(merged["complaint_sum_geojson"], errors="coerce")
            == pd.to_numeric(merged["complaint_sum_expected"], errors="coerce")
        ).all()
    )


def assert_coordinates_match_geojson(
    test_case: unittest.TestCase,
    features_df: pd.DataFrame,
    expected_coords: pd.DataFrame,
) -> None:
    """Assert that GeoJSON point coordinates match expected license coordinates."""
    merged_coords = features_df.merge(
        expected_coords,
        on="business_id",
        how="left",
    )

    test_case.assertTrue(merged_coords["latitude_expected"].notna().all())
    test_case.assertTrue(merged_coords["longitude_expected"].notna().all())
    test_case.assertTrue(
        (
            merged_coords["latitude"].round(8)
            == merged_coords["latitude_expected"].round(8)
        ).all()
    )
    test_case.assertTrue(
        (
            merged_coords["longitude"].round(8)
            == merged_coords["longitude_expected"].round(8)
        ).all()
    )


class TestMapbox(unittest.TestCase):
    """Test suite for mapbox GeoJSON preparation helpers and pipeline."""

    def test_load_joined_dataset_parses_month_and_business_id(self):
        """Load joined data with required identifiers and parsed month values."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        self.assertIn("business_id", joined.columns)
        self.assertIn("month", joined.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(joined["month"]))

    def test_validate_joined_dataset_accepts_valid_data(self):
        """Accept a valid joined dataset without raising validation errors."""
        joined = load_joined_dataset(TEST_DATA_DIR / "joined_dataset.csv")
        validate_joined_dataset(joined)

    def test_validate_joined_dataset_rejects_missing_required_columns(self):
        """Reject joined datasets that omit required columns."""
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
        """Accept a valid licenses dataset without raising validation errors."""
        licenses = load_licenses_dataset(TEST_DATA_DIR / "licenses_sample.csv")
        validate_licenses_dataset(licenses)

    def test_validate_licenses_dataset_rejects_missing_required_columns(self):
        """Reject license datasets that omit required columns."""
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
        """Keep only rows whose borough is one of the five NYC boroughs."""
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
        """Keep only license coordinates that fall within NYC bounds."""
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
        """Build business summary rows with core activity and complaint fields."""
        joined, _licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)

        self.assertFalse(summary.empty)
        self.assertIn("business_id", summary.columns)
        self.assertIn("complaint_sum", summary.columns)
        self.assertIn("last_month", summary.columns)
        self.assertIn("active", summary.columns)
        self.assertTrue(set(summary["active"].unique()).issubset({0, 1}))

    def test_build_business_license_metadata_has_expected_columns(self):
        """Build license metadata with coordinates and record lists per business."""
        _joined, licenses = prepare_clean_inputs()
        metadata = build_business_license_metadata(licenses)

        self.assertFalse(metadata.empty)
        self.assertIn("business_id", metadata.columns)
        self.assertIn("latitude", metadata.columns)
        self.assertIn("longitude", metadata.columns)
        self.assertIn("license_count", metadata.columns)
        self.assertIn("license_records", metadata.columns)
        self.assertTrue(
            metadata["license_records"].apply(lambda x: isinstance(x, list)).all()
        )

    def test_merge_business_summary_with_license_metadata_returns_business_rows(self):
        """Merge summary and metadata into a business-level GeoJSON source table."""
        joined, licenses = prepare_clean_inputs()
        summary = build_business_summary(joined, cutoff_date=CUTOFF_DATE)
        metadata = build_business_license_metadata(licenses)

        merged = merge_business_summary_with_license_metadata(summary, metadata)

        self.assertFalse(merged.empty)
        self.assertIn("business_id", merged.columns)
        self.assertIn("license_records", merged.columns)

    def test_build_feature_returns_valid_geojson_feature(self):
        """Convert one business row into a valid GeoJSON point feature."""
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
        """Build feature rows and wrap them in a GeoJSON feature collection."""
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
        """Run the pipeline and write a valid non-empty GeoJSON file."""
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
        """Verify GeoJSON output matches expected business summary and coordinates."""
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
        expected_coords = build_expected_coords(licenses)

        self.assertFalse(features_df.empty)
        self.assertTrue(features_df["business_id"].notna().all())
        self.assertFalse(features_df["business_id"].duplicated().any())
        self.assertTrue((features_df["geometry_type"] == "Point").all())
        self.assertTrue(features_df["latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX).all())
        self.assertTrue(features_df["longitude"].between(NYC_LNG_MIN, NYC_LNG_MAX).all())
        self.assertTrue(
            features_df["license_records"].apply(lambda x: isinstance(x, list)).all()
        )

        for records in features_df["license_records"]:
            for record in records:
                borough = record.get("borough")
                self.assertTrue(borough is None or borough in VALID_BOROUGHS)

        assert_summary_matches_geojson(self, features_df, expected_summary)
        assert_coordinates_match_geojson(self, features_df, expected_coords)

    def test_build_geojson_with_no_features(self):
        """Build an empty FeatureCollection when there are no features."""
        geojson = build_geojson([])

        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertIn("features", geojson)
        self.assertEqual(geojson["features"], [])


if __name__ == "__main__":
    unittest.main()
