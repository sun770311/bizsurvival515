# tests/test_cox_feature_builder_utils.py
"""
Unit tests for Cox feature builder helper utilities.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from pipeline.utils import LOCATION_FEATURE_COLUMNS
from tests.test_helpers import assert_clamp_case, assert_nearest_cluster_case

from app.utils.cox_feature_builder import (
    CoxProfileInputs,
    _cluster_reference_df,
    _mutate_active_license_count,
    _mutate_complaint_counts,
    _mutate_location,
    _sample_initial_categories,
    _sample_initial_complaint_counts,
    baseline_standard_cox_profile,
    baseline_time_varying_cox_profile,
    build_standard_cox_profile,
    build_time_varying_cox_profile,
    build_time_varying_cox_profiles_over_time,
    build_zero_profile,
    cox_category_display_to_column_map,
    cox_category_feature_columns,
    cox_complaint_display_to_column_map,
    cox_complaint_feature_columns,
    generate_time_varying_example_timelines,
    get_reference_median_lat_lng,
    prettify_cox_feature_name,
    summarize_generated_time_varying_timelines,
)
from app.utils.location_utils import (
    NYC_LAT_MAX,
    NYC_LAT_MIN,
    NYC_LNG_MAX,
    NYC_LNG_MIN,
    assign_nearest_cluster_info,
    clamp_to_nyc_bounds,
)


class TestCoxFeatureBuilderUtils(unittest.TestCase):
    """Tests for Cox feature builder utility functions."""

    def setUp(self):
        """Create shared test fixtures."""
        self.reference_df = pd.DataFrame(
            {
                "location_cluster_first12m_first": [1, 2],
                "location_cluster_lat_first12m_first": [40.70, 40.80],
                "location_cluster_lng_first12m_first": [-74.00, -73.95],
                "business_latitude_first12m_first": [40.71, 40.81],
                "business_longitude_first12m_first": [-74.01, -73.96],
            }
        )
        self.kept_columns = [
            "active_license_count",
            *LOCATION_FEATURE_COLUMNS,
            "business_category_alpha",
            "business_category_beta",
            "complaint_type_noise",
            "complaint_type_water_leak",
        ]

    def test_prettify_cox_feature_name(self):
        """Test feature-name prettification for category, complaint, and numeric fields."""
        self.assertEqual(
            prettify_cox_feature_name("business_category_home_improvement_contractor"),
            "Home Improvement Contractor",
        )
        self.assertEqual(
            prettify_cox_feature_name("complaint_type_water_leak"),
            "Water Leak",
        )
        self.assertEqual(
            prettify_cox_feature_name("active_license_count"),
            "Active License Count",
        )

    def test_category_and_complaint_feature_columns(self):
        """Test extraction of kept category and complaint feature columns."""
        self.assertEqual(
            cox_category_feature_columns(self.kept_columns),
            ["business_category_alpha", "business_category_beta"],
        )
        self.assertEqual(
            cox_complaint_feature_columns(self.kept_columns),
            ["complaint_type_noise", "complaint_type_water_leak"],
        )

    def test_display_maps(self):
        """Test display-name to raw-column mappings."""
        category_map = cox_category_display_to_column_map(self.kept_columns)
        complaint_map = cox_complaint_display_to_column_map(self.kept_columns)

        self.assertEqual(category_map["Alpha"], "business_category_alpha")
        self.assertEqual(complaint_map["Noise"], "complaint_type_noise")

    def test_build_zero_profile(self):
        """Test creation of a single-row zero-initialized profile."""
        profile = build_zero_profile(["a", "b"])
        self.assertEqual(profile.shape, (1, 2))
        self.assertTrue((profile.iloc[0] == 0.0).all())

    def test_cluster_reference_df_aggregated_columns(self):
        """Test cluster reference extraction from aggregated coordinate columns."""
        cluster_df = _cluster_reference_df(self.reference_df)
        self.assertFalse(cluster_df.empty)
        self.assertIn("location_cluster_lat", cluster_df.columns)
        self.assertIn("location_cluster_lng", cluster_df.columns)

    def test_assign_nearest_cluster_info(self):
        """Test nearest-cluster lookup using a prepared cluster reference dataframe."""
        cluster_df = _cluster_reference_df(self.reference_df)
        assert_nearest_cluster_case(self, assign_nearest_cluster_info, cluster_df)

    def test_get_reference_median_lat_lng(self):
        """Test median latitude and longitude extraction from reference data."""
        lat, lng = get_reference_median_lat_lng(self.reference_df)
        self.assertAlmostEqual(lat, 40.76)
        self.assertAlmostEqual(lng, -73.985)

    def test_baseline_profiles(self):
        """Test both standard and time-varying baseline Cox profile builders."""
        standard_profile = baseline_standard_cox_profile(
            self.kept_columns,
            self.reference_df,
        )
        self.assertEqual(standard_profile.loc[0, "active_license_count"], 1.0)
        self.assertIn("business_latitude", standard_profile.columns)
        self.assertIn("location_cluster_lat", standard_profile.columns)

        time_varying_profile = baseline_time_varying_cox_profile(
            self.kept_columns,
            self.reference_df,
        )
        self.assertEqual(time_varying_profile.loc[0, "active_license_count"], 1.0)
        self.assertIn("business_longitude", time_varying_profile.columns)

    def test_build_standard_cox_profile(self):
        """Test construction of a populated standard Cox profile."""
        inputs = CoxProfileInputs(
            selected_category_columns=["business_category_beta"],
            active_license_count=3,
            business_latitude=40.73,
            business_longitude=-74.01,
            complaint_counts={"complaint_type_water_leak": 2},
        )
        profile = build_standard_cox_profile(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            inputs=inputs,
        )
        self.assertEqual(profile.loc[0, "active_license_count"], 3.0)
        self.assertEqual(profile.loc[0, "business_category_beta"], 1.0)
        self.assertEqual(profile.loc[0, "complaint_type_water_leak"], 2.0)

    def test_build_time_varying_cox_profile(self):
        """Test construction of a populated time-varying Cox profile."""
        inputs = CoxProfileInputs(
            selected_category_columns=["business_category_alpha"],
            active_license_count=2,
            business_latitude=40.72,
            business_longitude=-74.02,
            complaint_counts={"complaint_type_noise": 3},
        )
        profile = build_time_varying_cox_profile(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            inputs=inputs,
        )
        self.assertEqual(profile.loc[0, "active_license_count"], 2.0)
        self.assertEqual(profile.loc[0, "business_category_alpha"], 1.0)
        self.assertEqual(profile.loc[0, "complaint_type_noise"], 3.0)

    def test_build_time_varying_cox_profiles_over_time(self):
        """Test assembly and month-ordering of multiple time-varying profile rows."""
        profiles = build_time_varying_cox_profiles_over_time(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            timepoint_specs=[
                {
                    "month": 24,
                    "selected_category_columns": ["business_category_alpha"],
                    "active_license_count": 2,
                    "business_latitude": 40.80,
                    "business_longitude": -73.95,
                    "complaint_counts": {"complaint_type_noise": 1},
                },
                {
                    "month": 12,
                    "selected_category_columns": [],
                    "active_license_count": 1,
                    "business_latitude": 40.70,
                    "business_longitude": -74.00,
                    "complaint_counts": {},
                },
            ],
        )
        self.assertEqual(list(profiles["month"]), [12, 24])
        self.assertEqual(len(profiles), 2)

    def test_mutate_active_license_count_bounds(self):
        """Test that mutated active license counts stay within valid bounds."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            value = _mutate_active_license_count(1, rng)
            self.assertGreaterEqual(value, 1)
            self.assertLessEqual(value, 5)

    def test_sample_initial_categories(self):
        """Test sampling of initial category selections."""
        rng = np.random.default_rng(42)
        categories = _sample_initial_categories(
            ["business_category_alpha", "business_category_beta"],
            rng,
        )
        self.assertLessEqual(len(categories), 2)
        self.assertTrue(
            set(categories).issubset(
                {"business_category_alpha", "business_category_beta"}
            )
        )

    def test_sample_initial_complaint_counts(self):
        """Test sampling of initial complaint-count assignments."""
        rng = np.random.default_rng(42)
        counts = _sample_initial_complaint_counts(
            ["complaint_type_noise", "complaint_type_water_leak"],
            rng,
        )
        self.assertTrue(
            set(counts.keys()).issubset(
                {"complaint_type_noise", "complaint_type_water_leak"}
            )
        )
        self.assertTrue(all(value >= 1 for value in counts.values()))

    def test_mutate_complaint_counts_nonnegative(self):
        """Test that mutated complaint counts never become negative."""
        rng = np.random.default_rng(42)
        counts = _mutate_complaint_counts(
            {"complaint_type_noise": 2.0},
            ["complaint_type_noise", "complaint_type_water_leak"],
            rng,
        )
        self.assertTrue(all(value >= 0 for value in counts.values()))

    def test_mutate_location_stays_in_bounds(self):
        """Test that mutated locations remain inside NYC bounds."""
        rng = np.random.default_rng(42)
        lat, lng = _mutate_location(40.71, -74.00, rng)
        self.assertGreaterEqual(lat, NYC_LAT_MIN)
        self.assertLessEqual(lat, NYC_LAT_MAX)
        self.assertGreaterEqual(lng, NYC_LNG_MIN)
        self.assertLessEqual(lng, NYC_LNG_MAX)

    def test_generate_time_varying_example_timelines(self):
        """Test generation of example timelines with expected month spacing."""
        generated = generate_time_varying_example_timelines(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            num_businesses=2,
            num_timepoints=3,
            random_state=42,
        )
        self.assertEqual(len(generated), 2)
        for business in generated:
            self.assertIn("label", business)
            self.assertEqual(len(business["timepoints"]), 3)
            months = [tp["month"] for tp in business["timepoints"]]
            self.assertEqual(months, [0, 12, 24])

    def test_summarize_generated_time_varying_timelines(self):
        """Test summarization of generated time-varying timelines into a dataframe."""
        generated = [
            {
                "label": "Business 1",
                "timepoints": [
                    {
                        "month": 0,
                        "selected_category_columns": ["business_category_alpha"],
                        "active_license_count": 1,
                        "business_latitude": 40.71,
                        "business_longitude": -74.00,
                        "complaint_counts": {"complaint_type_noise": 2.0},
                    }
                ],
            }
        ]
        summary = summarize_generated_time_varying_timelines(generated)
        self.assertEqual(len(summary), 1)
        self.assertIn("Business", summary.columns)
        self.assertIn("Categories", summary.columns)
        self.assertIn("Complaint counts", summary.columns)

    def test_clamp_to_nyc_bounds(self):
        """Test clamping of coordinates to NYC geographic bounds."""
        assert_clamp_case(self, clamp_to_nyc_bounds, NYC_LAT_MAX, NYC_LNG_MIN)


if __name__ == "__main__":
    unittest.main()
