"""
Unit tests for logistic feature builder helper utilities.
"""

from __future__ import annotations

import unittest

import pandas as pd

from bizsurvival515.tests.test_helpers import assert_clamp_case, assert_nearest_cluster_case

from bizsurvival515.app.utils.feature_builder import (
    BusinessProfileInputs,
    LOGISTIC_CLUSTER_COL,
    LOGISTIC_CLUSTER_LAT_COL,
    LOGISTIC_CLUSTER_LNG_COL,
    LOGISTIC_LAT_COL,
    LOGISTIC_LICENSE_COL,
    LOGISTIC_LNG_COL,
    _rename_logistic_cluster_columns,
    assign_nearest_cluster_centroid,
    baseline_new_business_profile,
    build_logistic_profile,
    build_zero_profile,
    category_display_to_column_map,
    category_feature_columns,
    complaint_display_to_column_map,
    complaint_feature_columns,
    prettify_feature_name,
)
from bizsurvival515.app.utils.location_utils import (
    NYC_LAT_MAX,
    NYC_LNG_MIN,
    assign_nearest_cluster_info,
    build_cluster_reference_df,
    clamp_to_nyc_bounds,
)


class TestFeatureBuilderUtils(unittest.TestCase):
    """Tests for logistic feature builder utility functions."""

    def setUp(self):
        """Create shared test fixtures."""
        self.reference_df = pd.DataFrame(
            {
                LOGISTIC_CLUSTER_COL: [1, 2],
                LOGISTIC_CLUSTER_LAT_COL: [40.70, 40.80],
                LOGISTIC_CLUSTER_LNG_COL: [-74.00, -73.95],
                LOGISTIC_LAT_COL: [40.71, 40.81],
                LOGISTIC_LNG_COL: [-74.01, -73.96],
            }
        )
        self.kept_columns = [
            LOGISTIC_LICENSE_COL,
            LOGISTIC_LAT_COL,
            LOGISTIC_LNG_COL,
            LOGISTIC_CLUSTER_COL,
            LOGISTIC_CLUSTER_LAT_COL,
            LOGISTIC_CLUSTER_LNG_COL,
            "business_category_alpha_first12m_max",
            "complaint_type_noise_first12m_sum",
        ]

    def test_prettify_feature_name(self):
        """Test prettification of raw feature names into display labels."""
        self.assertEqual(
            prettify_feature_name("business_category_alpha_first12m_max"),
            "Alpha",
        )
        self.assertEqual(
            prettify_feature_name("complaint_type_noise_first12m_sum"),
            "Noise",
        )
        self.assertEqual(
            prettify_feature_name("active_license_count_first12m_mean"),
            "Active License Count",
        )

    def test_category_and_complaint_feature_columns(self):
        """Test extraction of kept category and complaint feature columns."""
        self.assertEqual(
            category_feature_columns(self.kept_columns),
            ["business_category_alpha_first12m_max"],
        )
        self.assertEqual(
            complaint_feature_columns(self.kept_columns),
            ["complaint_type_noise_first12m_sum"],
        )

    def test_display_maps(self):
        """Test display-name to raw-column mappings for categories and complaints."""
        self.assertEqual(
            category_display_to_column_map(self.kept_columns),
            {"Alpha": "business_category_alpha_first12m_max"},
        )
        self.assertEqual(
            complaint_display_to_column_map(self.kept_columns),
            {"Noise": "complaint_type_noise_first12m_sum"},
        )

    def test_build_zero_profile(self):
        """Test creation of a single-row zero-initialized profile."""
        profile = build_zero_profile(["a", "b"])
        self.assertEqual(profile.shape, (1, 2))
        self.assertTrue((profile.iloc[0] == 0.0).all())

    def test_build_cluster_reference_df(self):
        """Test construction of a cluster reference dataframe from renamed columns."""
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_df = build_cluster_reference_df(
            reference_df=renamed,
            lat_column="location_cluster_lat",
            lng_column="location_cluster_lng",
            cluster_column="location_cluster",
        )
        self.assertFalse(cluster_df.empty)
        self.assertIn("location_cluster_lat", cluster_df.columns)

    def test_assign_nearest_cluster_info(self):
        """Test nearest-cluster lookup from a prepared cluster dataframe."""
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_df = build_cluster_reference_df(
            reference_df=renamed,
            lat_column="location_cluster_lat",
            lng_column="location_cluster_lng",
            cluster_column="location_cluster",
        )
        assert_nearest_cluster_case(self, assign_nearest_cluster_info, cluster_df)

    def test_assign_nearest_cluster_centroid(self):
        """Test nearest centroid lookup wrapper."""
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_lat, cluster_lng = assign_nearest_cluster_centroid(
            40.805,
            -73.955,
            renamed,
        )
        self.assertAlmostEqual(cluster_lat, 40.80, places=2)
        self.assertAlmostEqual(cluster_lng, -73.95, places=2)

    def test_baseline_new_business_profile(self):
        """Test construction of the baseline logistic business profile."""
        profile = baseline_new_business_profile(self.kept_columns, self.reference_df)
        self.assertEqual(profile.loc[0, LOGISTIC_LICENSE_COL], 1.0)
        self.assertIn(LOGISTIC_LAT_COL, profile.columns)
        self.assertIn(LOGISTIC_CLUSTER_LAT_COL, profile.columns)

    def test_build_logistic_profile(self):
        """Test construction of a populated logistic business profile."""
        inputs = BusinessProfileInputs(
            selected_category_columns=["business_category_alpha_first12m_max"],
            active_license_count=2,
            business_latitude=40.72,
            business_longitude=-74.02,
            complaint_counts={"complaint_type_noise_first12m_sum": 4},
        )

        profile = build_logistic_profile(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            inputs=inputs,
        )
        self.assertEqual(profile.loc[0, LOGISTIC_LICENSE_COL], 2.0)
        self.assertEqual(
            profile.loc[0, "business_category_alpha_first12m_max"],
            1.0,
        )
        self.assertEqual(
            profile.loc[0, "complaint_type_noise_first12m_sum"],
            4.0,
        )

    def test_build_logistic_profile_ignores_unknown_inputs(self):
        """Ignore unknown category and complaint keys not present in kept columns."""
        inputs = BusinessProfileInputs(
            selected_category_columns=[
                "business_category_alpha_first12m_max",
                "business_category_not_in_model_first12m_max",
            ],
            active_license_count=2,
            business_latitude=40.72,
            business_longitude=-74.02,
            complaint_counts={
                "complaint_type_noise_first12m_sum": 4,
                "complaint_type_unknown_first12m_sum": 9,
            },
        )

        profile = build_logistic_profile(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            inputs=inputs,
        )

        self.assertIn("business_category_alpha_first12m_max", profile.columns)
        self.assertEqual(
            profile.loc[0, "business_category_alpha_first12m_max"],
            1.0,
        )
        self.assertEqual(
            profile.loc[0, "complaint_type_noise_first12m_sum"],
            4.0,
        )
        self.assertNotIn("business_category_not_in_model_first12m_max", profile.columns)
        self.assertNotIn("complaint_type_unknown_first12m_sum", profile.columns)

    def test_clamp_to_nyc_bounds(self):
        """Test clamping of coordinates to NYC geographic bounds."""
        assert_clamp_case(self, clamp_to_nyc_bounds, NYC_LAT_MAX, NYC_LNG_MIN)


if __name__ == "__main__":
    unittest.main()
