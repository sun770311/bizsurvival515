from __future__ import annotations

import unittest

import pandas as pd

from app.utils.feature_builder import (
    LOGISTIC_CLUSTER_COL,
    LOGISTIC_CLUSTER_LAT_COL,
    LOGISTIC_CLUSTER_LNG_COL,
    LOGISTIC_LAT_COL,
    LOGISTIC_LICENSE_COL,
    LOGISTIC_LNG_COL,
    NYC_LAT_MAX,
    NYC_LNG_MIN,
    _cluster_reference_df,
    _rename_logistic_cluster_columns,
    assign_nearest_cluster_centroid,
    assign_nearest_cluster_info,
    baseline_new_business_profile,
    build_logistic_profile,
    build_zero_profile,
    category_display_to_column_map,
    category_feature_columns,
    clamp_to_nyc_bounds,
    complaint_display_to_column_map,
    complaint_feature_columns,
    prettify_feature_name,
)


class TestFeatureBuilderUtils(unittest.TestCase):
    def setUp(self):
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
        self.assertEqual(
            category_feature_columns(self.kept_columns),
            ["business_category_alpha_first12m_max"],
        )
        self.assertEqual(
            complaint_feature_columns(self.kept_columns),
            ["complaint_type_noise_first12m_sum"],
        )

    def test_display_maps(self):
        self.assertEqual(
            category_display_to_column_map(self.kept_columns),
            {"Alpha": "business_category_alpha_first12m_max"},
        )
        self.assertEqual(
            complaint_display_to_column_map(self.kept_columns),
            {"Noise": "complaint_type_noise_first12m_sum"},
        )

    def test_build_zero_profile(self):
        profile = build_zero_profile(["a", "b"])
        self.assertEqual(profile.shape, (1, 2))
        self.assertTrue((profile.iloc[0] == 0.0).all())

    def test_cluster_reference_df(self):
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_df = _cluster_reference_df(renamed)
        self.assertFalse(cluster_df.empty)
        self.assertIn("location_cluster_lat", cluster_df.columns)

    def test_assign_nearest_cluster_info(self):
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_id, cluster_lat, cluster_lng = assign_nearest_cluster_info(
            40.705,
            -74.005,
            renamed,
        )
        self.assertIsNotNone(cluster_id)
        self.assertAlmostEqual(cluster_lat, 40.70, places=2)
        self.assertAlmostEqual(cluster_lng, -74.00, places=2)

    def test_assign_nearest_cluster_centroid(self):
        renamed = _rename_logistic_cluster_columns(self.reference_df)
        cluster_lat, cluster_lng = assign_nearest_cluster_centroid(
            40.805,
            -73.955,
            renamed,
        )
        self.assertAlmostEqual(cluster_lat, 40.80, places=2)
        self.assertAlmostEqual(cluster_lng, -73.95, places=2)

    def test_baseline_new_business_profile(self):
        profile = baseline_new_business_profile(self.kept_columns, self.reference_df)
        self.assertEqual(profile.loc[0, LOGISTIC_LICENSE_COL], 1.0)
        self.assertIn(LOGISTIC_LAT_COL, profile.columns)
        self.assertIn(LOGISTIC_CLUSTER_LAT_COL, profile.columns)

    def test_build_logistic_profile(self):
        profile = build_logistic_profile(
            kept_columns=self.kept_columns,
            reference_df=self.reference_df,
            selected_category_columns=["business_category_alpha_first12m_max"],
            active_license_count=2,
            business_latitude=40.72,
            business_longitude=-74.02,
            complaint_counts={"complaint_type_noise_first12m_sum": 4},
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

    def test_clamp_to_nyc_bounds(self):
        lat, lng = clamp_to_nyc_bounds(50.0, -80.0)
        self.assertEqual(lat, NYC_LAT_MAX)
        self.assertEqual(lng, NYC_LNG_MIN)


if __name__ == "__main__":
    unittest.main()