from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from app.utils.prediction_tools import (
    predict_logistic_profile,
    predict_standard_cox_profile,
    predict_time_varying_cox_profile,
    predict_time_varying_cox_profiles,
    top_positive_negative,
)


class DummyPipeline:
    def predict_proba(self, profile_df: pd.DataFrame):
        return np.array([[0.25, 0.75]])

    def predict(self, profile_df: pd.DataFrame):
        return np.array([1])


class FailingPipeline:
    def predict_proba(self, profile_df: pd.DataFrame):
        raise ValueError("boom")

    def predict(self, profile_df: pd.DataFrame):
        return [0]


class DummyScaler:
    def transform(self, feature_df: pd.DataFrame):
        return feature_df.to_numpy(dtype=float)


class DummyPartialHazardResult:
    def __init__(self, values: list[float]):
        self._series = pd.Series(values)

    @property
    def iloc(self):
        return self._series.iloc

    def to_numpy(self, dtype=float):
        return self._series.to_numpy(dtype=dtype)


class DummyCoxModel:
    def predict_partial_hazard(self, scaled_df: pd.DataFrame):
        values = [float(row.sum()) for _, row in scaled_df.iterrows()]
        return DummyPartialHazardResult(values)

    def predict_survival_function(self, scaled_df: pd.DataFrame, times: list[int]):
        partial_hazard = float(scaled_df.iloc[0].sum())
        survival_values = [max(0.0, 1.0 - 0.01 * partial_hazard - 0.001 * t) for t in times]
        return pd.DataFrame({0: survival_values}, index=times)


class TestPredictionToolsUtils(unittest.TestCase):
    def setUp(self):
        self.profile_df = pd.DataFrame(
            {"x1": [1.0], "x2": [2.0]}
        )
        self.multi_profile_df = pd.DataFrame(
            {"x1": [1.0, 3.0], "x2": [2.0, 4.0], "month": [0, 12]}
        )
        self.kept_columns = ["x1", "x2"]

    def test_predict_logistic_profile(self):
        result = predict_logistic_profile(DummyPipeline(), self.profile_df)
        self.assertAlmostEqual(result["predicted_survival_probability"], 0.75)
        self.assertEqual(result["predicted_class"], 1)

    def test_predict_logistic_profile_wraps_error(self):
        with self.assertRaises(RuntimeError):
            predict_logistic_profile(FailingPipeline(), self.profile_df)

    def test_predict_standard_cox_profile(self):
        result = predict_standard_cox_profile(
            model=DummyCoxModel(),
            scaler=DummyScaler(),
            kept_columns=self.kept_columns,
            profile_df=self.profile_df,
            survival_times=[12, 36],
        )
        self.assertIn("partial_hazard", result)
        self.assertIn("survival_prob_12m", result)
        self.assertIn("survival_prob_36m", result)
        self.assertGreaterEqual(result["survival_prob_12m"], 0.0)

    def test_predict_time_varying_cox_profile(self):
        result = predict_time_varying_cox_profile(
            model=DummyCoxModel(),
            scaler=DummyScaler(),
            kept_columns=self.kept_columns,
            profile_df=self.profile_df,
        )
        self.assertEqual(result["partial_hazard"], 3.0)

    def test_predict_time_varying_cox_profiles(self):
        result = predict_time_varying_cox_profiles(
            model=DummyCoxModel(),
            scaler=DummyScaler(),
            kept_columns=self.kept_columns,
            profiles_df=self.multi_profile_df,
        )
        self.assertEqual(len(result), 2)
        self.assertIn("partial_hazard", result.columns)
        self.assertEqual(result.loc[0, "partial_hazard"], 3.0)
        self.assertEqual(result.loc[1, "partial_hazard"], 7.0)

    def test_top_positive_negative(self):
        summary_df = pd.DataFrame(
            {
                "feature": ["a", "b", "c", "d"],
                "coef": [0.4, -0.2, 0.1, -0.5],
            }
        )
        positive, negative = top_positive_negative(summary_df, "coef", top_n=2)

        self.assertEqual(list(positive["feature"]), ["a", "c"])
        self.assertEqual(list(negative["feature"]), ["d", "b"])


if __name__ == "__main__":
    unittest.main()