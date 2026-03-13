"""
Unit tests for prediction helper utilities used by the Streamlit app.
"""

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
    """Dummy sklearn-like pipeline that returns fixed predictions."""

    @staticmethod
    def predict_proba(_profile_df: pd.DataFrame) -> np.ndarray:
        """Return a fixed class-probability prediction."""
        return np.array([[0.25, 0.75]])

    @staticmethod
    def predict(_profile_df: pd.DataFrame) -> np.ndarray:
        """Return a fixed predicted class."""
        return np.array([1])


class FailingPipeline:
    """Dummy sklearn-like pipeline that raises during probability prediction."""

    @staticmethod
    def predict_proba(_profile_df: pd.DataFrame) -> np.ndarray:
        """Raise an error to simulate a broken pipeline."""
        raise ValueError("boom")

    @staticmethod
    def predict(_profile_df: pd.DataFrame) -> list[int]:
        """Return a placeholder class prediction."""
        return [0]


class DummyScaler:
    """Dummy scaler that mimics a minimal sklearn scaler interface."""

    @staticmethod
    def fit(feature_df: pd.DataFrame):
        """Dummy fit method for sklearn compatibility."""
        return feature_df

    @staticmethod
    def transform(feature_df: pd.DataFrame) -> np.ndarray:
        """Convert a dataframe to a NumPy array."""
        return feature_df.to_numpy(dtype=float)


class DummyPartialHazardResult:
    """Small wrapper mimicking the partial-hazard result interface."""

    def __init__(self, values: list[float]):
        """Store hazard values in a pandas Series."""
        self._series = pd.Series(values)

    @property
    def iloc(self):
        """Expose pandas iloc for single-row indexing compatibility."""
        return self._series.iloc

    def to_numpy(self, dtype=float) -> np.ndarray:
        """Convert stored values to a NumPy array."""
        return self._series.to_numpy(dtype=dtype)


class DummyCoxModel:
    """Dummy Cox model with predictable hazard and survival outputs."""

    @staticmethod
    def predict_partial_hazard(scaled_df: pd.DataFrame) -> DummyPartialHazardResult:
        """Return row-wise sums as dummy partial hazards."""
        values = [float(row.sum()) for _, row in scaled_df.iterrows()]
        return DummyPartialHazardResult(values)

    @staticmethod
    def predict_survival_function(
        scaled_df: pd.DataFrame,
        times: list[int],
    ) -> pd.DataFrame:
        """Return a simple decreasing survival curve over requested times."""
        partial_hazard = float(scaled_df.iloc[0].sum())
        survival_values = [
            max(0.0, 1.0 - 0.01 * partial_hazard - 0.001 * time_value)
            for time_value in times
        ]
        return pd.DataFrame({0: survival_values}, index=times)


class TestPredictionToolsUtils(unittest.TestCase):
    """Tests for prediction helper functions."""

    def setUp(self):
        """Create shared profile fixtures for prediction tests."""
        self.profile_df = pd.DataFrame({"x1": [1.0], "x2": [2.0]})
        self.multi_profile_df = pd.DataFrame(
            {"x1": [1.0, 3.0], "x2": [2.0, 4.0], "month": [0, 12]}
        )
        self.kept_columns = ["x1", "x2"]

    def test_predict_logistic_profile(self):
        """Test logistic prediction output for a valid dummy pipeline."""
        result = predict_logistic_profile(DummyPipeline(), self.profile_df)
        self.assertAlmostEqual(result["predicted_survival_probability"], 0.75)
        self.assertEqual(result["predicted_class"], 1)

    def test_predict_logistic_profile_wraps_error(self):
        """Test that logistic prediction errors are wrapped in RuntimeError."""
        with self.assertRaises(RuntimeError):
            predict_logistic_profile(FailingPipeline(), self.profile_df)

    def test_predict_standard_cox_profile(self):
        """Test standard Cox prediction output structure and survival keys."""
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
        """Test single-row time-varying Cox prediction."""
        result = predict_time_varying_cox_profile(
            model=DummyCoxModel(),
            scaler=DummyScaler(),
            kept_columns=self.kept_columns,
            profile_df=self.profile_df,
        )
        self.assertEqual(result["partial_hazard"], 3.0)

    def test_predict_time_varying_cox_profiles(self):
        """Test multi-row time-varying Cox predictions."""
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
        """Test extraction of top positive and negative coefficients."""
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
