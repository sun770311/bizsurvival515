"""
Unit tests for artifact loading utilities used in the Streamlit application.

These tests validate that artifact loading helpers correctly deserialize
model artifacts and datasets from disk and that the artifact directory
structure is interpreted correctly.
"""

from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.utils import artifact_loader


class TestArtifactLoaderUtils(unittest.TestCase):
    """Tests for artifact loader helper functions."""

    def _write_minimal_logistic_dir(self, base_dir: Path) -> Path:
        """Create a minimal logistic artifact directory for testing."""
        logistic_dir = base_dir / "logistic"
        logistic_dir.mkdir(parents=True, exist_ok=True)

        with (logistic_dir / "logistic_pipeline.pkl").open("wb") as file_obj:
            pickle.dump({"kind": "pipeline"}, file_obj)

        with (logistic_dir / "logistic_kept_columns.pkl").open("wb") as file_obj:
            pickle.dump(["x1", "x2"], file_obj)

        with (logistic_dir / "logistic_dropped_columns.pkl").open("wb") as file_obj:
            pickle.dump(["x3"], file_obj)

        pd.DataFrame(
            {"feature": ["x1"], "coefficient": [0.5]}
        ).to_csv(logistic_dir / "logistic_coefficient_summary.csv", index=False)

        with (logistic_dir / "logistic_evaluation_metrics.json").open(
            "w", encoding="utf-8"
        ) as file_obj:
            json.dump({"accuracy": 0.8}, file_obj)

        pd.DataFrame({"x1": [1.0], "survived_36m": [1]}).to_csv(
            logistic_dir / "business_survival_balanced_dataset.csv",
            index=False,
        )
        pd.DataFrame({"x1": [1.0], "survived_36m": [1]}).to_csv(
            logistic_dir / "X_train_balanced_split.csv",
            index=False,
        )
        pd.DataFrame({"x1": [0.0], "survived_36m": [0]}).to_csv(
            logistic_dir / "X_test_balanced_split.csv",
            index=False,
        )

        return logistic_dir

    def _write_minimal_standard_cox_dir(self, base_dir: Path) -> Path:
        """Create a minimal standard Cox artifact directory."""
        cox_dir = base_dir / "cox_standard"
        cox_dir.mkdir(parents=True, exist_ok=True)

        with (cox_dir / "coxph_model.pkl").open("wb") as file_obj:
            pickle.dump({"kind": "cox_model"}, file_obj)

        with (cox_dir / "coxph_scaler.pkl").open("wb") as file_obj:
            pickle.dump({"kind": "cox_scaler"}, file_obj)

        with (cox_dir / "coxph_kept_columns.pkl").open("wb") as file_obj:
            pickle.dump(["active_license_count"], file_obj)

        with (cox_dir / "coxph_dropped_columns.pkl").open("wb") as file_obj:
            pickle.dump(["unused_col"], file_obj)

        pd.DataFrame(
            {"feature": ["active_license_count"], "coef": [0.1]}
        ).to_csv(cox_dir / "coxph_summary.csv", index=False)

        return cox_dir

    def _write_minimal_time_varying_cox_dir(self, base_dir: Path) -> Path:
        """Create a minimal time-varying Cox artifact directory."""
        cox_dir = base_dir / "cox_time_varying"
        cox_dir.mkdir(parents=True, exist_ok=True)

        with (cox_dir / "cox_time_varying_model.pkl").open("wb") as file_obj:
            pickle.dump({"kind": "tv_model"}, file_obj)

        with (cox_dir / "cox_time_varying_scaler.pkl").open("wb") as file_obj:
            pickle.dump({"kind": "tv_scaler"}, file_obj)

        with (cox_dir / "cox_time_varying_kept_columns.pkl").open("wb") as file_obj:
            pickle.dump(["active_license_count"], file_obj)

        with (cox_dir / "cox_time_varying_dropped_columns.pkl").open("wb") as file_obj:
            pickle.dump(["unused_col"], file_obj)

        pd.DataFrame(
            {"feature": ["active_license_count"], "coef": [0.2]}
        ).to_csv(cox_dir / "cox_time_varying_summary.csv", index=False)

        return cox_dir

    def test_load_logistic_artifacts(self):
        """Verify logistic artifacts are correctly loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            logistic_dir = self._write_minimal_logistic_dir(base_dir)

            with patch.object(artifact_loader, "LOGISTIC_DIR", logistic_dir):
                load_fn = getattr(
                    artifact_loader.load_logistic_artifacts,
                    "__wrapped__",
                    artifact_loader.load_logistic_artifacts,
                )
                loaded = load_fn()

        self.assertEqual(loaded["pipeline"], {"kind": "pipeline"})
        self.assertEqual(loaded["kept_columns"], ["x1", "x2"])
        self.assertEqual(loaded["dropped_columns"], ["x3"])
        self.assertIn("feature", loaded["coef_summary"].columns)
        self.assertEqual(loaded["metrics"]["accuracy"], 0.8)

    def test_load_logistic_reference_data(self):
        """Verify logistic reference datasets load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            logistic_dir = self._write_minimal_logistic_dir(base_dir)

            with patch.object(artifact_loader, "LOGISTIC_DIR", logistic_dir):
                load_fn = getattr(
                    artifact_loader.load_logistic_reference_data,
                    "__wrapped__",
                    artifact_loader.load_logistic_reference_data,
                )
                loaded = load_fn()

        self.assertEqual(set(loaded.keys()), {"businesses", "x_train", "x_test"})
        self.assertFalse(loaded["businesses"].empty)
        self.assertFalse(loaded["x_train"].empty)
        self.assertFalse(loaded["x_test"].empty)

    def test_load_standard_cox_artifacts(self):
        """Verify standard Cox artifacts load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            cox_dir = self._write_minimal_standard_cox_dir(base_dir)

            with patch.object(artifact_loader, "COX_STANDARD_DIR", cox_dir):
                load_fn = getattr(
                    artifact_loader.load_standard_cox_artifacts,
                    "__wrapped__",
                    artifact_loader.load_standard_cox_artifacts,
                )
                loaded = load_fn()

        self.assertEqual(loaded["model"], {"kind": "cox_model"})
        self.assertEqual(loaded["scaler"], {"kind": "cox_scaler"})
        self.assertEqual(loaded["kept_columns"], ["active_license_count"])
        self.assertEqual(loaded["dropped_columns"], ["unused_col"])
        self.assertIn("feature", loaded["summary"].columns)

    def test_load_time_varying_cox_artifacts(self):
        """Verify time-varying Cox artifacts load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            cox_dir = self._write_minimal_time_varying_cox_dir(base_dir)

            with patch.object(artifact_loader, "COX_TIME_VARYING_DIR", cox_dir):
                load_fn = getattr(
                    artifact_loader.load_time_varying_cox_artifacts,
                    "__wrapped__",
                    artifact_loader.load_time_varying_cox_artifacts,
                )
                loaded = load_fn()

        self.assertEqual(loaded["model"], {"kind": "tv_model"})
        self.assertEqual(loaded["scaler"], {"kind": "tv_scaler"})
        self.assertEqual(loaded["kept_columns"], ["active_license_count"])
        self.assertEqual(loaded["dropped_columns"], ["unused_col"])
        self.assertIn("feature", loaded["summary"].columns)


if __name__ == "__main__":
    unittest.main()
