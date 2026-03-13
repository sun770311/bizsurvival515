"""Unit tests for the full pipeline orchestrator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from pipeline.run_pipeline import main


class TestRunPipeline(unittest.TestCase):
    """Unit tests for the full pipeline orchestrator."""

    def _make_args(
        self,
        data_dir: Path,
        output_dir: Path,
        write_joined_to_data_dir: bool = False,
    ) -> SimpleNamespace:
        """Build a mock argparse namespace."""
        return SimpleNamespace(
            data_dir=data_dir,
            output_dir=output_dir,
            licenses_file="licenses_sample.csv",
            service_reqs_file="service_reqs_sample.csv",
            joined_file="joined_dataset.csv",
            write_joined_to_data_dir=write_joined_to_data_dir,
            location_k=25,
            radius_meters=50.0,
            study_end="2026-03-01",
            survival_months=36,
            variance_threshold=1e-8,
            test_size=0.2,
            random_state=42,
            max_iter=5000,
            penalizer=0.1,
        )

    def _create_input_files(self, data_dir: Path) -> tuple[Path, Path]:
        """Create minimal input CSV files required by the pipeline."""
        licenses_path = data_dir / "licenses_sample.csv"
        service_reqs_path = data_dir / "service_reqs_sample.csv"

        licenses_path.write_text("id\n1\n", encoding="utf-8")
        service_reqs_path.write_text("id\n1\n", encoding="utf-8")
        return licenses_path, service_reqs_path

    def _build_test_context(
        self,
        data_dir: Path,
        output_dir: Path,
    ) -> dict:
        """Build shared paths and mocked stage outputs for the full pipeline test."""
        licenses_path, service_reqs_path = self._create_input_files(data_dir)
        joined_path = output_dir / "preprocess" / "joined_dataset.csv"
        geojson_path = output_dir / "geojson" / "businesses.geojson"
        logistic_results = {"accuracy": 0.84, "auc": 0.90}
        cox_results = {
            "standard": {"c_index": 0.71},
            "time_varying": {"c_index": 0.74},
        }

        return {
            "licenses_path": licenses_path,
            "service_reqs_path": service_reqs_path,
            "joined_path": joined_path,
            "geojson_path": geojson_path,
            "logistic_results": logistic_results,
            "cox_results": cox_results,
        }

    def _run_main_with_patches(
        self,
        args: SimpleNamespace,
        context: dict,
    ) -> dict:
        """Run main() with all pipeline stages patched and return mocks."""
        with (
            patch("pipeline.run_pipeline.parse_args", return_value=args),
            patch(
                "pipeline.run_pipeline.run_preprocess_pipeline",
                return_value=context["joined_path"],
            ) as mock_preprocess,
            patch(
                "pipeline.run_pipeline.run_logistic_pipeline",
                return_value=context["logistic_results"],
            ) as mock_logistic,
            patch(
                "pipeline.run_pipeline.run_cox_full_pipeline",
                return_value=context["cox_results"],
            ) as mock_cox,
            patch(
                "pipeline.run_pipeline.run_geojson_pipeline",
                return_value=context["geojson_path"],
            ) as mock_geojson,
            patch("builtins.print") as mock_print,
        ):
            main()

        return {
            "preprocess": mock_preprocess,
            "logistic": mock_logistic,
            "cox": mock_cox,
            "geojson": mock_geojson,
            "print": mock_print,
        }

    def _assert_preprocess_config(
        self,
        mock_preprocess,
        context: dict,
    ) -> None:
        """Assert preprocess stage received the expected config."""
        mock_preprocess.assert_called_once()
        preprocess_config = mock_preprocess.call_args.args[0]

        self.assertEqual(preprocess_config.licenses_path, context["licenses_path"])
        self.assertEqual(
            preprocess_config.service_reqs_path,
            context["service_reqs_path"],
        )
        self.assertEqual(preprocess_config.output_path, context["joined_path"])
        self.assertEqual(preprocess_config.location_k, 25)
        self.assertAlmostEqual(
            preprocess_config.radius_radians,
            50.0 / 6_371_000,
        )

    def _assert_model_configs(
        self,
        mocks: dict,
        output_dir: Path,
        context: dict,
    ) -> None:
        """Assert logistic, cox, and geojson stages received expected configs."""
        mocks["logistic"].assert_called_once()
        logistic_config = mocks["logistic"].call_args.args[0]
        self.assertEqual(logistic_config.data_path, context["joined_path"])
        self.assertEqual(logistic_config.output_dir, output_dir / "logistic")
        self.assertEqual(
            logistic_config.settings.study_end,
            pd.Timestamp("2026-03-01"),
        )
        self.assertEqual(logistic_config.settings.survival_months, 36)
        self.assertEqual(logistic_config.settings.variance_threshold, 1e-8)
        self.assertEqual(logistic_config.settings.test_size, 0.2)
        self.assertEqual(logistic_config.settings.random_state, 42)
        self.assertEqual(logistic_config.settings.max_iter, 5000)

        mocks["cox"].assert_called_once()
        cox_config = mocks["cox"].call_args.args[0]
        self.assertEqual(cox_config.data_path, context["joined_path"])
        self.assertEqual(cox_config.output_dir, output_dir / "cox")
        self.assertEqual(cox_config.study_end, pd.Timestamp("2026-03-01"))
        self.assertEqual(cox_config.variance_threshold, 1e-8)
        self.assertEqual(cox_config.penalizer, 0.1)

        mocks["geojson"].assert_called_once()
        geojson_config = mocks["geojson"].call_args.args[0]
        self.assertEqual(geojson_config.joined_data_path, context["joined_path"])
        self.assertEqual(geojson_config.licenses_path, context["licenses_path"])
        self.assertEqual(
            geojson_config.output_path,
            output_dir / "geojson" / "businesses.geojson",
        )

    def _assert_printed_summary(
        self,
        mock_print,
        output_dir: Path,
        context: dict,
    ) -> None:
        """Assert printed JSON summary contains expected inputs and outputs."""
        mock_print.assert_called_once()
        printed_summary = mock_print.call_args.args[0]
        parsed_summary = json.loads(printed_summary)

        self.assertEqual(
            parsed_summary["inputs"]["licenses_path"],
            str(context["licenses_path"]),
        )
        self.assertEqual(
            parsed_summary["inputs"]["service_reqs_path"],
            str(context["service_reqs_path"]),
        )
        self.assertEqual(
            parsed_summary["outputs"]["joined_dataset"],
            str(context["joined_path"]),
        )
        self.assertEqual(
            parsed_summary["outputs"]["logistic_output_dir"],
            str(output_dir / "logistic"),
        )
        self.assertEqual(
            parsed_summary["outputs"]["cox_output_dir"],
            str(output_dir / "cox"),
        )
        self.assertEqual(
            parsed_summary["outputs"]["geojson_path"],
            str(context["geojson_path"]),
        )
        self.assertEqual(
            parsed_summary["logistic"],
            context["logistic_results"],
        )
        self.assertEqual(parsed_summary["cox"], context["cox_results"])

    def test_main_runs_full_pipeline_successfully(self) -> None:
        """Run the full pipeline and verify stage configs and summary output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            data_dir = base_dir / "data"
            output_dir = base_dir / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)

            context = self._build_test_context(data_dir, output_dir)
            args = self._make_args(data_dir=data_dir, output_dir=output_dir)
            mocks = self._run_main_with_patches(args=args, context=context)

            self.assertTrue((output_dir / "preprocess").exists())
            self.assertTrue((output_dir / "logistic").exists())
            self.assertTrue((output_dir / "cox").exists())
            self.assertTrue((output_dir / "geojson").exists())

            self._assert_preprocess_config(
                mock_preprocess=mocks["preprocess"],
                context=context,
            )
            self._assert_model_configs(
                mocks=mocks,
                output_dir=output_dir,
                context=context,
            )
            self._assert_printed_summary(
                mock_print=mocks["print"],
                output_dir=output_dir,
                context=context,
            )

    def test_main_writes_joined_file_to_data_dir_when_requested(self) -> None:
        """Write the joined dataset to the data directory when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            data_dir = base_dir / "data"
            output_dir = base_dir / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)

            self._create_input_files(data_dir)
            joined_path = data_dir / "joined_dataset.csv"
            args = self._make_args(
                data_dir=data_dir,
                output_dir=output_dir,
                write_joined_to_data_dir=True,
            )

            with (
                patch("pipeline.run_pipeline.parse_args", return_value=args),
                patch(
                    "pipeline.run_pipeline.run_preprocess_pipeline",
                    return_value=joined_path,
                ) as mock_preprocess,
                patch(
                    "pipeline.run_pipeline.run_logistic_pipeline",
                    return_value={},
                ),
                patch(
                    "pipeline.run_pipeline.run_cox_full_pipeline",
                    return_value={},
                ),
                patch(
                    "pipeline.run_pipeline.run_geojson_pipeline",
                    return_value=output_dir / "geojson" / "businesses.geojson",
                ),
                patch("builtins.print"),
            ):
                main()

            preprocess_config = mock_preprocess.call_args.args[0]
            self.assertEqual(preprocess_config.output_path, joined_path)

    def test_main_raises_when_input_files_are_missing(self) -> None:
        """Raise FileNotFoundError when required inputs are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            data_dir = base_dir / "data"
            output_dir = base_dir / "outputs"
            data_dir.mkdir(parents=True, exist_ok=True)

            args = self._make_args(data_dir=data_dir, output_dir=output_dir)

            with patch("pipeline.run_pipeline.parse_args", return_value=args):
                with self.assertRaises(FileNotFoundError) as context:
                    main()

            error_message = str(context.exception)
            self.assertIn("Missing required input file(s)", error_message)
            self.assertIn("licenses_sample.csv", error_message)
            self.assertIn("service_reqs_sample.csv", error_message)


if __name__ == "__main__":
    unittest.main()
