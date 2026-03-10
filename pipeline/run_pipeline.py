"""Main entry point to execute the complete data processing and modeling pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.preprocess import PipelineConfig
from pipeline.preprocess import run_pipeline as run_preprocess_pipeline
from pipeline.logistic import LogisticConfig, ModelingParams, run_logistic_pipeline
from pipeline.cox import CoxConfig
from pipeline.cox import run_full_pipeline as run_cox_full_pipeline
from pipeline.mapbox import GeoJSONConfig, run_geojson_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline configuration."""
    parser = argparse.ArgumentParser(
        description="Run the full pipeline: preprocess, logistic, cox, geojson."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("tests/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--licenses-file", type=str, default="licenses_sample.csv")
    parser.add_argument("--service-reqs-file", type=str, default="service_reqs_sample.csv")
    parser.add_argument("--joined-file", type=str, default="joined_dataset.csv")
    parser.add_argument("--write-joined-to-data-dir", action="store_true")
    parser.add_argument("--location-k", type=int, default=25)
    parser.add_argument("--radius-meters", type=float, default=50.0)
    parser.add_argument("--study-end", type=str, default="2026-03-01")
    parser.add_argument("--survival-months", type=int, default=36)
    parser.add_argument("--variance-threshold", type=float, default=1e-8)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--penalizer", type=float, default=0.1)
    return parser.parse_args()


def get_joined_output_path(args: argparse.Namespace, preprocess_out: Path) -> Path:
    """Determine the path for the joined output dataset."""
    if args.write_joined_to_data_dir:
        return args.data_dir / args.joined_file
    return preprocess_out / args.joined_file


def execute_preprocess(args: argparse.Namespace, joined_path: Path) -> Path:
    """Execute the preprocessing pipeline."""
    earth_radius_meters = 6_371_000
    radius_radians = args.radius_meters / earth_radius_meters
    preprocess_config = PipelineConfig(
        licenses_path=args.data_dir / args.licenses_file,
        service_reqs_path=args.data_dir / args.service_reqs_file,
        output_path=joined_path,
        location_k=args.location_k,
        radius_radians=radius_radians,
    )
    return run_preprocess_pipeline(preprocess_config)


def main() -> None:
    """Execute the main data processing and modeling workflow."""
    args = parse_args()
    study_end = pd.Timestamp(args.study_end)

    preprocess_out = args.output_dir / "preprocess"
    logistic_out = args.output_dir / "logistic"
    cox_out = args.output_dir / "cox"
    geojson_out = args.output_dir / "geojson"

    for directory in [preprocess_out, logistic_out, cox_out, geojson_out]:
        directory.mkdir(parents=True, exist_ok=True)

    joined_output_path = get_joined_output_path(args, preprocess_out)

    missing_inputs = [
        str(path)
        for path in [args.data_dir / args.licenses_file, args.data_dir / args.service_reqs_file]
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required input file(s): {missing_inputs}")

    joined_path = execute_preprocess(args, joined_output_path)

    logistic_params = ModelingParams(
        survival_months=args.survival_months,
        variance_threshold=args.variance_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    logistic_config = LogisticConfig(
        data_path=joined_path,
        output_dir=logistic_out,
        study_end=study_end,
        params=logistic_params,
    )
    logistic_results = run_logistic_pipeline(logistic_config)

    cox_config = CoxConfig(
        data_path=joined_path,
        output_dir=cox_out,
        study_end=study_end,
        variance_threshold=args.variance_threshold,
        penalizer=args.penalizer,
    )
    cox_results = run_cox_full_pipeline(cox_config)

    geojson_config = GeoJSONConfig(
        joined_data_path=joined_path,
        licenses_path=args.data_dir / args.licenses_file,
        output_path=geojson_out / "businesses.geojson",
    )
    geojson_path = run_geojson_pipeline(geojson_config)

    summary = {
        "inputs": {
            "licenses_path": str(args.data_dir / args.licenses_file),
            "service_reqs_path": str(args.data_dir / args.service_reqs_file),
        },
        "outputs": {
            "joined_dataset": str(joined_path),
            "logistic_output_dir": str(logistic_out),
            "cox_output_dir": str(cox_out),
            "geojson_path": str(geojson_path),
        },
        "logistic": logistic_results,
        "cox": cox_results,
    }

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
