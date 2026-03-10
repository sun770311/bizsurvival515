from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.preprocess import PipelineConfig, run_pipeline as run_preprocess_pipeline
from pipeline.logistic import LogisticConfig, run_logistic_pipeline
from pipeline.cox import CoxConfig, run_full_pipeline as run_cox_full_pipeline
from pipeline.mapbox import GeoJSONConfig, run_geojson_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full sampled-data pipeline: preprocess, logistic, cox, and geojson."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("tests/data"),
        help="Directory containing licenses_sample.csv and service_reqs_sample.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where pipeline outputs will be written.",
    )
    parser.add_argument(
        "--licenses-file",
        type=str,
        default="licenses_sample.csv",
        help="Sampled licenses filename inside --data-dir.",
    )
    parser.add_argument(
        "--service-reqs-file",
        type=str,
        default="service_reqs_sample.csv",
        help="Sampled service requests filename inside --data-dir.",
    )
    parser.add_argument(
        "--joined-file",
        type=str,
        default="joined_dataset.csv",
        help="Joined dataset filename to create inside --data-dir or --output-dir.",
    )
    parser.add_argument(
        "--write-joined-to-data-dir",
        action="store_true",
        help="Write joined_dataset.csv into --data-dir instead of --output-dir/preprocess.",
    )
    parser.add_argument(
        "--location-k",
        type=int,
        default=25,
        help="Number of location clusters for preprocessing.",
    )
    parser.add_argument(
        "--radius-meters",
        type=float,
        default=50.0,
        help="Radius in meters for joining 311 requests to businesses.",
    )
    parser.add_argument(
        "--study-end",
        type=str,
        default="2026-03-01",
        help="Study end date in YYYY-MM-DD format for logistic and cox.",
    )
    parser.add_argument(
        "--survival-months",
        type=int,
        default=36,
        help="Survival horizon for logistic regression.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=1e-8,
        help="Variance threshold for logistic and cox feature filtering.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split fraction for logistic regression.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for logistic regression.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for logistic regression.",
    )
    parser.add_argument(
        "--penalizer",
        type=float,
        default=0.1,
        help="Penalizer for Cox models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    study_end = pd.Timestamp(args.study_end)

    data_dir = args.data_dir
    output_dir = args.output_dir
    preprocess_output_dir = output_dir / "preprocess"
    logistic_output_dir = output_dir / "logistic"
    cox_output_dir = output_dir / "cox"
    geojson_output_dir = output_dir / "geojson"

    licenses_path = data_dir / args.licenses_file
    service_reqs_path = data_dir / args.service_reqs_file

    if args.write_joined_to_data_dir:
        joined_output_path = data_dir / args.joined_file
    else:
        joined_output_path = preprocess_output_dir / args.joined_file

    geojson_output_path = geojson_output_dir / "businesses.geojson"

    missing_inputs = [
        str(path)
        for path in [licenses_path, service_reqs_path]
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(
            f"Missing required input file(s): {missing_inputs}"
        )

    preprocess_output_dir.mkdir(parents=True, exist_ok=True)
    logistic_output_dir.mkdir(parents=True, exist_ok=True)
    cox_output_dir.mkdir(parents=True, exist_ok=True)
    geojson_output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess sampled raw data into joined_dataset.csv
    earth_radius_meters = 6_371_000
    radius_radians = args.radius_meters / earth_radius_meters

    preprocess_config = PipelineConfig(
        licenses_path=licenses_path,
        service_reqs_path=service_reqs_path,
        output_path=joined_output_path,
        location_k=args.location_k,
        radius_radians=radius_radians,
    )
    joined_path = run_preprocess_pipeline(preprocess_config)

    # 2) Logistic regression pipeline
    logistic_config = LogisticConfig(
        data_path=joined_path,
        output_dir=logistic_output_dir,
        study_end=study_end,
        survival_months=args.survival_months,
        variance_threshold=args.variance_threshold,
        test_size=args.test_size,
        random_state=args.random_state,
        max_iter=args.max_iter,
    )
    logistic_results = run_logistic_pipeline(logistic_config)

    # 3) Cox pipeline (both time-varying and standard)
    cox_config = CoxConfig(
        data_path=joined_path,
        output_dir=cox_output_dir,
        study_end=study_end,
        variance_threshold=args.variance_threshold,
        penalizer=args.penalizer,
    )
    cox_results = run_cox_full_pipeline(cox_config)

    # 4) Mapbox GeoJSON export
    geojson_config = GeoJSONConfig(
        joined_data_path=joined_path,
        licenses_path=licenses_path,
        output_path=geojson_output_path,
    )
    geojson_path = run_geojson_pipeline(geojson_config)

    summary = {
        "inputs": {
            "licenses_path": str(licenses_path),
            "service_reqs_path": str(service_reqs_path),
        },
        "outputs": {
            "joined_dataset": str(joined_path),
            "logistic_output_dir": str(logistic_output_dir),
            "cox_output_dir": str(cox_output_dir),
            "geojson_path": str(geojson_path),
        },
        "logistic": logistic_results,
        "cox": cox_results,
    }

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()