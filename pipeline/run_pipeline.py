"""Run the full business survival modeling pipeline end to end."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pipeline.cox import CoxConfig, run_full_pipeline as run_cox_full_pipeline
from pipeline.logistic import (
    LogisticConfig,
    LogisticModelSettings,
    run_logistic_pipeline,
)
from pipeline.mapbox import GeoJSONConfig, run_geojson_pipeline
from pipeline.preprocess import PipelineConfig, run_pipeline as run_preprocess_pipeline


EARTH_RADIUS_METERS = 6_371_000


@dataclass(frozen=True)
class InputPaths:
    """Container for raw input paths."""

    data_dir: Path
    licenses_path: Path
    service_reqs_path: Path


@dataclass(frozen=True)
class OutputDirs:
    """Container for pipeline output directories."""

    root_dir: Path
    preprocess_dir: Path
    logistic_dir: Path
    cox_dir: Path
    geojson_dir: Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Container for generated artifact file paths."""

    joined_dataset_path: Path
    geojson_path: Path


@dataclass(frozen=True)
class PipelinePaths:
    """Container for all grouped pipeline paths."""

    inputs: InputPaths
    outputs: OutputDirs
    artifacts: ArtifactPaths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the full pipeline runner."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the full sampled-data pipeline: preprocess, logistic, "
            "cox, and geojson."
        )
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


def build_paths(args: argparse.Namespace) -> PipelinePaths:
    """Build all file and directory paths used by the pipeline."""
    output_dirs = OutputDirs(
        root_dir=args.output_dir,
        preprocess_dir=args.output_dir / "preprocess",
        logistic_dir=args.output_dir / "logistic",
        cox_dir=args.output_dir / "cox",
        geojson_dir=args.output_dir / "geojson",
    )

    input_paths = InputPaths(
        data_dir=args.data_dir,
        licenses_path=args.data_dir / args.licenses_file,
        service_reqs_path=args.data_dir / args.service_reqs_file,
    )

    if args.write_joined_to_data_dir:
        joined_dataset_path = args.data_dir / args.joined_file
    else:
        joined_dataset_path = output_dirs.preprocess_dir / args.joined_file

    artifact_paths = ArtifactPaths(
        joined_dataset_path=joined_dataset_path,
        geojson_path=output_dirs.geojson_dir / "businesses.geojson",
    )

    return PipelinePaths(
        inputs=input_paths,
        outputs=output_dirs,
        artifacts=artifact_paths,
    )


def validate_input_paths(paths: PipelinePaths) -> None:
    """Raise an error if any required raw input files are missing."""
    missing_inputs = [
        str(path)
        for path in (
            paths.inputs.licenses_path,
            paths.inputs.service_reqs_path,
        )
        if not path.exists()
    ]
    if missing_inputs:
        raise FileNotFoundError(f"Missing required input file(s): {missing_inputs}")


def ensure_output_dirs(paths: PipelinePaths) -> None:
    """Create all pipeline output directories if they do not already exist."""
    for directory in (
        paths.outputs.preprocess_dir,
        paths.outputs.logistic_dir,
        paths.outputs.cox_dir,
        paths.outputs.geojson_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def build_preprocess_config(
    args: argparse.Namespace,
    paths: PipelinePaths,
) -> PipelineConfig:
    """Build the preprocessing configuration object."""
    radius_radians = args.radius_meters / EARTH_RADIUS_METERS
    return PipelineConfig(
        licenses_path=paths.inputs.licenses_path,
        service_reqs_path=paths.inputs.service_reqs_path,
        output_path=paths.artifacts.joined_dataset_path,
        location_k=args.location_k,
        radius_radians=radius_radians,
    )


def build_logistic_config(
    args: argparse.Namespace,
    paths: PipelinePaths,
    study_end: pd.Timestamp,
) -> LogisticConfig:
    """Build the logistic regression configuration object."""
    return LogisticConfig(
        data_path=paths.artifacts.joined_dataset_path,
        output_dir=paths.outputs.logistic_dir,
        settings=LogisticModelSettings(
            study_end=study_end,
            survival_months=args.survival_months,
            variance_threshold=args.variance_threshold,
            test_size=args.test_size,
            random_state=args.random_state,
            max_iter=args.max_iter,
        ),
    )


def build_cox_config(
    args: argparse.Namespace,
    paths: PipelinePaths,
    study_end: pd.Timestamp,
) -> CoxConfig:
    """Build the Cox modeling configuration object."""
    return CoxConfig(
        data_path=paths.artifacts.joined_dataset_path,
        output_dir=paths.outputs.cox_dir,
        study_end=study_end,
        variance_threshold=args.variance_threshold,
        penalizer=args.penalizer,
    )


def build_summary(
    paths: PipelinePaths,
    logistic_results: dict,
    cox_results: dict,
) -> dict:
    """Assemble a JSON-serializable summary of pipeline inputs and outputs."""
    return {
        "inputs": {
            "licenses_path": str(paths.inputs.licenses_path),
            "service_reqs_path": str(paths.inputs.service_reqs_path),
        },
        "outputs": {
            "joined_dataset": str(paths.artifacts.joined_dataset_path),
            "logistic_output_dir": str(paths.outputs.logistic_dir),
            "cox_output_dir": str(paths.outputs.cox_dir),
            "geojson_path": str(paths.artifacts.geojson_path),
        },
        "logistic": logistic_results,
        "cox": cox_results,
    }


def main() -> None:
    """Run preprocessing, logistic, Cox, and GeoJSON export pipelines."""
    args = parse_args()
    paths = build_paths(args)
    study_end = pd.Timestamp(args.study_end)

    validate_input_paths(paths)
    ensure_output_dirs(paths)

    joined_path = run_preprocess_pipeline(build_preprocess_config(args, paths))
    paths = PipelinePaths(
        inputs=paths.inputs,
        outputs=paths.outputs,
        artifacts=ArtifactPaths(
            joined_dataset_path=joined_path,
            geojson_path=paths.artifacts.geojson_path,
        ),
    )

    logistic_results = run_logistic_pipeline(
        build_logistic_config(args, paths, study_end)
    )

    cox_results = run_cox_full_pipeline(
        build_cox_config(args, paths, study_end)
    )

    geojson_path = run_geojson_pipeline(
        GeoJSONConfig(
            joined_data_path=paths.artifacts.joined_dataset_path,
            licenses_path=paths.inputs.licenses_path,
            output_path=paths.artifacts.geojson_path,
        )
    )

    paths = PipelinePaths(
        inputs=paths.inputs,
        outputs=paths.outputs,
        artifacts=ArtifactPaths(
            joined_dataset_path=paths.artifacts.joined_dataset_path,
            geojson_path=geojson_path,
        ),
    )

    print(
        json.dumps(
            build_summary(
                paths=paths,
                logistic_results=logistic_results,
                cox_results=cox_results,
            ),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
