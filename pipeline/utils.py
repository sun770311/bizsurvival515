"""Shared utilities for the business survival pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


STUDY_END = pd.Timestamp("2026-03-01")
VARIANCE_THRESHOLD = 1e-8

REQUIRED_JOINED_COLUMNS = [
    "business_id",
    "month",
    "active_license_count",
    "total_311",
    "open",
    "months_since_first_license",
    "location_cluster",
    "location_cluster_lat",
    "location_cluster_lng",
    "business_latitude",
    "business_longitude",
    "business_category_sum",
    "complaint_sum",
]

LOCATION_FEATURE_COLUMNS = [
    "business_latitude",
    "business_longitude",
    "location_cluster",
    "location_cluster_lat",
    "location_cluster_lng",
]

JOINED_PANEL_TRAILING_COLUMNS = [
    "open",
    "months_since_first_license",
    "location_cluster",
    "location_cluster_lat",
    "location_cluster_lng",
    "business_latitude",
    "business_longitude",
    "business_category_sum",
    "complaint_sum",
]


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for retained and dropped feature names."""

    kept_columns: list[str]
    dropped_columns: list[str]


def load_joined_dataset(data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse the month column."""
    joined = pd.read_csv(data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate required columns and uniqueness of panel rows."""
    missing_columns = [
        column for column in REQUIRED_JOINED_COLUMNS if column not in joined.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if joined["month"].isna().any():
        raise ValueError("Some month values could not be parsed as datetimes.")

    if joined.duplicated(["business_id", "month"]).any():
        raise ValueError("Duplicate business_id-month rows found in joined dataset.")


def restrict_to_study_window(
    joined: pd.DataFrame,
    study_end: pd.Timestamp,
) -> pd.DataFrame:
    """Restrict panel rows to the study window."""
    return joined.loc[joined["month"] <= study_end].copy()


def save_pickle_artifact(obj: object, output_path: Path) -> Path:
    """Save object as pickle artifact."""
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return output_path


def save_dataframe_artifact(df: pd.DataFrame, output_path: Path) -> Path:
    """Save dataframe artifact as CSV."""
    df.to_csv(output_path, index=False)
    return output_path


def save_json_artifact(payload: dict[str, object], output_path: Path) -> Path:
    """Save dictionary artifact as JSON."""
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
    return output_path


def add_standard_modeling_args(parser: argparse.ArgumentParser) -> None:
    """Add shared CLI arguments used by the Cox and logistic pipelines."""
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to joined_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model artifacts will be written",
    )
    parser.add_argument(
        "--study-end",
        type=str,
        default=str(STUDY_END.date()),
        help="Study end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help="Variance threshold for feature filtering",
    )
