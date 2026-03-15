"""Provide shared constants, validation helpers, artifact I/O utilities, and CLI helpers.

This module centralizes common utilities used across the business survival
pipeline. It defines shared constants for study-window control, feature
filtering, and required joined-dataset structure; provides validation helpers
for the joined monthly panel; offers reusable functions for loading and saving
pipeline artifacts; and exposes a helper for adding standard command-line
arguments used by modeling scripts.

Constants:
- STUDY_END:
  Default study end date used across modeling pipelines.
- VARIANCE_THRESHOLD:
  Default variance threshold used for low-variance feature filtering.
- REQUIRED_JOINED_COLUMNS:
  Required columns expected in the joined monthly panel dataset.
- LOCATION_FEATURE_COLUMNS:
  Standard location-related feature columns used across pipeline stages.
- JOINED_PANEL_TRAILING_COLUMNS:
  Standard trailing columns expected in the joined panel layout.

Classes:
- FeatureSelectionResult:
  Stores lists of retained and dropped feature names after feature selection.

Functions:
- load_joined_dataset:
  Load the joined dataset and parse the month column.
- validate_joined_dataset:
  Validate required columns, parsed dates, and panel-row uniqueness.
- restrict_to_study_window:
  Restrict the joined panel to rows within the study window.
- save_pickle_artifact:
  Save an object as a pickle artifact.
- save_dataframe_artifact:
  Save a dataframe artifact as CSV.
- save_json_artifact:
  Save a dictionary artifact as JSON.
- add_standard_modeling_args:
  Add shared CLI arguments used by the Cox and logistic pipelines.
"""

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
    """Store retained and dropped feature names after feature selection.

    Attributes:
        kept_columns: Feature names retained for modeling.
        dropped_columns: Feature names removed during feature selection.
    """

    kept_columns: list[str]
    dropped_columns: list[str]


def load_joined_dataset(data_path: Path) -> pd.DataFrame:
    """Load the joined monthly panel dataset and parse the month column.

    Args:
        data_path: Path to the joined dataset CSV file.

    Returns:
        A dataframe containing the joined dataset, with the ``month`` column
        parsed as datetime where possible.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        OSError: If an I/O error occurs while reading the file.
    """
    joined = pd.read_csv(data_path)
    joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def validate_joined_dataset(joined: pd.DataFrame) -> None:
    """Validate required columns, parsed month values, and business-month uniqueness.

    Args:
        joined: Joined monthly panel dataframe to validate.

    Returns:
        None.

    Raises:
        ValueError: If one or more required columns are missing.
        ValueError: If any ``month`` values could not be parsed as datetimes.
        ValueError: If duplicate ``business_id``-``month`` rows are found.
    """
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
    """Restrict the joined monthly panel to rows on or before the study end date.

    Args:
        joined: Joined monthly panel dataframe.
        study_end: Final month to retain in the study window.

    Returns:
        A filtered copy of the joined dataframe containing only rows whose
        ``month`` is less than or equal to ``study_end``.
    """
    return joined.loc[joined["month"] <= study_end].copy()


def save_pickle_artifact(obj: object, output_path: Path) -> Path:
    """Save a Python object to disk as a pickle artifact.

    Args:
        obj: Python object to serialize and save.
        output_path: Destination path for the pickle file.

    Returns:
        The path where the pickle artifact was written.

    Raises:
        OSError: If an I/O error occurs while writing the file.
        pickle.PicklingError: If the object cannot be pickled successfully.
    """
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return output_path


def save_dataframe_artifact(df: pd.DataFrame, output_path: Path) -> Path:
    """Save a dataframe artifact to disk as a CSV file.

    Args:
        df: Dataframe to save.
        output_path: Destination path for the CSV file.

    Returns:
        The path where the CSV artifact was written.

    Raises:
        OSError: If an I/O error occurs while writing the file.
    """
    df.to_csv(output_path, index=False)
    return output_path


def save_json_artifact(payload: dict[str, object], output_path: Path) -> Path:
    """Save a dictionary artifact to disk as a JSON file.

    Args:
        payload: Dictionary payload to serialize as JSON.
        output_path: Destination path for the JSON file.

    Returns:
        The path where the JSON artifact was written.

    Raises:
        OSError: If an I/O error occurs while writing the file.
        TypeError: If the payload contains values that are not JSON serializable.
    """
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
    return output_path


def add_standard_modeling_args(parser: argparse.ArgumentParser) -> None:
    """Add shared command-line arguments used by Cox and logistic modeling scripts.

    Args:
        parser: Argument parser to which shared modeling arguments will be added.

    Returns:
        None.
    """
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
