"""Shared utilities for the data processing and modeling pipeline."""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

STUDY_END = pd.Timestamp("2026-03-01")
VARIANCE_THRESHOLD = 1e-8
CUTOFF_DATE = pd.Timestamp("2026-03-01")
VALID_BOROUGHS = {"Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"}


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box coordinates for spatial filtering."""
    lat_min: float
    lat_max: float
    lng_min: float
    lng_max: float


NYC_BBOX = BoundingBox(40.49, 40.92, -74.27, -73.68)


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Container for retained and dropped feature names."""
    kept_columns: list[str]
    dropped_columns: list[str]


def load_joined_dataset(data_path: Path) -> pd.DataFrame:
    """Load joined dataset and parse date columns."""
    joined = pd.read_csv(data_path)
    if "business_id" in joined.columns:
        joined["business_id"] = joined["business_id"].astype("string").str.strip()
        joined["business_id"] = joined["business_id"].replace(
            {"": pd.NA, "nan": pd.NA, "None": pd.NA}
        )
    if "month" in joined.columns:
        joined["month"] = pd.to_datetime(joined["month"], errors="coerce")
    return joined


def validate_joined_dataset(joined: pd.DataFrame, required_columns: list[str]) -> None:
    """Validate that required columns are present and well-formed."""
    if missing := [c for c in required_columns if c not in joined.columns]:
        raise ValueError(f"Missing required columns: {missing}")

    if "month" in joined.columns and joined["month"].isna().any():
        raise ValueError("Some month values could not be parsed as datetimes.")

    if "business_id" in joined.columns and "month" in joined.columns:
        if joined.duplicated(["business_id", "month"]).any():
            raise ValueError("Duplicate business_id-month rows found in joined dataset.")


def get_model_drop_columns() -> list[str]:
    """Return columns excluded from model fitting."""
    return [
        "business_id", "month", "open", "months_since_first_license",
        "business_category_sum", "complaint_sum", "total_311",
    ]


def save_pickle_artifact(obj: object, output_path: Path) -> Path:
    """Save an object as a pickle file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)
    return output_path


def save_dataframe_artifact(df: pd.DataFrame, output_path: Path) -> Path:
    """Save dataframe artifact as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def save_json_artifact(payload: dict[str, object], output_path: Path) -> Path:
    """Save dictionary artifact as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
    return output_path


def add_standard_modeling_args(parser: argparse.ArgumentParser) -> None:
    """Add standard generic modeling arguments to a parser."""
    parser.add_argument("--data", type=Path, required=True, help="Input CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--study-end", type=str, default=str(STUDY_END.date()))
    parser.add_argument("--variance-threshold", type=float, default=VARIANCE_THRESHOLD)
