"""Load cached model artifacts and reference datasets for the NYC Business Survival app.

Inputs:
- Logistic artifact files under ``artifacts/logistic/``
- Standard Cox artifact files under ``artifacts/cox_standard/``
- Time-varying Cox artifact files under ``artifacts/cox_time_varying/``

Processing steps:
- Resolve artifact directory paths relative to the app directory
- Load serialized pickle, CSV, and JSON artifacts from disk
- Package loaded artifacts into dictionaries for downstream app use
- Cache loaded resources and datasets with Streamlit decorators

Outputs:
- Cached dictionaries of trained model artifacts
- Cached dictionaries of reference datasets used by the app

Functions:
- _load_pickle:
  Load a pickled artifact from disk with a helpful missing-file error.
- load_logistic_artifacts:
  Load trained logistic regression artifacts and evaluation outputs.
- load_logistic_reference_data:
  Load logistic reference datasets used for simulation and comparison.
- load_standard_cox_artifacts:
  Load trained standard Cox model artifacts.
- load_time_varying_cox_artifacts:
  Load trained time-varying Cox model artifacts.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = APP_DIR / "artifacts"

LOGISTIC_DIR = ARTIFACT_DIR / "logistic"
COX_STANDARD_DIR = ARTIFACT_DIR / "cox_standard"
COX_TIME_VARYING_DIR = ARTIFACT_DIR / "cox_time_varying"


def _load_pickle(path: Path) -> Any:
    """Load a pickled artifact from disk.

    Args:
        path: Path to the pickle artifact file.

    Returns:
        The deserialized Python object stored in the pickle file.

    Raises:
        FileNotFoundError: If the specified artifact file does not exist.
        pickle.UnpicklingError: If the file cannot be unpickled successfully.
        OSError: If an I/O error occurs while reading the file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected artifact not found: {path}\n"
            f"Contents of parent directory: {[p.name for p in path.parent.glob('*')]}"
        )
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


@st.cache_resource
def load_logistic_artifacts() -> dict[str, Any]:
    """Load trained logistic regression model artifacts and evaluation outputs.

    Args:
        None.

    Returns:
        A dictionary containing:
        - ``pipeline``: fitted sklearn logistic pipeline
        - ``kept_columns``: feature columns retained by the model
        - ``dropped_columns``: feature columns removed during preprocessing
        - ``coef_summary``: logistic coefficient summary dataframe
        - ``metrics``: evaluation metrics loaded from JSON

    Raises:
        FileNotFoundError: If one or more required artifact files are missing.
        pickle.UnpicklingError: If a pickle artifact cannot be unpickled.
        pd.errors.EmptyDataError: If a required CSV artifact is empty.
        json.JSONDecodeError: If the metrics JSON file is invalid.
        OSError: If an I/O error occurs while reading an artifact.
    """
    return {
        "pipeline": _load_pickle(LOGISTIC_DIR / "logistic_pipeline.pkl"),
        "kept_columns": _load_pickle(LOGISTIC_DIR / "logistic_kept_columns.pkl"),
        "dropped_columns": _load_pickle(LOGISTIC_DIR / "logistic_dropped_columns.pkl"),
        "coef_summary": pd.read_csv(LOGISTIC_DIR / "logistic_coefficient_summary.csv"),
        "metrics": json.loads(
            (LOGISTIC_DIR / "logistic_evaluation_metrics.json").read_text(
                encoding="utf-8"
            )
        ),
    }


@st.cache_data
def load_logistic_reference_data() -> dict[str, pd.DataFrame]:
    """Load reference datasets used for logistic model evaluation and simulation.

    Args:
        None.

    Returns:
        A dictionary containing:
        - ``businesses``: balanced business survival dataset
        - ``x_train``: training split dataframe
        - ``x_test``: test split dataframe

    Raises:
        FileNotFoundError: If one or more required CSV files are missing.
        pd.errors.EmptyDataError: If a required CSV file is empty.
        OSError: If an I/O error occurs while reading a dataset file.
    """
    return {
        "businesses": pd.read_csv(LOGISTIC_DIR / "business_survival_balanced_dataset.csv"),
        "x_train": pd.read_csv(LOGISTIC_DIR / "X_train_balanced_split.csv"),
        "x_test": pd.read_csv(LOGISTIC_DIR / "X_test_balanced_split.csv"),
    }


@st.cache_resource
def load_standard_cox_artifacts() -> dict[str, Any]:
    """Load artifacts for the standard Cox proportional hazards model.

    Args:
        None.

    Returns:
        A dictionary containing:
        - ``model``: fitted CoxPH model
        - ``scaler``: fitted feature scaler
        - ``kept_columns``: feature columns retained by the model
        - ``dropped_columns``: feature columns removed during preprocessing
        - ``summary``: Cox coefficient summary dataframe

    Raises:
        FileNotFoundError: If one or more required artifact files are missing.
        pickle.UnpicklingError: If a pickle artifact cannot be unpickled.
        pd.errors.EmptyDataError: If the summary CSV is empty.
        OSError: If an I/O error occurs while reading an artifact.
    """
    return {
        "model": _load_pickle(COX_STANDARD_DIR / "coxph_model.pkl"),
        "scaler": _load_pickle(COX_STANDARD_DIR / "coxph_scaler.pkl"),
        "kept_columns": _load_pickle(COX_STANDARD_DIR / "coxph_kept_columns.pkl"),
        "dropped_columns": _load_pickle(COX_STANDARD_DIR / "coxph_dropped_columns.pkl"),
        "summary": pd.read_csv(COX_STANDARD_DIR / "coxph_summary.csv"),
    }


@st.cache_resource
def load_time_varying_cox_artifacts() -> dict[str, Any]:
    """Load artifacts for the time-varying Cox survival model.

    Args:
        None.

    Returns:
        A dictionary containing:
        - ``model``: fitted time-varying Cox model
        - ``scaler``: fitted feature scaler
        - ``kept_columns``: feature columns retained by the model
        - ``dropped_columns``: feature columns removed during preprocessing
        - ``summary``: time-varying Cox coefficient summary dataframe

    Raises:
        FileNotFoundError: If one or more required artifact files are missing.
        pickle.UnpicklingError: If a pickle artifact cannot be unpickled.
        pd.errors.EmptyDataError: If the summary CSV is empty.
        OSError: If an I/O error occurs while reading an artifact.
    """
    return {
        "model": _load_pickle(COX_TIME_VARYING_DIR / "cox_time_varying_model.pkl"),
        "scaler": _load_pickle(COX_TIME_VARYING_DIR / "cox_time_varying_scaler.pkl"),
        "kept_columns": _load_pickle(
            COX_TIME_VARYING_DIR / "cox_time_varying_kept_columns.pkl"
        ),
        "dropped_columns": _load_pickle(
            COX_TIME_VARYING_DIR / "cox_time_varying_dropped_columns.pkl"
        ),
        "summary": pd.read_csv(COX_TIME_VARYING_DIR / "cox_time_varying_summary.csv"),
    }
