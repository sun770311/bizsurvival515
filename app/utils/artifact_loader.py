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
    if not path.exists():
        raise FileNotFoundError(
            f"Expected artifact not found: {path}\n"
            f"Contents of parent directory: {[p.name for p in path.parent.glob('*')]}"
        )
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


@st.cache_resource
def load_logistic_artifacts() -> dict[str, Any]:
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
    return {
        "businesses": pd.read_csv(LOGISTIC_DIR / "business_survival_balanced_dataset.csv"),
        "x_train": pd.read_csv(LOGISTIC_DIR / "X_train_balanced_split.csv"),
        "x_test": pd.read_csv(LOGISTIC_DIR / "X_test_balanced_split.csv"),
    }


@st.cache_resource
def load_standard_cox_artifacts() -> dict[str, Any]:
    return {
        "model": _load_pickle(COX_STANDARD_DIR / "coxph_model.pkl"),
        "scaler": _load_pickle(COX_STANDARD_DIR / "coxph_scaler.pkl"),
        "kept_columns": _load_pickle(COX_STANDARD_DIR / "coxph_kept_columns.pkl"),
        "dropped_columns": _load_pickle(COX_STANDARD_DIR / "coxph_dropped_columns.pkl"),
        "summary": pd.read_csv(COX_STANDARD_DIR / "coxph_summary.csv"),
    }


@st.cache_resource
def load_time_varying_cox_artifacts() -> dict[str, Any]:
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