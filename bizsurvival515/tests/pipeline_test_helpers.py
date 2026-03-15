"""
Shared pipeline setup helpers for unit tests.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path

from bizsurvival515.pipeline.cox import CoxConfig, run_standard_cox_pipeline
from bizsurvival515.pipeline.logistic import LogisticConfig, run_logistic_pipeline
from bizsurvival515.tests.test_helpers import TEST_DATA_DIR


@contextmanager
def fitted_cox_output_dir():
    """Yield a temporary directory containing fitted Cox artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = CoxConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=tmp_path,
        )
        run_standard_cox_pipeline(config)
        yield tmp_path


@contextmanager
def fitted_logistic_output_dir():
    """Yield a temporary directory containing fitted logistic artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        config = LogisticConfig(
            data_path=TEST_DATA_DIR / "joined_dataset.csv",
            output_dir=tmp_path,
        )
        run_logistic_pipeline(config)
        yield tmp_path
