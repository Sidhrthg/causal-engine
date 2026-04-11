"""Tests for DoWhy-based ATE estimation from a DAG file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Only import when running tests; avoid running estimation at collection time.
# pytest collects this module and would otherwise run top-level code.


def _sensor_data_path() -> Path:
    return Path(__file__).resolve().parent / "sensor_test_data.csv"


def _dag_path() -> Path:
    return Path(__file__).resolve().parent.parent / "dag_registry" / "sensor_reliability.dot"


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    path = _sensor_data_path()
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return pd.read_csv(path)


@pytest.fixture
def dag_path() -> Path:
    p = _dag_path()
    if not p.exists():
        pytest.skip(f"DAG file not found: {p}")
    return p


def test_estimate_from_dag_path_returns_result(sensor_df: pd.DataFrame, dag_path: Path):
    """Run ATE estimation and assert result shape and keys."""
    from src.estimate import estimate_from_dag_path

    result = estimate_from_dag_path(
        df=sensor_df,
        treatment="CalibrationInterval",
        outcome="Failure",
        controls=["MaterialType", "Temperature", "Drift"],
        dag_path=str(dag_path),
    )
    assert hasattr(result, "ate")
    assert hasattr(result, "ate_ci")
    assert hasattr(result, "method")
    assert hasattr(result, "model_summary")
    assert result.method is not None
    assert len(result.ate_ci) == 2


def test_estimate_from_dag_path_ate_finite(sensor_df: pd.DataFrame, dag_path: Path):
    """ATE and CI should be finite numbers."""
    from src.estimate import estimate_from_dag_path

    result = estimate_from_dag_path(
        df=sensor_df,
        treatment="CalibrationInterval",
        outcome="Failure",
        controls=["MaterialType", "Temperature", "Drift"],
        dag_path=str(dag_path),
    )
    assert result.ate == result.ate  # not NaN
    assert abs(result.ate) < float("inf")
    low, high = result.ate_ci
    assert low == low and high == high
    assert abs(low) < float("inf") and abs(high) < float("inf")
