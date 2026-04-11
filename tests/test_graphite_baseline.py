"""Comprehensive tests for graphite baseline scenario and run_scenario integration."""

from __future__ import annotations

from pathlib import Path

import pytest
import pandas as pd

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario


def test_graphite_baseline_golden_metrics():
    """Baseline scenario: no shocks, deterministic; golden metrics must match."""
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    df, metrics = run_scenario(cfg)

    # Basic invariants
    assert len(df) == (cfg.time.end_year - cfg.time.start_year + 1)
    assert metrics["peak_shortage"] >= 0.0
    assert metrics["total_shortage"] >= 0.0
    assert metrics["avg_price"] > 0.0
    assert metrics["final_inventory_cover"] >= 0.0

    # Golden expectations (deterministic because sigma_P=0 and no shocks)
    # These numbers should remain stable unless core equations change.
    # Tolerances are tight to catch drift.
    assert abs(metrics["total_shortage"] - 0.0) < 1e-6
    assert abs(metrics["peak_shortage"] - 0.0) < 1e-6

    # avg_price should remain near 1.0 in steady baseline with balanced supply/demand
    assert abs(metrics["avg_price"] - 1.0) < 1e-6

    # inventory cover should converge to around cover_star; keep tolerance moderate
    assert abs(metrics["final_inventory_cover"] - 0.20) < 1e-6


def test_graphite_baseline_dataframe_structure():
    """DataFrame must have required columns and correct dtypes."""
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    df, _ = run_scenario(cfg)

    required = [
        "year", "K", "I", "P", "Q", "Q_eff", "D",
        "shortage", "tight", "cover",
        "shock_export_restriction", "shock_demand_surge", "shock_capex_shock",
        "shock_stockpile_release", "shock_policy_supply_mult",
        "shock_capacity_supply_mult", "shock_demand_destruction_mult",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    assert df["year"].min() == cfg.time.start_year
    assert df["year"].max() == cfg.time.end_year
    assert df["year"].is_monotonic_increasing
    assert (df["P"] > 0).all()
    assert (df["shortage"] >= 0).all()
    assert (df["cover"] >= 0).all()
    assert df["K"].dtype in (float, "float64")
    assert df["year"].dtype in (int, "int32", "int64")


def test_graphite_baseline_deterministic():
    """Same config and seed must produce identical results."""
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    df1, m1 = run_scenario(cfg)
    df2, m2 = run_scenario(cfg)
    assert m1 == m2
    pd.testing.assert_frame_equal(df1, df2)


def test_graphite_baseline_no_nan_or_inf():
    """Outputs must be finite and non-NaN."""
    cfg = load_scenario("scenarios/graphite_baseline.yaml")
    df, metrics = run_scenario(cfg)
    assert df.notna().all().all(), "DataFrame must not contain NaN"
    assert (df.select_dtypes(include=["number"]).abs() < float("inf")).all().all()
    for k, v in metrics.items():
        assert v == v, f"Metric {k} must not be NaN"
        assert abs(v) < float("inf"), f"Metric {k} must be finite"
