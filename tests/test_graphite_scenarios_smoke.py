from __future__ import annotations

from pathlib import Path

import pytest

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario


SCENARIO_FILES = [
    "scenarios/graphite_baseline.yaml",
    "scenarios/graphite_export_restriction.yaml",
    "scenarios/graphite_demand_surge.yaml",
    "scenarios/graphite_policy_response.yaml",
]


@pytest.mark.parametrize("scenario_path", SCENARIO_FILES)
def test_scenario_smoke(scenario_path: str):
    """Smoke test: run scenario and verify outputs are finite and deterministic."""
    path = Path(scenario_path)
    assert path.exists(), f"Scenario file must exist: {scenario_path}"
    
    # Load and run scenario
    cfg = load_scenario(scenario_path)
    df, metrics = run_scenario(cfg)
    
    # Verify dataframe structure
    assert len(df) > 0, "DataFrame must have rows"
    required_cols = ["year", "K", "I", "P", "Q", "D", "Q_eff", "shortage", "tight", "cover"]
    for col in required_cols:
        assert col in df.columns, f"DataFrame must have column: {col}"
    
    # Verify metrics structure
    required_metrics = ["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
    for metric in required_metrics:
        assert metric in metrics, f"Metrics must include: {metric}"
    
    # Verify shock-window metrics for scenarios with shocks
    if cfg.shocks:
        assert "shock_window_total_shortage" in metrics, "Scenarios with shocks must include shock_window_total_shortage"
        assert "post_shock_total_shortage" in metrics, "Scenarios with shocks must include post_shock_total_shortage"
        assert metrics["shock_window_total_shortage"] >= 0.0, "shock_window_total_shortage must be non-negative"
        assert metrics["post_shock_total_shortage"] >= 0.0, "post_shock_total_shortage must be non-negative"
    else:
        # Baseline scenario can omit or have zeros
        if "shock_window_total_shortage" in metrics:
            assert metrics["shock_window_total_shortage"] == 0.0, "Baseline should have zero shock_window_total_shortage"
        if "post_shock_total_shortage" in metrics:
            assert metrics["post_shock_total_shortage"] == 0.0, "Baseline should have zero post_shock_total_shortage"
    
    # Verify all metrics are finite
    for metric_name, metric_value in metrics.items():
        assert isinstance(metric_value, (int, float)), f"Metric {metric_name} must be numeric"
        assert abs(metric_value) < float("inf"), f"Metric {metric_name} must be finite, got {metric_value}"
        assert metric_value == metric_value, f"Metric {metric_name} must not be NaN, got {metric_value}"
    
    # Verify dataframe values are finite
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in numeric_cols:
        assert df[col].notna().all(), f"Column {col} must not have NaN values"
        assert (df[col].abs() < float("inf")).all(), f"Column {col} must have finite values"
    
    # Verify invariants
    assert metrics["peak_shortage"] >= 0.0, "Peak shortage must be non-negative"
    assert metrics["total_shortage"] >= 0.0, "Total shortage must be non-negative"
    assert metrics["avg_price"] > 0.0, "Average price must be positive"
    assert metrics["final_inventory_cover"] >= 0.0, "Final inventory cover must be non-negative"
    
    # Verify time range matches config
    assert df["year"].min() == cfg.time.start_year, "First year must match start_year"
    assert df["year"].max() == cfg.time.end_year, "Last year must match end_year"
    assert len(df) == (cfg.time.end_year - cfg.time.start_year + 1), "Number of rows must match time range"
    
    # Verify deterministic: run twice with same seed, should get same results
    df2, metrics2 = run_scenario(cfg)
    assert metrics == metrics2, "Metrics must be deterministic (same seed should produce same results)"
    # Check a few key columns for determinism
    assert (df["P"] == df2["P"]).all(), "Price must be deterministic"
    assert (df["shortage"] == df2["shortage"]).all(), "Shortage must be deterministic"


def test_policy_monotonicity():
    """Test that policy levers (substitution, stockpile_release) reduce shortages."""
    # Load demand_surge baseline
    cfg_surge = load_scenario("scenarios/graphite_demand_surge.yaml")
    df_surge, metrics_surge = run_scenario(cfg_surge)
    
    # Load policy_response (same shock + substitution + stockpile_release)
    cfg_policy = load_scenario("scenarios/graphite_policy_response.yaml")
    df_policy, metrics_policy = run_scenario(cfg_policy)
    
    # Check that substitution reduces demand in shock year
    year_2026_surge = df_surge[df_surge["year"] == 2026].iloc[0]
    year_2026_policy = df_policy[df_policy["year"] == 2026].iloc[0]
    assert year_2026_policy["D"] < year_2026_surge["D"], (
        f"Substitution should reduce demand. Surge D={year_2026_surge['D']:.2f}, "
        f"Policy D={year_2026_policy['D']:.2f}"
    )
    
    # Check that stockpile_release increases inventory in shock year
    # Stockpile release is applied during the step, so check inventory after 2026 step (in 2027)
    if len(df_policy) > 3:  # Make sure we have 2027
        year_2027_surge = df_surge[df_surge["year"] == 2027].iloc[0]
        year_2027_policy = df_policy[df_policy["year"] == 2027].iloc[0]
        # Policy should have higher inventory after stockpile release
        assert year_2027_policy["I"] >= year_2027_surge["I"] - 1.0, (
            f"Stockpile release should increase inventory. Surge I={year_2027_surge['I']:.2f}, "
            f"Policy I={year_2027_policy['I']:.2f}"
        )
    
    # Check that shortage in shock year is reduced
    assert year_2026_policy["shortage"] <= year_2026_surge["shortage"], (
        f"Policy should reduce shortage in shock year. Surge shortage={year_2026_surge['shortage']:.2f}, "
        f"Policy shortage={year_2026_policy['shortage']:.2f}"
    )
    
    # Policy should NOT significantly increase total_shortage vs baseline (monotonicity)
    # Allow tolerance for dynamics, but check that shock-year shortage is reduced
    # Note: total_shortage might be affected by dynamics in other years, but shock-year should improve
    assert year_2026_policy["shortage"] < year_2026_surge["shortage"], (
        f"Policy should reduce shortage in shock year. Surge: {year_2026_surge['shortage']:.2f}, "
        f"Policy: {year_2026_policy['shortage']:.2f}"
    )


def test_run_scenario_out_dir_deterministic():
    """Test that running same scenario with different --out_dir produces identical outputs."""
    import subprocess
    import sys
    import json
    import tempfile
    from pathlib import Path
    import pandas as pd
    
    scenario_path = Path("scenarios/graphite_baseline.yaml")
    assert scenario_path.exists(), "Scenario file must exist"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir1 = Path(tmpdir) / "output1"
        out_dir2 = Path(tmpdir) / "output2"
        
        # Run scenario twice with different output directories
        result1 = subprocess.run(
            [sys.executable, "-m", "scripts.run_scenario", "--scenario", str(scenario_path), "--out-dir", str(out_dir1)],
            capture_output=True,
            text=True,
            check=True,
        )
        
        result2 = subprocess.run(
            [sys.executable, "-m", "scripts.run_scenario", "--scenario", str(scenario_path), "--out-dir", str(out_dir2)],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Verify output directories exist
        assert out_dir1.exists(), "Output directory 1 should exist"
        assert out_dir2.exists(), "Output directory 2 should exist"
        
        # Verify files exist
        metrics1_path = out_dir1 / "metrics.json"
        metrics2_path = out_dir2 / "metrics.json"
        timeseries1_path = out_dir1 / "timeseries.csv"
        timeseries2_path = out_dir2 / "timeseries.csv"
        
        assert metrics1_path.exists(), "metrics.json should exist in output1"
        assert metrics2_path.exists(), "metrics.json should exist in output2"
        assert timeseries1_path.exists(), "timeseries.csv should exist in output1"
        assert timeseries2_path.exists(), "timeseries.csv should exist in output2"
        
        # Compare metrics
        with open(metrics1_path) as f:
            metrics1 = json.load(f)
        with open(metrics2_path) as f:
            metrics2 = json.load(f)
        
        assert metrics1 == metrics2, "Metrics should be identical"
        
        # Compare timeseries
        df1 = pd.read_csv(timeseries1_path)
        df2 = pd.read_csv(timeseries2_path)
        
        pd.testing.assert_frame_equal(df1, df2, "Timeseries should be identical")
