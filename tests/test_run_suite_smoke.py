from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_suite import run_suite


def test_run_suite_smoke():
    """Smoke test: run suite on all scenarios and verify outputs."""
    scenarios_dir = Path("scenarios")
    output_base = Path("runs")
    
    # Run suite
    summary_df = run_suite(scenarios_dir, output_base)
    
    # Verify summary has expected number of rows (4 scenarios)
    assert len(summary_df) == 4, f"Expected 4 scenarios, got {len(summary_df)}"
    
    # Verify required columns exist
    required_cols = ["scenario", "scenario_file", "total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
    for col in required_cols:
        assert col in summary_df.columns, f"Summary must have column: {col}"
    
    # Verify all scenarios are present
    expected_scenarios = {"graphite_baseline", "graphite_export_restriction", "graphite_demand_surge", "graphite_policy_response"}
    actual_scenarios = set(summary_df["scenario"].values)
    assert actual_scenarios == expected_scenarios, (
        f"Expected scenarios {expected_scenarios}, got {actual_scenarios}"
    )
    
    # Verify metrics are finite
    metric_cols = ["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
    for col in metric_cols:
        assert summary_df[col].notna().all(), f"Column {col} must not have NaN values"
        assert (summary_df[col].abs() < float("inf")).all(), f"Column {col} must have finite values"
    
    # Verify shock-window metrics exist for scenarios with shocks
    shock_scenarios = {"graphite_export_restriction", "graphite_demand_surge", "graphite_policy_response"}
    if "shock_window_total_shortage" in summary_df.columns:
        for scenario in shock_scenarios:
            row = summary_df[summary_df["scenario"] == scenario].iloc[0]
            assert "shock_window_total_shortage" in row.index, f"{scenario} should have shock_window_total_shortage"
            assert pd.notna(row["shock_window_total_shortage"]), f"{scenario} shock_window_total_shortage should not be NaN"
    
    # Verify no errors
    if "error" in summary_df.columns:
        errors = summary_df[summary_df["error"].notna()]
        assert len(errors) == 0, f"Some scenarios had errors: {errors[['scenario', 'error']].to_dict('records')}"

