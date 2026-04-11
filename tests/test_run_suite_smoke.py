from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_suite import run_suite


def test_run_suite_smoke():
    """Smoke test: run suite on all scenarios and verify outputs."""
    scenarios_dir = Path("scenarios")
    output_base = Path("runs")

    # Run suite (discovers all .yaml in scenarios/)
    summary_df = run_suite(scenarios_dir, output_base)

    assert len(summary_df) >= 1, "Suite must run at least one scenario"

    # Required columns
    required_cols = ["scenario", "scenario_file", "total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
    for col in required_cols:
        assert col in summary_df.columns, f"Summary must have column: {col}"

    # Core scenarios (used in test_graphite_scenarios_smoke) must be present
    core_scenarios = {"graphite_baseline", "graphite_export_restriction", "graphite_demand_surge", "graphite_policy_response"}
    actual_scenarios = set(summary_df["scenario"].values)
    missing = core_scenarios - actual_scenarios
    assert not missing, f"Suite must include core scenarios: {missing}"

    # Core metrics should be finite where present (allow NaN for optional shock metrics)
    metric_cols = ["total_shortage", "peak_shortage", "avg_price", "final_inventory_cover"]
    for col in metric_cols:
        valid = summary_df[col].notna()
        assert (summary_df.loc[valid, col].abs() < float("inf")).all(), f"Column {col} must have finite values where not NaN"

    # Shock-window column present for scenarios with shocks (if column exists)
    if "shock_window_total_shortage" in summary_df.columns:
        shock_scenarios = {"graphite_export_restriction", "graphite_demand_surge", "graphite_policy_response"}
        for scenario in shock_scenarios:
            if scenario not in actual_scenarios:
                continue
            row = summary_df[summary_df["scenario"] == scenario].iloc[0]
            assert pd.notna(row.get("shock_window_total_shortage")), (
                f"{scenario} should have non-NaN shock_window_total_shortage"
            )

    # No run errors
    if "error" in summary_df.columns:
        errors = summary_df[summary_df["error"].notna()]
        assert len(errors) == 0, f"Some scenarios had errors: {errors[['scenario', 'error']].to_dict('records')}"

