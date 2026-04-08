from __future__ import annotations

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario


def test_graphite_baseline_golden_metrics():
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
