"""Tests for memo generation using the deterministic MockLLM."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_memo import generate_memo


def test_generate_memo_mock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()

    report = {
        "scenario_path": "scenarios/graphite_baseline.yaml",
        "fitted_params": {"beta": 1.0},
        "metrics": {"rmse": 0.1, "mae": 0.05},
        "artifacts": ["timeseries.csv", "metrics.json"],
        "narrative": "Baseline scenario with stable demand.",
    }
    (run_dir / "report.json").write_text(json.dumps(report))

    memo_path = generate_memo(run_dir, provider="mock")
    memo_text = memo_path.read_text()

    assert "graphite_baseline.yaml" in memo_text
    assert "rmse" in memo_text
    assert "avg_price" in memo_text
    assert "shock_year_shortage" in memo_text
    assert "Caveats" in memo_text
    assert "No narrative provided." not in memo_text

    # Deterministic output when run twice
    second = generate_memo(run_dir, provider="mock")
    assert memo_text == second.read_text()

