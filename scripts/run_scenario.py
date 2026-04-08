#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minerals scenario (graphite baseline).")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML.")
    parser.add_argument("--out-dir", dest="out_dir", type=str, default=None,
                       help="Output directory (if not provided, uses runs/<scenario>/<timestamp>/)")
    args = parser.parse_args()

    cfg = load_scenario(args.scenario)
    df, metrics = run_scenario(cfg)

    # Determine output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg.outputs.out_dir) / cfg.name / ts
        out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.outputs.save_csv:
        df.to_csv(out_dir / "timeseries.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    # Print summary
    print(f"Scenario: {cfg.name} ({cfg.commodity})")
    print(f"Years: {cfg.time.start_year}–{cfg.time.end_year} dt={cfg.time.dt}")
    for k in cfg.outputs.metrics:
        if k in metrics:
            print(f"{k}: {metrics[k]:.6f}")
    # Also print shock-window metrics if present (even if not in outputs.metrics list)
    shock_metrics = ["shock_window_total_shortage", "post_shock_total_shortage", "shock_year_shortage"]
    for k in shock_metrics:
        if k in metrics and k not in cfg.outputs.metrics:
            print(f"{k}: {metrics[k]:.6f}")
    print(f"Outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
