#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario


def find_scenarios(scenarios_dir: Path) -> List[Path]:
    """Find all YAML scenario files in the scenarios directory."""
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    return sorted(scenarios_dir.glob("*.yaml"))


def run_suite(scenarios_dir: Path = Path("scenarios"), output_base: Path = Path("runs")) -> pd.DataFrame:
    """
    Run all scenarios in the scenarios directory and return summary DataFrame.
    
    Args:
        scenarios_dir: Directory containing scenario YAML files
        output_base: Base directory for outputs
        
    Returns:
        DataFrame with one row per scenario and all metrics as columns
    """
    scenario_files = find_scenarios(scenarios_dir)
    
    if not scenario_files:
        raise ValueError(f"No scenario files found in {scenarios_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = output_base / f"suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    
    summary_rows = []
    
    for scenario_path in scenario_files:
        try:
            # Load and run scenario
            cfg = load_scenario(str(scenario_path))
            df, metrics = run_scenario(cfg)
            
            # Create scenario output directory
            scenario_dir = suite_dir / cfg.name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            # Write timeseries
            df.to_csv(scenario_dir / "timeseries.csv", index=False)
            
            # Write metrics
            with open(scenario_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            
            # Prepare summary row
            row = {
                "scenario": cfg.name,
                "scenario_file": scenario_path.name,
                **metrics
            }
            summary_rows.append(row)
            
        except Exception as e:
            # Log error but continue with other scenarios
            print(f"Error running {scenario_path.name}: {e}")
            row = {
                "scenario": scenario_path.stem,
                "scenario_file": scenario_path.name,
                "error": str(e)
            }
            summary_rows.append(row)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_rows)
    
    # Write summary CSV
    summary_path = suite_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Suite run complete: {len(summary_df)} scenarios")
    print(f"Summary: {summary_path}")
    print(f"Outputs: {suite_dir}")
    
    return summary_df


def main() -> int:
    """Main entrypoint for batch scenario runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run all scenarios in scenarios/ directory")
    parser.add_argument("--scenarios-dir", type=str, default="scenarios",
                       help="Directory containing scenario YAML files")
    parser.add_argument("--output-dir", type=str, default="runs",
                       help="Base output directory")
    args = parser.parse_args()
    
    scenarios_dir = Path(args.scenarios_dir)
    output_base = Path(args.output_dir)
    
    summary_df = run_suite(scenarios_dir, output_base)
    
    # Print summary
    print("\n=== Summary ===")
    print(summary_df.to_string(index=False))
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

