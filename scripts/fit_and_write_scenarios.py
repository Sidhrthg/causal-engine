#!/usr/bin/env python3
"""
Fit empirical parameters from CEPII data and write calibrated scenario YAMLs.

Closes the gap between fitted parameters (from real BACI data) and the hardcoded
values that currently live in every scenario YAML (eta_D: -0.25, alpha_P: 0.80,
tau_K: 3.0).

What this script does
---------------------
1. For each commodity with a canonical CEPII CSV, runs
   ``fit_commodity_parameters()`` to estimate η_D, α_P, τ_K via 2SLS / OLS / AR(1).
2. Scans every scenario YAML in ``scenarios/`` for that commodity.
3. Writes a calibrated copy to ``scenarios/calibrated/<name>_calibrated.yaml``
   with the three fitted parameters substituted in.
4. Prints a diff of what changed and a fit diagnostics summary.

Usage
-----
    # Fit all available commodities
    python scripts/fit_and_write_scenarios.py

    # One commodity only
    python scripts/fit_and_write_scenarios.py --commodity graphite

    # Show what would change without writing files
    python scripts/fit_and_write_scenarios.py --dry-run

    # Skip scenarios that already have calibrated copies
    python scripts/fit_and_write_scenarios.py --skip-existing

Why this matters
----------------
Counterfactual queries P(Y_x | X', Y') are only causally meaningful when the
structural parameters in the scenario reflect the empirical data-generating
process.  Using hardcoded literature guesses for η_D and α_P introduces
systematic bias in every L2 and L3 answer — the fitted values from CEPII data
are the correct priors.

Identification
--------------
  η_D  (demand price elasticity):
        2SLS with 2-year lagged domestic supply changes as instrument.
        Breaks contemporaneous endogeneity of price ↔ demand.
        Sign-enforced negative; falls back to OLS if F < 5.

  α_P  (price adjustment speed):
        OLS on Hodrick-Prescott filtered price gap (stationary proxy).
        Falls back to literature prior 0.30 if estimated value ≤ 0.

  τ_K  (capacity adjustment time):
        AR(1) on log(dom_supply) → implied mean-reversion rate → τ_K.
        Delta-method standard error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.minerals.parameter_fitting import fit_commodity_parameters, FittedParameters

SCENARIOS_DIR = ROOT / "scenarios"
CALIBRATED_DIR = SCENARIOS_DIR / "calibrated"
DATA_DIR = ROOT / "data" / "canonical"

# Commodities with canonical CEPII CSVs available
AVAILABLE = {
    "graphite":  {"dominant_exporter": "China"},
    "lithium":   {"dominant_exporter": "Australia"},
    "soybeans":  {"dominant_exporter": "USA"},
}


# ─── Fit ─────────────────────────────────────────────────────────────────────

def fit_for_commodity(commodity: str, verbose: bool = True) -> FittedParameters:
    csv_path = DATA_DIR / f"cepii_{commodity}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing canonical CSV: {csv_path}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Fitting parameters: {commodity.upper()}")
        print(f"  Data: {csv_path} ({csv_path.stat().st_size // 1024} KB)")
        print(f"{'='*60}")

    params = fit_commodity_parameters(
        commodity=commodity,
        data_path=str(csv_path),
        dominant_exporter=AVAILABLE[commodity]["dominant_exporter"],
    )

    if verbose:
        print(params.summary())

    return params


# ─── YAML calibration ────────────────────────────────────────────────────────

def calibrate_scenario(
    scenario_path: Path,
    fitted: FittedParameters,
    dry_run: bool = False,
    out_dir: Path = CALIBRATED_DIR,
) -> dict:
    """
    Read a scenario YAML, substitute fitted parameters, write calibrated copy.

    Returns a dict with 'changed' (list of (key, old, new)) and 'out_path'.
    """
    with open(scenario_path) as f:
        data = yaml.safe_load(f)

    if data.get("commodity") != fitted.commodity:
        return {"changed": [], "out_path": None, "skipped": True,
                "reason": f"commodity mismatch ({data.get('commodity')} ≠ {fitted.commodity})"}

    params = data.setdefault("parameters", {})
    changed = []

    PARAM_MAP = {
        "eta_D":   fitted.eta_D,
        "alpha_P": fitted.alpha_P,
        "tau_K":   fitted.tau_K,
    }

    for key, new_val in PARAM_MAP.items():
        if new_val is None:
            continue
        old_val = params.get(key)
        if old_val != new_val:
            changed.append((key, old_val, round(new_val, 4)))
            params[key] = round(new_val, 4)

    # Add calibration provenance comment
    data["_calibration"] = {
        "fitted_from": str(DATA_DIR / f"cepii_{fitted.commodity}.csv"),
        "n_obs": fitted.n_obs,
        "eta_D_ci": [round(x, 4) for x in fitted.eta_D_ci] if fitted.eta_D_ci else None,
        "alpha_P_ci": [round(x, 4) for x in fitted.alpha_P_ci] if fitted.alpha_P_ci else None,
        "tau_K_ci": [round(x, 4) for x in fitted.tau_K_ci] if fitted.tau_K_ci else None,
        "eta_D_method": fitted.eta_D_method,
        "alpha_P_method": fitted.alpha_P_method,
    }

    out_path = out_dir / f"{scenario_path.stem}_calibrated.yaml"

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            # Preserve the original comment header if it exists
            original_text = scenario_path.read_text()
            header_lines = []
            for line in original_text.splitlines():
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break
            if header_lines:
                f.write("\n".join(header_lines) + "\n")
                f.write("# [CALIBRATED] Parameters fitted from CEPII data by fit_and_write_scenarios.py\n\n")

            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return {"changed": changed, "out_path": out_path, "skipped": False}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--commodity", choices=list(AVAILABLE), help="Fit one commodity only")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing files")
    parser.add_argument("--skip-existing", action="store_true", help="Skip scenarios with existing calibrated copies")
    parser.add_argument("--quiet", action="store_true", help="Suppress fitting diagnostics")
    args = parser.parse_args()

    commodities = [args.commodity] if args.commodity else list(AVAILABLE)

    total_written = 0
    total_skipped = 0
    total_unchanged = 0

    for commodity in commodities:
        try:
            fitted = fit_for_commodity(commodity, verbose=not args.quiet)
        except FileNotFoundError as e:
            print(f"\nWARNING: {e} — skipping {commodity}")
            continue
        except Exception as e:
            print(f"\nERROR fitting {commodity}: {e}")
            continue

        # Find all scenario YAMLs for this commodity
        scenario_files = sorted(SCENARIOS_DIR.glob("*.yaml"))
        commodity_scenarios = []
        for f in scenario_files:
            try:
                with open(f) as fh:
                    d = yaml.safe_load(fh)
                if d.get("commodity") == commodity:
                    commodity_scenarios.append(f)
            except Exception:
                continue

        print(f"\nFound {len(commodity_scenarios)} scenarios for {commodity}")

        for scenario_path in commodity_scenarios:
            out_path = CALIBRATED_DIR / f"{scenario_path.stem}_calibrated.yaml"
            if args.skip_existing and out_path.exists():
                print(f"  SKIP  {scenario_path.name} (calibrated copy exists)")
                total_skipped += 1
                continue

            result = calibrate_scenario(scenario_path, fitted, dry_run=args.dry_run)

            if result.get("skipped"):
                print(f"  SKIP  {scenario_path.name}: {result['reason']}")
                total_skipped += 1
                continue

            if not result["changed"]:
                print(f"  ---   {scenario_path.name}: no parameter changes")
                total_unchanged += 1
                continue

            action = "DRY-RUN" if args.dry_run else "WROTE"
            print(f"  {action}  {scenario_path.name} → {result['out_path'].name if result['out_path'] else '(dry)'}")
            for key, old, new in result["changed"]:
                arrow = "↑" if new > (old or 0) else "↓"
                print(f"         {key:12s}  {str(old):>8s}  →  {str(new):>8s}  {arrow}")
            total_written += 1

    print(f"\n{'─'*50}")
    print(f"  Written:   {total_written}")
    print(f"  Unchanged: {total_unchanged}")
    print(f"  Skipped:   {total_skipped}")
    if args.dry_run:
        print(f"  (dry-run — no files written)")
    else:
        print(f"  Calibrated scenarios → {CALIBRATED_DIR}/")
    print()


if __name__ == "__main__":
    main()
