#!/usr/bin/env python3
"""
Calibrate the scenario (causal dynamics) model to Comtrade data.

Fits parameters (tau_K, alpha_P, eta_D) so that simulated P*Q tracks
Comtrade trade value over the same years. Outputs a calibrated scenario YAML
so the causal model is trained on real trade data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from src.minerals.schema import load_scenario, ScenarioConfig, ParametersConfig
from src.minerals.simulate import run_scenario


def load_comtrade_series(path: Path) -> pd.Series:
    """Load Comtrade CSV and return a Series: index=year, value=trade_value_usd."""
    df = pd.read_csv(path)
    if "date" in df.columns:
        year_col = "date"
    elif "year" in df.columns:
        year_col = "year"
    else:
        raise ValueError("Comtrade CSV must have 'date' or 'year' column")
    if "value" in df.columns:
        val_col = "value"
    elif "trade_value_usd" in df.columns:
        val_col = "trade_value_usd"
    elif "value_kusd" in df.columns:
        # CEPII canonical: value in thousands USD; use as-is (calibration scales)
        val_col = "value_kusd"
    elif "P" in df.columns:
        val_col = "P"
    else:
        raise ValueError(
            "CSV must have 'value', 'trade_value_usd', 'value_kusd' (CEPII), or 'P' column"
        )
    df = df.dropna(subset=[year_col, val_col])
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    # If multiple rows per year, aggregate
    out = df.groupby(year_col)[val_col].sum()
    return out.sort_index()


def load_all_bilateral_comtrade(canonical_dir: Path) -> pd.Series:
    """Load all trade CSV files in data/canonical (comtrade_*.csv, cepii_*.csv), aggregate by year."""
    canonical_dir = Path(canonical_dir)
    files = sorted(canonical_dir.glob("comtrade_*.csv")) + sorted(canonical_dir.glob("cepii_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No comtrade_*.csv or cepii_*.csv files in {canonical_dir}"
        )
    all_series: list[pd.Series] = []
    for path in files:
        try:
            s = load_comtrade_series(path)
            all_series.append(s)
        except (ValueError, KeyError) as e:
            # Skip files that don't have expected columns
            continue
    if not all_series:
        raise ValueError(f"No valid Comtrade series could be loaded from {canonical_dir}")
    # Sum across all files by year (align, fill 0 for missing)
    combined = pd.concat(all_series, axis=1).fillna(0).sum(axis=1).sort_index()
    return combined


def run_scenario_get_pq(cfg: ScenarioConfig) -> pd.DataFrame:
    """Run scenario and return DataFrame with year, P, Q."""
    df, _ = run_scenario(cfg)
    return df[["year", "P", "Q"]].copy()


def objective(
    x: np.ndarray,
    cfg: ScenarioConfig,
    comtrade: pd.Series,
    param_names: list[str],
) -> float:
    """RMSE between scaled simulated P*Q and Comtrade (minimize this)."""
    param_dict = dict(zip(param_names, x))
    new_params = ParametersConfig(**{**cfg.parameters.model_dump(), **param_dict})
    updated = cfg.model_copy(update={"parameters": new_params})
    df = run_scenario_get_pq(updated)
    df["trade_proxy"] = df["P"] * df["Q"]
    df = df.set_index("year")
    common = comtrade.index.intersection(df.index)
    if len(common) < 2:
        return 1e10
    sim = df.loc[common, "trade_proxy"].values.astype(float)
    act = comtrade.loc[common].values.astype(float)
    if sim.max() <= 0:
        return 1e10
    scale = np.dot(sim, act) / (np.dot(sim, sim) + 1e-12)
    err = scale * sim - act
    return float(np.sqrt(np.mean(err ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate scenario parameters to Comtrade so the causal model trains on real trade data."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(PROJECT_ROOT / "scenarios" / "graphite_baseline.yaml"),
        help="Base scenario YAML",
    )
    parser.add_argument(
        "--comtrade",
        "--input",
        dest="comtrade",
        type=str,
        default="",
        help="Single trade CSV (Comtrade or CEPII canonical). Expects year + value/trade_value_usd/value_kusd. Alias: --input.",
    )
    parser.add_argument(
        "--use-all-bilateral",
        action="store_true",
        help="Use all bilateral Comtrade files in data/canonical (comtrade_*.csv), aggregated by year.",
    )
    parser.add_argument(
        "--canonical-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "canonical"),
        help="Directory to scan for comtrade_*.csv when --use-all-bilateral is set.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "scenarios" / "graphite_comtrade_calibrated.yaml"),
        help="Output calibrated scenario YAML",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="+",
        default=["tau_K", "alpha_P", "eta_D"],
        help="Parameter names to calibrate",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        default="tau_K:1,10 alpha_P:0.5,5 eta_D:-1.5,0",
        help="Bounds as 'name:lo,hi name:lo,hi'",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=80,
        help="Max optimization iterations",
    )
    args = parser.parse_args()

    canonical_dir = Path(args.canonical_dir)
    if args.use_all_bilateral:
        if not canonical_dir.exists():
            print(f"❌ Canonical dir not found: {canonical_dir}")
            return 1
        try:
            comtrade = load_all_bilateral_comtrade(canonical_dir)
            n_comtrade = len(list(canonical_dir.glob("comtrade_*.csv")))
            n_cepii = len(list(canonical_dir.glob("cepii_*.csv")))
            n_files = n_comtrade + n_cepii
            print(f"Trade data (all files: {n_comtrade} comtrade + {n_cepii} cepii in {canonical_dir}): {len(comtrade)} years, {comtrade.index.min()}–{comtrade.index.max()}")
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ {e}")
            return 1
    elif args.comtrade:
        comtrade_path = Path(args.comtrade)
        if not comtrade_path.exists():
            print(f"❌ Comtrade file not found: {comtrade_path}")
            return 1
        comtrade = load_comtrade_series(comtrade_path)
        print(f"Comtrade: {comtrade_path.name}, {len(comtrade)} years, {comtrade.index.min()}–{comtrade.index.max()}")
    else:
        # Default: single file
        default_path = PROJECT_ROOT / "data" / "canonical" / "comtrade_graphite_trade.csv"
        if not default_path.exists():
            print(f"❌ No Comtrade data. Use --comtrade <path> or add files to data/canonical/ and use --use-all-bilateral.")
            return 1
        comtrade = load_comtrade_series(default_path)
        print(f"Comtrade: {default_path.name}, {len(comtrade)} years, {comtrade.index.min()}–{comtrade.index.max()}")

    cfg = load_scenario(args.scenario)
    # Restrict scenario to Comtrade year range for calibration
    start = int(comtrade.index.min())
    end = int(comtrade.index.max())
    cfg = cfg.model_copy(
        update={
            "time": cfg.time.model_copy(update={"start_year": start, "end_year": end})
        }
    )
    print(f"Scenario: {cfg.name}, years {cfg.time.start_year}–{cfg.time.end_year}")

    param_names_raw = [p.strip() for p in args.params]
    bounds_str = args.bounds.strip().split()
    bounds_dict = {}
    for b in bounds_str:
        name, rest = b.split(":", 1)
        name = name.strip()
        lo, hi = rest.split(",")
        bounds_dict[name] = (float(lo), float(hi))
    for p in param_names_raw:
        if p not in bounds_dict:
            bounds_dict[p] = (0.1, 20.0)
    param_names = []
    x0 = []
    bounds_tuples = []
    for p in param_names_raw:
        current = getattr(cfg.parameters, p, None)
        if current is None:
            print(f"⚠️ Parameter {p} not in scenario, skipping")
            continue
        param_names.append(p)
        x0.append(float(current))
        bounds_tuples.append(bounds_dict.get(p, (0.1, 20.0)))
    if not param_names:
        print("❌ No parameters to calibrate")
        return 1

    from scipy.optimize import minimize

    def obj(x: np.ndarray) -> float:
        return objective(x, cfg, comtrade, param_names)

    result = minimize(
        obj,
        np.array(x0),
        method="L-BFGS-B",
        bounds=bounds_tuples,
        options={"maxiter": args.maxiter},
    )
    if not result.success:
        print(f"⚠️ Optimizer: {result.message}")
    print(f"RMSE (scaled P*Q vs Comtrade): {result.fun:.2f}")

    # Write calibrated scenario
    updates = dict(zip(param_names, result.x.tolist()))
    new_params = {**cfg.parameters.model_dump(), **updates}
    out_data = cfg.model_dump()
    out_data["parameters"] = new_params
    out_data["name"] = Path(args.out).stem
    out_data["description"] = "Calibrated to Comtrade trade data (P*Q vs trade_value_usd)."
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(out_data, f, default_flow_style=False, sort_keys=False)
    print(f"✅ Calibrated scenario written to {out_path}")
    print("   Calibrated parameters:", updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
