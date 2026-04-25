#!/usr/bin/env python3
"""
Sensitivity grid for ODE fixed parameters.

Perturbs cover_star, u0, and lambda_cover across a 3x4x3 grid (36 combinations)
while holding calibrated parameters (alpha_P, eta_D, tau_K, g) fixed at their
episode-specific values. Re-runs all 8 in-sample episodes and the 7 OOS transfers
at each grid point.

Reports:
  - Mean in-sample DA across the grid
  - Mean OOS DA across the grid
  - Per-episode DA range (max - min) across grid points → sensitivity ranking
  - Which OOS transfer degrades most under perturbation

Output:
  data/sensitivity_grid.json   — full results
  (stdout)                     — summary table
"""

from __future__ import annotations

import json
import sys
import copy
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.predictability import (
    _directional_accuracy, _spearman_rho,
    _cepii_series, _wb_cobalt_series, _wb_nickel_series,
    _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS,
    _LITHIUM_2022_PARAMS, _SOYBEANS_2022_PARAMS,
    _COBALT_2016_PARAMS, _COBALT_2022_PARAMS,
    _NICKEL_2006_PARAMS, _NICKEL_2022_PARAMS,
)


# ── Grid definition ────────────────────────────────────────────────────────────

GRID = {
    "cover_star":    [0.10, 0.15, 0.20, 0.25],
    "u0":            [0.85, 0.90, 0.92, 0.95],
    "lambda_cover":  [0.40, 0.60, 0.80],
}

N_GRID = (
    len(GRID["cover_star"]) *
    len(GRID["u0"]) *
    len(GRID["lambda_cover"])
)  # 48 combinations


# ── Episode builders (return cfg given fixed-param overrides) ─────────────────

def _params(p: dict, u0: float, cover_star: float, lambda_cover: float) -> ParametersConfig:
    return ParametersConfig(
        eps=1e-9,
        u0=u0, beta_u=0.10, u_min=0.70, u_max=1.00,
        tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
        demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
        alpha_P=p["alpha_P"],
        cover_star=cover_star, lambda_cover=lambda_cover, sigma_P=0.0,
    )


def _run_episode(cfg: ScenarioConfig, years: list, base_year: int,
                  data_series, price_col: str = "implied_price"):
    df, _ = run_scenario(cfg)
    m = df.set_index("year")
    model_P = m.loc[years, "P"]
    model_idx = model_P / model_P.loc[base_year]
    data_P = data_series.loc[years, price_col]
    data_idx = data_P / data_P.loc[base_year]
    da = _directional_accuracy(model_idx, data_idx)
    return da, model_idx, data_idx


def build_and_run_all(u0: float, cover_star: float, lambda_cover: float) -> Dict[str, float]:
    """Run all 8 in-sample + 7 OOS episodes with given fixed params. Return {name: DA}."""
    results = {}

    cg = _cepii_series("data/canonical/cepii_graphite.csv", "China")
    cl = _cepii_series("data/canonical/cepii_lithium.csv", "Australia")
    cs_raw = _cepii_series.__func__ if hasattr(_cepii_series, '__func__') else None

    # Load soybeans — global clearing price
    import pandas as pd
    soy_df = pd.read_csv("data/canonical/cepii_soybeans.csv")
    cs = (
        soy_df.groupby("year")
        .agg(value_kusd=("value_kusd", "sum"), qty_tonnes=("quantity_tonnes", "sum"))
        .assign(implied_price=lambda d: d["value_kusd"] / d["qty_tonnes"])
    )

    cobalt_wb  = _wb_cobalt_series()
    nickel_wb  = _wb_nickel_series()

    # ── 1. graphite_2008 ──────────────────────────────────────────────────────
    p = _GRAPHITE_2008_PARAMS
    cfg = ScenarioConfig(
        name="g08", commodity="graphite", seed=123,
        time=TimeConfig(dt=1.0, start_year=2004, end_year=2011),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2008, end_year=2008, magnitude=0.46),
            ShockConfig(type="macro_demand_shock",  start_year=2009, end_year=2009, magnitude=-0.40, demand_destruction=-0.40),
            ShockConfig(type="policy_shock",        start_year=2010, end_year=2011, magnitude=0.35, quota_reduction=0.35),
            ShockConfig(type="capex_shock",         start_year=2010, end_year=2011, magnitude=0.50),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    da, _, _ = _run_episode(cfg, [2006,2007,2008,2009,2010,2011], 2006, cg)
    results["graphite_2008"] = da

    # ── 2. graphite_2022 ──────────────────────────────────────────────────────
    p = _GRAPHITE_2022_PARAMS
    cfg = ScenarioConfig(
        name="g22", commodity="graphite", seed=42,
        time=TimeConfig(dt=1.0, start_year=2019, end_year=2024),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.22),
            ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2024, magnitude=-0.15),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    da, _, _ = _run_episode(cfg, [2021,2022,2023,2024], 2021, cg)
    results["graphite_2022"] = da

    # ── 3. lithium_2022 ───────────────────────────────────────────────────────
    p = _LITHIUM_2022_PARAMS
    cfg = ScenarioConfig(
        name="li22", commodity="lithium", seed=42,
        time=TimeConfig(dt=1.0, start_year=2019, end_year=2024),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.40),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2024, magnitude=-0.30),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    da, _, _ = _run_episode(cfg, [2021,2022,2023,2024], 2021, cl)
    results["lithium_2022"] = da

    # ── 4-8. soybeans ─────────────────────────────────────────────────────────
    p = _SOYBEANS_2022_PARAMS

    soy_episodes = [
        ("soybeans_2011", [2009,2010,2011], 2009, [
            ShockConfig(type="demand_surge", start_year=2010, end_year=2011, magnitude=0.18),
        ]),
        ("soybeans_2015", [2014,2015,2016,2017], 2014, [
            ShockConfig(type="demand_surge", start_year=2015, end_year=2016, magnitude=-0.12),
        ]),
        ("soybeans_2018", [2016,2017,2018,2020,2021], 2016, [
            ShockConfig(type="export_restriction", start_year=2018, end_year=2020, magnitude=0.10),
        ]),
        ("soybeans_2020", [2018,2020,2021], 2018, [
            ShockConfig(type="demand_surge", start_year=2020, end_year=2021, magnitude=0.12),
        ]),
        ("soybeans_2022", [2020,2021,2022,2023,2024], 2020, [
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.20),
            ShockConfig(type="export_restriction", start_year=2022, end_year=2023, magnitude=0.08),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2024, magnitude=-0.10),
        ]),
    ]

    for name, years, base, shocks in soy_episodes:
        cfg = ScenarioConfig(
            name=name, commodity="soybeans", seed=42,
            time=TimeConfig(dt=1.0, start_year=years[0]-2, end_year=years[-1]),
            baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
            parameters=_params(p, u0, cover_star, lambda_cover),
            policy=PolicyConfig(),
            shocks=shocks,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )
        try:
            da, _, _ = _run_episode(cfg, years, base, cs)
            results[name] = da
        except Exception:
            results[name] = float("nan")

    # ── OOS: graphite cross-epoch ─────────────────────────────────────────────
    # graphite_2022_oos: 2022 episode with 2008 params
    p = _GRAPHITE_2008_PARAMS
    cfg = ScenarioConfig(
        name="g22_oos", commodity="graphite", seed=42,
        time=TimeConfig(dt=1.0, start_year=2019, end_year=2024),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.22),
            ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2024, magnitude=-0.15),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    try:
        da, _, _ = _run_episode(cfg, [2021,2022,2023,2024], 2021, cg)
        results["oos_graphite_2022"] = da
    except Exception:
        results["oos_graphite_2022"] = float("nan")

    # graphite_2008_oos: 2008 episode with 2022 params
    p = _GRAPHITE_2022_PARAMS
    cfg = ScenarioConfig(
        name="g08_oos", commodity="graphite", seed=123,
        time=TimeConfig(dt=1.0, start_year=2004, end_year=2011),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2008, end_year=2008, magnitude=0.46),
            ShockConfig(type="macro_demand_shock",  start_year=2009, end_year=2009, magnitude=-0.40, demand_destruction=-0.40),
            ShockConfig(type="policy_shock",        start_year=2010, end_year=2011, magnitude=0.35, quota_reduction=0.35),
            ShockConfig(type="capex_shock",         start_year=2010, end_year=2011, magnitude=0.50),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    try:
        da, _, _ = _run_episode(cfg, [2006,2007,2008,2009,2010,2011], 2006, cg)
        results["oos_graphite_2008"] = da
    except Exception:
        results["oos_graphite_2008"] = float("nan")

    # soybeans_2022_oos: 2022 episode with soybeans_2011 params (alpha_P=0.80 fixed)
    p = dict(alpha_P=0.80, eta_D=-0.791, tau_K=8.445, g=1.0891)
    cfg = ScenarioConfig(
        name="s22_oos", commodity="soybeans", seed=42,
        time=TimeConfig(dt=1.0, start_year=2018, end_year=2024),
        baseline=BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0),
        parameters=_params(p, u0, cover_star, lambda_cover),
        policy=PolicyConfig(),
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.20),
            ShockConfig(type="export_restriction", start_year=2022, end_year=2023, magnitude=0.08),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2024, magnitude=-0.10),
        ],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    try:
        da, _, _ = _run_episode(cfg, [2020,2021,2022,2023,2024], 2020, cs)
        results["oos_soybeans_2022"] = da
    except Exception:
        results["oos_soybeans_2022"] = float("nan")

    return results


# ── Run the grid ──────────────────────────────────────────────────────────────

def run_grid() -> List[dict]:
    rows = []
    total = len(GRID["cover_star"]) * len(GRID["u0"]) * len(GRID["lambda_cover"])
    done = 0
    for cs_val, u0_val, lc_val in product(
        GRID["cover_star"], GRID["u0"], GRID["lambda_cover"]
    ):
        done += 1
        print(f"  [{done:2d}/{total}] cover_star={cs_val} u0={u0_val} lambda_cover={lc_val}", flush=True)
        ep_results = build_and_run_all(u0_val, cs_val, lc_val)
        row = {"cover_star": cs_val, "u0": u0_val, "lambda_cover": lc_val}
        row.update(ep_results)
        rows.append(row)
    return rows


def summarise(rows: List[dict]) -> None:
    in_sample_keys = [
        "graphite_2008", "graphite_2022", "lithium_2022",
        "soybeans_2011", "soybeans_2015", "soybeans_2018",
        "soybeans_2020", "soybeans_2022",
    ]
    oos_keys = [
        "oos_graphite_2022", "oos_graphite_2008", "oos_soybeans_2022",
    ]

    # Mean DA at each grid point
    is_means = []
    oos_means = []
    for row in rows:
        is_vals = [row[k] for k in in_sample_keys if not np.isnan(row.get(k, float("nan")))]
        oos_vals = [row[k] for k in oos_keys if not np.isnan(row.get(k, float("nan")))]
        is_means.append(np.mean(is_vals) if is_vals else float("nan"))
        oos_means.append(np.mean(oos_vals) if oos_vals else float("nan"))

    print("\n" + "="*70)
    print("SENSITIVITY GRID SUMMARY")
    print("="*70)
    print(f"Grid points: {len(rows)}")
    print(f"In-sample DA:  mean={np.nanmean(is_means):.3f}  "
          f"min={np.nanmin(is_means):.3f}  max={np.nanmax(is_means):.3f}  "
          f"range={np.nanmax(is_means)-np.nanmin(is_means):.3f}")
    print(f"OOS DA:        mean={np.nanmean(oos_means):.3f}  "
          f"min={np.nanmin(oos_means):.3f}  max={np.nanmax(oos_means):.3f}  "
          f"range={np.nanmax(oos_means)-np.nanmin(oos_means):.3f}")

    # Per-episode sensitivity (range across grid)
    print("\nPer-episode DA range across grid (higher = more sensitive):")
    print(f"  {'Episode':<30}  {'Mean':>6}  {'Min':>6}  {'Max':>6}  {'Range':>7}")
    all_keys = in_sample_keys + oos_keys
    ep_ranges = []
    for k in all_keys:
        vals = [r.get(k, float("nan")) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            ep_ranges.append((k, np.mean(vals), np.min(vals), np.max(vals), np.max(vals)-np.min(vals)))

    for k, mn, mi, mx, rng in sorted(ep_ranges, key=lambda x: -x[4]):
        print(f"  {k:<30}  {mn:>6.3f}  {mi:>6.3f}  {mx:>6.3f}  {rng:>7.3f}")

    # Worst grid point
    worst_is = rows[int(np.nanargmin(is_means))]
    worst_oos = rows[int(np.nanargmin(oos_means))]
    best_is  = rows[int(np.nanargmax(is_means))]
    print(f"\nWorst in-sample grid point:  cover_star={worst_is['cover_star']} u0={worst_is['u0']} lambda_cover={worst_is['lambda_cover']}  DA={np.nanmin(is_means):.3f}")
    print(f"Best  in-sample grid point:  cover_star={best_is['cover_star']}  u0={best_is['u0']} lambda_cover={best_is['lambda_cover']}  DA={np.nanmax(is_means):.3f}")
    print(f"Worst OOS grid point:        cover_star={worst_oos['cover_star']} u0={worst_oos['u0']} lambda_cover={worst_oos['lambda_cover']}  DA={np.nanmin(oos_means):.3f}")

    # Parameter-level marginal effect
    print("\nMarginal effect of each parameter on mean OOS DA:")
    for param, vals in GRID.items():
        by_val = {}
        for v in vals:
            pts = [oos_means[i] for i, r in enumerate(rows) if r[param] == v and not np.isnan(oos_means[i])]
            by_val[v] = np.mean(pts) if pts else float("nan")
        rng = max(by_val.values()) - min(by_val.values())
        best_v = max(by_val, key=lambda x: by_val[x])
        worst_v = min(by_val, key=lambda x: by_val[x])
        print(f"  {param:<15}: range={rng:.3f}  best={best_v} ({by_val[best_v]:.3f})  worst={worst_v} ({by_val[worst_v]:.3f})")


if __name__ == "__main__":
    print(f"Running {N_GRID}-point sensitivity grid ({len(GRID['cover_star'])} × {len(GRID['u0'])} × {len(GRID['lambda_cover'])})...")
    rows = run_grid()

    out = ROOT / "data" / "sensitivity_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nFull results saved to {out}")

    summarise(rows)
