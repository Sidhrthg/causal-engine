#!/usr/bin/env python3
"""
Calibrate structural parameters for uranium 2022 Russia sanctions episode.

Russia/Ukraine war created import risk for ~20% of US SWU supply.
Kazatomprom ~10% production cut tightened physical market.
Sprott Physical Uranium Trust buying added speculative demand.
US Prohibiting Russian Uranium Imports Act (May 2024) formalised the ban.
EIA spot price: $28.70 (2020) → $71.92 (2024) — monotone 2.5× rise.

Calibration: differential_evolution maximising DA + Spearman rho - log_rmse penalty
Episode window: 2020-2024 (4 directional steps, all UP)
Primary focus: improve magnitude ratio (initial prior gives MagR=1.53, target ~1.0).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import differential_evolution

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.predictability import _eia_uranium_series, _directional_accuracy, _spearman_rho

# Documented historical shocks — fixed, not calibrated
SHOCKS = [
    ShockConfig(type="export_restriction", start_year=2022, end_year=2022, magnitude=0.12),
    ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.08),
    ShockConfig(type="export_restriction", start_year=2023, end_year=2023, magnitude=0.20),
    ShockConfig(type="export_restriction", start_year=2024, end_year=2024, magnitude=0.15),
    ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=0.10),
]

# Pearl L2: US utilities diversify to Cameco/Kazatomprom/Paladin
EXTRA = dict(substitution_elasticity=0.5, substitution_cap=0.4)

TIME_CFG = TimeConfig(dt=1.0, start_year=2018, end_year=2025)
BASELINE = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)

YEARS     = [2020, 2021, 2022, 2023, 2024]
BASE_YEAR = 2020


def _run(alpha_P, eta_D, tau_K, g, eia):
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        eta_K=0.40, retire_rate=0.0,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        tau_K=tau_K, eta_D=eta_D,
        demand_growth=DemandGrowthConfig(type="constant", g=g),
        alpha_P=alpha_P,
        **EXTRA,
    )
    cfg = ScenarioConfig(
        name="u22_cal", commodity="graphite", seed=42,
        time=TIME_CFG, baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(), shocks=SHOCKS,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    try:
        df, _ = run_scenario(cfg)
        m = df.set_index("year")
        model_idx = m.loc[YEARS, "P"] / m.loc[BASE_YEAR, "P"]
        data_idx  = eia.loc[YEARS, "implied_price"] / eia.loc[BASE_YEAR, "implied_price"]
        da  = _directional_accuracy(model_idx, data_idx)
        rho = _spearman_rho(model_idx, data_idx)
        log_rmse = float(np.sqrt(np.mean(
            (np.log(np.maximum(model_idx.values, 1e-9)) -
             np.log(np.maximum(data_idx.values, 1e-9)))**2
        )))
        mag_penalty = min(log_rmse / 2.0, 1.0)
        return da, rho, mag_penalty
    except Exception:
        return 0.0, 0.0, 1.0


_EIA_CACHE = None

def objective(x):
    global _EIA_CACHE
    if _EIA_CACHE is None:
        _EIA_CACHE = _eia_uranium_series()
    alpha_P, eta_D, tau_K, g = x
    da, rho, mag_penalty = _run(alpha_P, eta_D, tau_K, g, _EIA_CACHE)
    return float(-(da + rho - mag_penalty))


def main():
    eia = _eia_uranium_series()

    print("=" * 62)
    print("Calibrating uranium_2022 episode (Russia sanctions era)")
    print(f"Target years: {YEARS}, base={BASE_YEAR}")
    print("EIA spot price index:")
    base_p = eia.loc[BASE_YEAR, "implied_price"]
    prev = None
    for y in YEARS:
        p = eia.loc[y, "implied_price"]
        direction = ("UP" if p > prev else "DN") if prev else "—"
        print(f"  {y}: {p/base_p:.3f}  ({direction})")
        prev = p
    print()

    bounds = [
        (0.10, 4.00),   # alpha_P
        (-0.50, -0.001),# eta_D  (nuclear fuel demand inelastic)
        (2.00, 15.0),   # tau_K
        (0.95,  1.10),  # g
    ]

    print("Running differential evolution (this may take ~60s)...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=500, popsize=15, tol=1e-5,
        mutation=(0.5, 1.5), recombination=0.7,
        workers=1,
    )

    alpha_P, eta_D, tau_K, g = result.x
    da, rho, mp = _run(alpha_P, eta_D, tau_K, g, eia)

    print()
    print("=" * 62)
    print("CALIBRATION RESULT")
    print("=" * 62)
    print(f"  alpha_P = {alpha_P:.4f}")
    print(f"  eta_D   = {eta_D:.4f}")
    print(f"  tau_K   = {tau_K:.4f}")
    print(f"  g       = {g:.4f}")
    print(f"  DA      = {da:.3f}")
    print(f"  rho     = {rho:.3f}")
    print(f"  mag_pen = {mp:.3f}")
    print()
    print(f"_URANIUM_2022_PARAMS = dict(alpha_P={alpha_P:.3f}, eta_D={eta_D:.3f}, "
          f"tau_K={tau_K:.3f}, g={g:.4f})")

    # Year-by-year table
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        eta_K=0.40, retire_rate=0.0,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        tau_K=tau_K, eta_D=eta_D,
        demand_growth=DemandGrowthConfig(type="constant", g=g),
        alpha_P=alpha_P, **EXTRA,
    )
    cfg = ScenarioConfig(
        name="u22_final", commodity="graphite", seed=42,
        time=TIME_CFG, baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(), shocks=SHOCKS,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")
    base_p_model = m.loc[BASE_YEAR, "P"]
    print()
    print(f"{'Year':>5}  {'Model':>8}  {'Data':>8}  {'M-dir':>6}  {'D-dir':>6}  {'Match':>6}")
    prev_m = prev_d = None
    for y in YEARS:
        mi = m.loc[y, "P"] / base_p_model
        di = eia.loc[y, "implied_price"] / base_p
        m_dir = ("UP" if mi > prev_m else "DN") if prev_m else "—"
        d_dir = ("UP" if di > prev_d else "DN") if prev_d else "—"
        match = "✓" if m_dir == d_dir else "✗" if m_dir != "—" else "—"
        print(f"{y:>5}  {mi:>8.3f}  {di:>8.3f}  {m_dir:>6}  {d_dir:>6}  {match:>6}")
        prev_m, prev_d = mi, di


if __name__ == "__main__":
    main()
