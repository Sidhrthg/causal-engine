#!/usr/bin/env python3
"""
Calibrate structural parameters for rare earths 2010-2012 export quota crisis.

China imposed export quotas on rare earth compounds (HS 2846) in 2010,
cutting export volume ~40%. Prices spiked +351% in 2011 (baseline $8.9k/t
→ peak $75.3k/t). WTO dispute filed 2012 by US/EU/Japan; China eliminated
quotas after ruling in 2015. Prices normalized by 2013-2014.

Dominant exporter: China (~60% pre-quota, dropping during restriction).
Validation: CEPII BACI HS07 unit values (independent of LME).

Episode window: 2008-2014 (5 directional steps)
Calibration: differential_evolution maximising DA + Spearman rho + magnitude fit
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
from src.minerals.predictability import _cepii_series, _directional_accuracy, _spearman_rho

CEPII_PATH   = "data/canonical/cepii_rare_earths.csv"
DOM_EXPORTER = "China"

# Documented shocks:
# 2010: China tightens export quotas ~40% reduction in licensed volume
# 2011: Quotas further tightened; WTO dispute filed; prices peak
# 2012: WTO process ongoing; China eases slightly; circumvention increases
# 2013-2014: Prices fall as WTO ruling nears and non-Chinese supply grows
SHOCKS = [
    ShockConfig(type="export_restriction", start_year=2010, end_year=2010, magnitude=0.25),
    ShockConfig(type="export_restriction", start_year=2011, end_year=2011, magnitude=0.40),
    ShockConfig(type="export_restriction", start_year=2012, end_year=2013, magnitude=0.20),
    ShockConfig(type="demand_surge",       start_year=2013, end_year=2013, magnitude=-0.15),
    ShockConfig(type="demand_surge",       start_year=2014, end_year=2014, magnitude=-0.20),
]

TIME_CFG = TimeConfig(dt=1.0, start_year=2005, end_year=2016)
BASELINE = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)

YEARS     = [2008, 2009, 2010, 2011, 2012, 2013, 2014]
BASE_YEAR = 2008

# Substitution supply: Japan/EU developed alternatives post-2011;
# fringe producers (Mountain Pass, Lynas) entered when prices surged
EXTRA = dict(substitution_elasticity=0.5, substitution_cap=0.4)


def _run(alpha_P, eta_D, tau_K, g, cepii):
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
        name="re10_cal", commodity="graphite", seed=42,
        time=TIME_CFG, baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(), shocks=SHOCKS,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    try:
        df, _ = run_scenario(cfg)
        m = df.set_index("year")
        model_idx = m.loc[YEARS, "P"] / m.loc[BASE_YEAR, "P"]
        data_idx  = cepii.loc[YEARS, "implied_price"] / cepii.loc[BASE_YEAR, "implied_price"]
        da  = _directional_accuracy(model_idx, data_idx)
        rho = _spearman_rho(model_idx, data_idx)
        log_rmse = float(np.sqrt(np.mean(
            (np.log(model_idx + 1e-9) - np.log(data_idx + 1e-9))**2
        )))
        mag_penalty = min(log_rmse / 2.0, 1.0)
        return da, rho, mag_penalty
    except Exception:
        return 0.0, 0.0, 1.0


_CEPII_CACHE = None

def objective(x):
    global _CEPII_CACHE
    if _CEPII_CACHE is None:
        _CEPII_CACHE = _cepii_series(CEPII_PATH, DOM_EXPORTER)
    alpha_P, eta_D, tau_K, g = x
    da, rho, mag_penalty = _run(alpha_P, eta_D, tau_K, g, _CEPII_CACHE)
    return float(-(da + rho - mag_penalty))


def main():
    cepii = _cepii_series(CEPII_PATH, DOM_EXPORTER)

    print("=" * 62)
    print("Calibrating rare_earths_2010 episode")
    print(f"Target years: {YEARS}, base={BASE_YEAR}")
    print("CEPII China unit value index:")
    base_p = cepii.loc[BASE_YEAR, "implied_price"]
    prev = None
    for y in YEARS:
        p = cepii.loc[y, "implied_price"]
        direction = ("UP" if p > prev else "DN") if prev else "—"
        print(f"  {y}: {p/base_p:.3f}  ({direction})")
        prev = p
    print()

    bounds = [
        (0.10, 4.00),   # alpha_P
        (-1.50, -0.01), # eta_D
        (0.50, 20.0),   # tau_K
        (0.92,  1.20),  # g
    ]

    print("Running differential evolution...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=400, popsize=15, tol=1e-4,
        mutation=(0.5, 1.5), recombination=0.7,
    )

    alpha_P, eta_D, tau_K, g = result.x
    da, rho, mp = _run(alpha_P, eta_D, tau_K, g, cepii)

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
    print(f"_RARE_EARTHS_2010_PARAMS = dict(alpha_P={alpha_P:.3f}, eta_D={eta_D:.3f}, "
          f"tau_K={tau_K:.3f}, g={g:.4f})")

    # Year-by-year
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        eta_K=0.40, retire_rate=0.0,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        tau_K=tau_K, eta_D=eta_D,
        demand_growth=DemandGrowthConfig(type="constant", g=g),
        alpha_P=alpha_P, **EXTRA,
    )
    cfg = ScenarioConfig(
        name="re10_final", commodity="graphite", seed=42,
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
        di = cepii.loc[y, "implied_price"] / base_p
        m_dir = ("UP" if mi > prev_m else "DN") if prev_m else "—"
        d_dir = ("UP" if di > prev_d else "DN") if prev_d else "—"
        match = "✓" if m_dir == d_dir else "✗" if m_dir != "—" else "—"
        print(f"{y:>5}  {mi:>8.3f}  {di:>8.3f}  {m_dir:>6}  {d_dir:>6}  {match:>6}")
        prev_m, prev_d = mi, di


if __name__ == "__main__":
    main()
