#!/usr/bin/env python3
"""
Calibrate structural parameters for rare earths 2014-2018 post-WTO oversupply.

After the WTO ruled against China's quotas (Aug 2014), Beijing eliminated
export quotas and Chinese suppliers reasserted market share by flooding the
market. CEPII China unit value fell from $13.11/kg (2014) to $8.43/kg (2017),
a 36% decline over three years.

This is a STRUCTURAL REGIME COMPLEMENT to rare_earths_2010 (the restriction era):
  - 2010 episode: supply restriction → demand-pull amplification (alpha_P high)
  - 2014 episode: supply flood → no amplification, supply-side overhang dominates

OOS test: do 2010 (restriction) params predict 2014 (oversupply) directionally,
and vice versa? If structural params are regime-specific, OOS DA will fall.
If they're commodity-level (same alpha_P, eta_D, tau_K), OOS DA stays high.

Episode window: 2014-2018 (4 directional steps)
Calibration: differential_evolution maximising DA + Spearman rho - log_rmse penalty
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
from src.minerals.constants import ODE_DEFAULTS, SCENARIO_EXTRAS

CEPII_PATH   = "data/canonical/cepii_rare_earths.csv"
DOM_EXPORTER = "China"

# Post-WTO oversupply mechanics:
#   2014: WTO ruling Aug 2014 forced quota removal; Chinese suppliers begin reassertion
#   2015: Continued price destruction as Chinese exports rebuild; Mountain Pass bankruptcy
#   2016: Bottom of the cycle; China consolidates SOE producers (six majors)
#   2017: Further decline as new Chinese capacity comes online
#   2018: Stabilisation at lower equilibrium
SHOCKS = [
    # Stockpile release captures Chinese inventory dump post-quota removal
    ShockConfig(type="stockpile_release", start_year=2015, end_year=2015, magnitude=15.0),
    # Demand destruction: substitution + EV/wind delays + investor exit (Molycorp Ch.11 2015)
    ShockConfig(type="demand_surge", start_year=2015, end_year=2015, magnitude=-0.20),
    ShockConfig(type="demand_surge", start_year=2016, end_year=2016, magnitude=-0.10),
    ShockConfig(type="demand_surge", start_year=2017, end_year=2017, magnitude=-0.15),
]

TIME_CFG = TimeConfig(dt=1.0, start_year=2012, end_year=2020)
BASELINE = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)

YEARS     = [2014, 2015, 2016, 2017, 2018]
BASE_YEAR = 2014

EXTRA = SCENARIO_EXTRAS["rare_earths"]


def _run(alpha_P, eta_D, tau_K, g, cepii):
    kw = {**ODE_DEFAULTS, "tau_K": tau_K, "eta_D": eta_D,
          "demand_growth": DemandGrowthConfig(type="constant", g=g),
          "alpha_P": alpha_P, **EXTRA}
    cfg = ScenarioConfig(
        name="re14_cal", commodity="rare_earths", seed=42,
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
    print("Calibrating rare_earths_2014 episode (post-WTO oversupply)")
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
        (0.85,  1.10),  # g — flat or declining (oversupply era)
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
    print(f"_RARE_EARTHS_2014_PARAMS = dict(alpha_P={alpha_P:.3f}, eta_D={eta_D:.3f}, "
          f"tau_K={tau_K:.3f}, g={g:.4f})")

    # Year-by-year
    kw = {**ODE_DEFAULTS, "tau_K": tau_K, "eta_D": eta_D,
          "demand_growth": DemandGrowthConfig(type="constant", g=g),
          "alpha_P": alpha_P, **EXTRA}
    cfg = ScenarioConfig(
        name="re14_final", commodity="rare_earths", seed=42,
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
