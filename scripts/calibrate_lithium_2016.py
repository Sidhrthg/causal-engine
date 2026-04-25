#!/usr/bin/env python3
"""
Calibrate structural parameters for lithium 2016-2019 EV first-wave episode.

Episode: Chilean lithium carbonate prices rose 48%/41%/23% in 2016/17/18
driven by the first EV demand wave (Tesla Model 3, Chinese NEV mandates),
then fell 22% in 2019 as Chinese EV subsidies were cut and Australian
hard-rock supply (Pilbara, Greenbushes) came online.

Calibration target: maximize DA + Spearman rho against CEPII Chile unit values.
Search over {alpha_P, eta_D, tau_K, g} via scipy differential_evolution.
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

# ── Episode definition ─────────────────────────────────────────────────────────

CEPII_PATH  = "data/canonical/cepii_lithium.csv"
DOM_EXPORTER = "Chile"

# Documented shocks:
# 2016: Chinese NEV mandate + Tesla Model 3 demand surge
# 2017-2018: Continued EV demand, supply lagging
# 2019: Chinese EV subsidy cuts (~50% reduction)
# 2020: COVID-19 demand destruction + subsidy further reduced
SHOCKS = [
    ShockConfig(type="demand_surge", start_year=2016, end_year=2016, magnitude=0.35),
    ShockConfig(type="demand_surge", start_year=2017, end_year=2017, magnitude=0.25),
    ShockConfig(type="demand_surge", start_year=2018, end_year=2018, magnitude=0.12),
    ShockConfig(type="demand_surge", start_year=2019, end_year=2019, magnitude=-0.25),
    ShockConfig(type="demand_surge", start_year=2020, end_year=2020, magnitude=-0.18),
]

TIME_CFG  = TimeConfig(dt=1.0, start_year=2011, end_year=2022)
BASELINE  = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)

# Evaluation window: 5 directional steps
YEARS    = [2014, 2015, 2016, 2017, 2018, 2019]
BASE_YEAR = 2014

# Fringe supply: Australian hard-rock mining enters as Chilean prices spike
EXTRA = dict(fringe_capacity_share=0.35, fringe_entry_price=1.40)


# ── Calibration objective ──────────────────────────────────────────────────────

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
        name="li16_cal", commodity="lithium", seed=42,
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
        return da, rho
    except Exception:
        return 0.0, 0.0


def objective(x):
    alpha_P, eta_D, tau_K, g = x
    cepii = _cepii_series(CEPII_PATH, DOM_EXPORTER)
    da, rho = _run(alpha_P, eta_D, tau_K, g, cepii)

    # Magnitude penalty: model price index range should roughly match data range
    # Without this, DE exploits Spearman/DA being rank-only and finds
    # parameters that produce correct directions but wildly wrong magnitudes.
    try:
        kw = dict(
            eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
            eta_K=0.40, retire_rate=0.0,
            cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
            tau_K=tau_K, eta_D=eta_D,
            demand_growth=DemandGrowthConfig(type="constant", g=g),
            alpha_P=alpha_P, **EXTRA,
        )
        cfg = ScenarioConfig(
            name="li16_mag", commodity="lithium", seed=42,
            time=TIME_CFG, baseline=BASELINE,
            parameters=ParametersConfig(**kw),
            policy=PolicyConfig(), shocks=SHOCKS,
            outputs=OutputsConfig(metrics=["avg_price"]),
        )
        df, _ = run_scenario(cfg)
        m = df.set_index("year")
        model_idx = m.loc[YEARS, "P"] / m.loc[BASE_YEAR, "P"]
        data_idx  = cepii.loc[YEARS, "implied_price"] / cepii.loc[BASE_YEAR, "implied_price"]
        # Log-RMSE between model and data price indices
        log_rmse = float(np.sqrt(np.mean((np.log(model_idx + 1e-9) - np.log(data_idx + 1e-9))**2)))
        mag_penalty = min(log_rmse / 2.0, 1.0)   # saturates at 1.0 for large errors
    except Exception:
        mag_penalty = 1.0

    return -(da + rho - mag_penalty)   # minimise negative


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cepii = _cepii_series(CEPII_PATH, DOM_EXPORTER)

    print("=" * 60)
    print("Calibrating lithium_2016 episode")
    print(f"Target years: {YEARS}, base={BASE_YEAR}")
    print("CEPII Chile price index:")
    base_p = cepii.loc[BASE_YEAR, "implied_price"]
    for y in YEARS:
        p = cepii.loc[y, "implied_price"]
        print(f"  {y}: {p/base_p:.3f}  ({'+' if p>cepii.loc[YEARS[YEARS.index(y)-1],'implied_price'] and YEARS.index(y)>0 else '-'})")
    print()

    bounds = [
        (0.10, 4.00),   # alpha_P
        (-1.50, -0.01), # eta_D
        (0.50, 20.0),   # tau_K
        (0.92,  1.25),  # g — lithium demand growth realistic range (EV first wave)
    ]

    print("Running differential evolution (may take 30-60s)...")
    result = differential_evolution(
        objective, bounds,
        seed=42, maxiter=300, popsize=15, tol=1e-4,
        mutation=(0.5, 1.5), recombination=0.7,
        workers=1,
    )

    alpha_P, eta_D, tau_K, g = result.x
    da, rho = _run(alpha_P, eta_D, tau_K, g, cepii)

    print()
    print("=" * 60)
    print("CALIBRATION RESULT")
    print("=" * 60)
    print(f"  alpha_P = {alpha_P:.4f}")
    print(f"  eta_D   = {eta_D:.4f}")
    print(f"  tau_K   = {tau_K:.4f}")
    print(f"  g       = {g:.4f}")
    print(f"  DA      = {da:.3f}")
    print(f"  rho     = {rho:.3f}")
    print()
    print("Add to predictability.py:")
    print(f"_LITHIUM_2016_PARAMS = dict(alpha_P={alpha_P:.3f}, eta_D={eta_D:.3f}, "
          f"tau_K={tau_K:.3f}, g={g:.4f})")

    # Also show year-by-year
    print()
    print("Year-by-year model vs data:")
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        eta_K=0.40, retire_rate=0.0,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
        tau_K=tau_K, eta_D=eta_D,
        demand_growth=DemandGrowthConfig(type="constant", g=g),
        alpha_P=alpha_P, **EXTRA,
    )
    cfg = ScenarioConfig(
        name="li16_final", commodity="lithium", seed=42,
        time=TIME_CFG, baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(), shocks=SHOCKS,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    m = df.set_index("year")
    base_p_data = cepii.loc[BASE_YEAR, "implied_price"]
    base_p_model = m.loc[BASE_YEAR, "P"]
    print(f"{'Year':>5}  {'Model':>8}  {'Data':>8}  {'M-dir':>6}  {'D-dir':>6}  {'Match':>6}")
    prev_m = prev_d = None
    for y in YEARS:
        mi = m.loc[y, "P"] / base_p_model
        di = cepii.loc[y, "implied_price"] / base_p_data
        m_dir = ("UP" if mi > prev_m else "DN") if prev_m else "—"
        d_dir = ("UP" if di > prev_d else "DN") if prev_d else "—"
        match = "✓" if m_dir == d_dir else "✗" if m_dir != "—" else "—"
        print(f"{y:>5}  {mi:>8.3f}  {di:>8.3f}  {m_dir:>6}  {d_dir:>6}  {match:>6}")
        prev_m, prev_d = mi, di


if __name__ == "__main__":
    main()
