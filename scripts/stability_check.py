#!/usr/bin/env python3
"""
Stability check: run every mineral × ban magnitude × start year scenario
N times and report whether results are consistent across runs.

Expected behaviour: ODE is deterministic (sigma_P=0, fixed seed=42).
All N runs should produce identical peak_idx, norm_yr, total_affected_yrs.
Any variation indicates numerical instability.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.predictability import (
    _GRAPHITE_2022_PARAMS, _RARE_EARTHS_2010_PARAMS,
    _COBALT_2016_PARAMS, _LITHIUM_2022_PARAMS,
    _NICKEL_2022_PARAMS, _URANIUM_2022_PARAMS,
)
from src.minerals.constants import ODE_DEFAULTS, SCENARIO_EXTRAS

BASELINE       = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY   = 0.9
NORM_THRESHOLD = 0.10
RESTRICTION_DURATION = 3
PROJECTION_END = 2035
N_RUNS = 5   # how many times to repeat each scenario

_MINERAL_PARAMS = {
    "graphite":    _GRAPHITE_2022_PARAMS,
    "rare_earths": _RARE_EARTHS_2010_PARAMS,
    "cobalt":      _COBALT_2016_PARAMS,
    "lithium":     _LITHIUM_2022_PARAMS,
    "nickel":      _NICKEL_2022_PARAMS,
    "uranium":     _URANIUM_2022_PARAMS,
}


def _stable_alpha_P(params):
    eta_D = params["eta_D"]
    return min(params["alpha_P"], EULER_SAFETY / max(abs(eta_D), 1e-6))


def _build_cfg(mineral, params, extra, magnitude, start_year, end_year):
    alpha_P = _stable_alpha_P(params)
    kw = {
        **ODE_DEFAULTS,
        "tau_K": params["tau_K"],
        "eta_D": params["eta_D"],
        "demand_growth": DemandGrowthConfig(type="constant", g=params["g"]),
        "alpha_P": alpha_P,
    }
    kw.update(extra)
    shocks = []
    if magnitude > 0:
        shocks.append(ShockConfig(
            type="export_restriction",
            start_year=start_year, end_year=end_year,
            magnitude=magnitude,
        ))
    return ScenarioConfig(
        name=f"{mineral}_stability",
        commodity=mineral,
        seed=42,
        time=TimeConfig(dt=1.0, start_year=2024, end_year=PROJECTION_END),
        baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def _norm_year(sc_df, bl_df, base_year, search_from):
    sc = sc_df.set_index("year")["P"]
    bl = bl_df.set_index("year")["P"]
    sc_base = sc.loc[base_year]
    bl_base = bl.loc[base_year]
    for yr in sorted(y for y in sc.index if y >= search_from and y in bl.index):
        if abs(sc.loc[yr] / sc_base - bl.loc[yr] / bl_base) < NORM_THRESHOLD:
            return yr
    return None


def run_one(mineral, params, extra, magnitude, start_yr):
    end_yr   = start_yr + RESTRICTION_DURATION - 1
    base_year = 2024

    cfg_ban  = _build_cfg(mineral, params, extra, magnitude, start_yr, end_yr)
    cfg_base = _build_cfg(mineral, params, extra, 0.0, start_yr, end_yr)

    df_ban,  _ = run_scenario(cfg_ban)
    df_base, _ = run_scenario(cfg_base)

    sc      = df_ban.set_index("year")["P"]
    sc_base = sc.loc[base_year]
    window  = {yr: sc.loc[yr] / sc_base for yr in sc.index if yr >= start_yr}
    peak_yr  = max(window, key=window.get)
    peak_idx = round(window[peak_yr], 6)

    norm_yr = _norm_year(df_ban, df_base, base_year, end_yr + 1)
    total   = (norm_yr - start_yr) if norm_yr else None

    return {"peak_idx": peak_idx, "norm_yr": norm_yr, "total_affected": total}


def main():
    print("=" * 70)
    print(f"STABILITY CHECK — {N_RUNS} runs per scenario")
    print("ODE is deterministic (sigma_P=0, seed=42). All runs should match.")
    print("=" * 70)

    magnitudes  = [0.25, 0.50, 0.75, 1.00]
    start_years = [2026, 2027]

    total_scenarios = 0
    unstable = []

    for mineral, params in _MINERAL_PARAMS.items():
        extra = SCENARIO_EXTRAS.get(mineral, {})
        print(f"\n{mineral.upper()}")
        print(f"  {'Start':>5}  {'Ban%':>5}  {'Run1 peak':>10}  {'All same?':>10}  {'Peak spread':>12}")
        print("  " + "─" * 55)

        for start_yr in start_years:
            for mag in magnitudes:
                results = [run_one(mineral, params, extra, mag, start_yr)
                           for _ in range(N_RUNS)]

                peaks  = [r["peak_idx"]      for r in results]
                norms  = [r["norm_yr"]        for r in results]
                totals = [r["total_affected"] for r in results]

                all_same = (len(set(peaks)) == 1 and
                            len(set(norms)) == 1 and
                            len(set(totals)) == 1)
                spread   = max(peaks) - min(peaks)

                status = "OK" if all_same else "UNSTABLE ⚠"
                if not all_same:
                    unstable.append((mineral, start_yr, mag, peaks, norms))

                print(f"  {start_yr:>5}  {int(mag*100):>4}%  "
                      f"{peaks[0]:>10.4f}  {status:>10}  {spread:>12.2e}")
                total_scenarios += 1

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {total_scenarios} scenarios × {N_RUNS} runs each")
    if unstable:
        print(f"UNSTABLE: {len(unstable)} scenarios showed variation across runs:")
        for mineral, start_yr, mag, peaks, norms in unstable:
            print(f"  {mineral} {int(mag*100)}% start {start_yr}: "
                  f"peaks={[round(p,4) for p in peaks]}, norm_yrs={norms}")
    else:
        print("ALL STABLE: every scenario produced identical results across all runs.")
        print("ODE is fully deterministic — no numerical instability detected.")


if __name__ == "__main__":
    main()
