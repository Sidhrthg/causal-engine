#!/usr/bin/env python3
"""
Concentration-weighted shock heuristic (China-share baseline).

This is the "strawman" version of the thesis's central empirical claim:
"disruption is proportional to the dominant supplier's market share."

For each year-on-year step, the heuristic predicts:
  ΔP > 0 (UP)   if Σ(shock_magnitude_i × direction_i) × dominant_share > 0
  ΔP < 0 (DOWN) if Σ(shock_magnitude_i × direction_i) × dominant_share < 0
  ΔP = prev      if no shock is active (momentum fallback)

Shock direction mapping:
  export_restriction  → +1 (supply reduction → price up)
  demand_surge > 0    → +1 (demand increase → price up)
  demand_surge < 0    → −1 (demand decrease → price down)
  capex_shock > 0     → +1 (deferred capacity → price up, lagged)
  macro_demand_shock  → −1 (demand destruction → price down)
  policy_shock        → +1 (quota restriction → price up)

This heuristic uses the same shock inputs as the causal engine but replaces
the full ODE simulation with a linear concentration weighting.

Purpose: establish whether the ODE machinery adds value beyond knowing
"China has 90% of graphite, so any restriction raises prices."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import numpy as np
import pandas as pd

from src.minerals.schema import load_scenario, ShockConfig
from src.minerals.predictability import _cepii_series


# ── Shock direction map ───────────────────────────────────────────────────────

def _shock_direction(shock: ShockConfig) -> float:
    """Return signed price-direction signal for a shock. Sign × magnitude."""
    if shock.type == "export_restriction":
        return +shock.magnitude
    if shock.type == "demand_surge":
        return +shock.magnitude  # positive = up, negative = down
    if shock.type == "capex_shock":
        return +shock.magnitude  # deferred investment → tighter future market
    if shock.type == "macro_demand_shock":
        return -abs(shock.magnitude)  # demand destruction → price down
    if shock.type in ("policy_shock",):
        return +shock.magnitude  # quota/restriction → price up
    if shock.type == "stockpile_release":
        return -shock.magnitude  # extra supply → price down
    return 0.0


def ch_directional_accuracy(shocks: list, price_series: pd.Series,
                             years: list, dominant_share: float) -> float:
    """
    Concentration heuristic directional accuracy.

    For each year t, aggregates active shock signals scaled by dominant_share.
    Predicts direction of t → t+1 price change.
    Falls back to momentum (carry last direction) if no shock is active.
    """
    correct = total = 0
    prev_direction = 0  # momentum carry

    for i in range(len(years) - 1):
        ya, yb = years[i], years[i + 1]
        if ya not in price_series.index or yb not in price_series.index:
            continue
        actual_delta = price_series.loc[yb] - price_series.loc[ya]
        if abs(actual_delta) < 1e-6:
            continue
        total += 1

        # Active shocks at start of year ya (cover ya through start of yb)
        active = [s for s in shocks if s.start_year <= ya <= s.end_year]
        net_signal = dominant_share * sum(_shock_direction(s) for s in active)

        if net_signal > 0.01:
            pred_dir = +1
        elif net_signal < -0.01:
            pred_dir = -1
        else:
            pred_dir = prev_direction  # momentum fallback

        if pred_dir != 0 and (pred_dir > 0) == (actual_delta > 0):
            correct += 1
        elif pred_dir == 0:
            # No information — abstain (count as 0.5 correct for scoring)
            correct += 0.5

        prev_direction = +1 if actual_delta > 0 else -1

    return correct / total if total > 0 else float("nan")


# ── Episode definitions (same shocks as predictability.py) ───────────────────

def _run_all():
    from src.minerals.predictability import _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS

    def _world_soy():
        df = pd.read_csv("data/canonical/cepii_soybeans.csv")
        world = df.groupby("year").agg(v=("value_kusd", "sum"), q=("quantity_tonnes", "sum")).reset_index()
        world["p"] = world["v"] / world["q"]
        return world.set_index("year")["p"]

    cg = _cepii_series("data/canonical/cepii_graphite.csv", "China")["implied_price"]
    cl = _cepii_series("data/canonical/cepii_lithium.csv",  "Chile")["implied_price"]
    cs = _world_soy()

    # ── 1. Graphite 2008 (China ~80% share in 2006-2011)
    shocks_g08 = [
        ShockConfig(type="demand_surge",       start_year=2008, end_year=2008, magnitude=0.46),
        ShockConfig(type="macro_demand_shock", start_year=2009, end_year=2009, magnitude=-0.40, demand_destruction=-0.40),
        ShockConfig(type="policy_shock",       start_year=2010, end_year=2011, magnitude=0.35, quota_reduction=0.35),
        ShockConfig(type="capex_shock",        start_year=2010, end_year=2011, magnitude=0.50),
    ]
    da_g08 = ch_directional_accuracy(shocks_g08, cg, [2006,2007,2008,2009,2010,2011], 0.80)

    # ── 2. Graphite 2022 (China ~90% share in 2021-2024)
    shocks_g22 = [
        ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
        ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
        ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.30),
        ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.05),
        ShockConfig(type="stockpile_release",  start_year=2023, end_year=2023, magnitude=20.0),
    ]
    da_g22 = ch_directional_accuracy(shocks_g22, cg, [2021,2022,2023,2024], 0.90)

    # ── 3. Lithium 2016 (Chile ~43% share — non-China critical mineral)
    shocks_li16 = [
        ShockConfig(type="demand_surge", start_year=2016, end_year=2016, magnitude=0.25),
        ShockConfig(type="demand_surge", start_year=2017, end_year=2017, magnitude=0.20),
    ]
    years_li16 = [y for y in [2015,2016,2017,2018,2019] if y in cl.index]
    da_li16 = ch_directional_accuracy(shocks_li16, cl, years_li16, 0.43)

    # ── 4. Lithium 2022 (Australia ~55% share)
    shocks_li22 = [
        ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30),
    ]
    da_li22 = ch_directional_accuracy(shocks_li22, cl, [2021,2022,2023,2024], 0.55)

    # ── 5. Rare earths 2010 (China ~97% share)
    cr = _cepii_series("data/canonical/cepii_rare_earths.csv", "China")["implied_price"]
    shocks_re10 = [
        ShockConfig(type="demand_surge",  start_year=2010, end_year=2010, magnitude=0.30),
        ShockConfig(type="policy_shock",  start_year=2010, end_year=2012, magnitude=0.40, quota_reduction=0.40),
        ShockConfig(type="capex_shock",   start_year=2010, end_year=2012, magnitude=0.50),
    ]
    years_re10 = [y for y in [2009,2010,2011,2012,2013] if y in cr.index]
    da_re10 = ch_directional_accuracy(shocks_re10, cr, years_re10, 0.97)

    # ── 6. Soybeans 2011
    cfg_s11 = load_scenario("scenarios/soybeans_2011_food_crisis.yaml")
    da_s11 = ch_directional_accuracy(cfg_s11.shocks, cs, [2009,2010,2011], 0.35)

    # ── 5. Soybeans 2015
    cfg_s15 = load_scenario("scenarios/soybeans_2015_supply_glut.yaml")
    da_s15 = ch_directional_accuracy(cfg_s15.shocks, cs, [2014,2015,2016,2017], 0.35)

    # ── 6. Soybeans 2018
    cfg_s18 = load_scenario("scenarios/soybeans_2018_trade_war.yaml")
    years_s18 = [y for y in [2016,2017,2018,2020,2021] if y in cs.index]
    da_s18 = ch_directional_accuracy(cfg_s18.shocks, cs, years_s18, 0.35)

    # ── 7. Soybeans 2020
    cfg_s20 = load_scenario("scenarios/soybeans_2020_phase1.yaml")
    years_s20 = [y for y in [2018,2020,2021] if y in cs.index]
    da_s20 = ch_directional_accuracy(cfg_s20.shocks, cs, years_s20, 0.35)

    # ── 8. Soybeans 2022 ukraine
    shocks_s22 = [
        ShockConfig(type="demand_surge", start_year=2021, end_year=2021, magnitude=0.20),
        ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.10),
        ShockConfig(type="capex_shock",  start_year=2022, end_year=2022, magnitude=0.15),
    ]
    years_s22 = [y for y in [2020,2021,2022,2023,2024] if y in cs.index]
    da_s22 = ch_directional_accuracy(shocks_s22, cs, years_s22, 0.35)

    episodes = [
        ("graphite_2008",      1.000, da_g08,  0.80),
        ("graphite_2022",      1.000, da_g22,  0.90),
        ("rare_earths_2010",   1.000, da_re10, 0.97),
        ("lithium_2016",       1.000, da_li16, 0.43),
        ("lithium_2022",       1.000, da_li22, 0.55),
        ("soybeans_2011",      1.000, da_s11,  0.35),
        ("soybeans_2015",      0.667, da_s15,  0.35),
        ("soybeans_2018",      0.500, da_s18,  0.35),
        ("soybeans_2020",      1.000, da_s20,  0.35),
        ("soybeans_2022",      1.000, da_s22,  0.35),
    ]

    # Also get the 4 existing baselines for comparison
    from src.minerals.baseline_comparison import run_baseline_comparison, summary_stats
    baseline_results = run_baseline_comparison()
    baseline_map = {r.episode.replace("_demand_spike_and_quota", "_2008")
                              .replace("_ev_surge_and_export_controls", "_2022")
                              .replace("graphite_2022_ev_", "graphite_2022_")
                              : r for r in baseline_results}
    # build by order
    baseline_by_order = {r.episode: r for r in baseline_results}

    print("=" * 90)
    print("BASELINE COMPARISON INCLUDING CONCENTRATION HEURISTIC (CH)")
    print("=" * 90)
    print(f"{'Episode':<32} {'Causal':>7} {'CH':>7} {'Momentm':>8} {'AR(1)':>7} {'MeanRev':>8} {'BestBase':>9}")
    print("-" * 90)

    all_causal, all_ch, all_mom, all_ar1, all_mr = [], [], [], [], []
    for ep_short, causal_da, ch_da, dom_share in episodes:
        # find matching baseline result
        br = None
        for r in baseline_results:
            if ep_short.replace("_", "") in r.episode.replace("_", ""):
                br = r
                break
        mom = br.momentum_da if br else float("nan")
        ar1 = br.ar1_da if br else float("nan")
        mr  = br.mean_reversion_da if br else float("nan")
        best = max([v for v in [mom, ar1, mr] if not math.isnan(v)] or [0])

        all_causal.append(causal_da)
        all_ch.append(ch_da)
        if not math.isnan(mom): all_mom.append(mom)
        if not math.isnan(ar1): all_ar1.append(ar1)
        if not math.isnan(mr):  all_mr.append(mr)

        def fmt(v): return f"{v:.3f}" if not math.isnan(v) else "  N/A"
        dom_s = f"({int(dom_share*100)}%)"
        print(f"{ep_short:<28} {dom_s:<4} {fmt(causal_da):>7} {fmt(ch_da):>7} {fmt(mom):>8} {fmt(ar1):>7} {fmt(mr):>8} {fmt(best):>9}")

    print("-" * 90)
    n = len(all_causal)
    mean_causal = sum(all_causal) / n
    mean_ch     = sum(all_ch) / n
    mean_mom    = sum(all_mom) / len(all_mom) if all_mom else float("nan")
    mean_ar1    = sum(all_ar1) / len(all_ar1) if all_ar1 else float("nan")
    mean_mr     = sum(all_mr) / len(all_mr) if all_mr else float("nan")
    mean_best   = max(v for v in [mean_mom, mean_ar1, mean_mr] if not math.isnan(v))
    print(f"{'Mean':<32} {mean_causal:>7.3f} {mean_ch:>7.3f} {mean_mom:>8.3f} {mean_ar1:>7.3f} {mean_mr:>8.3f} {mean_best:>9.3f}")
    print()
    print(f"Causal vs Momentum:     {(mean_causal - mean_mom)*100:+.1f} pp")
    print(f"Causal vs CH heuristic: {(mean_causal - mean_ch)*100:+.1f} pp")
    print(f"CH vs Momentum:         {(mean_ch - mean_mom)*100:+.1f} pp")
    print(f"Causal vs best baseline:{(mean_causal - mean_best)*100:+.1f} pp")


if __name__ == "__main__":
    _run_all()
