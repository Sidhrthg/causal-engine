#!/usr/bin/env python3
"""
Stockpile analysis: lead time to price impact and minimum release schedule.

The ODE is calibrated per-episode for DA maximisation over 4-year windows.
Episode-calibrated parameters with |alpha_P * eta_D| > 1 are unstable under
annual Euler integration beyond the calibrated window — they produce limit
cycles rather than convergent long-run trajectories.

Fix: for long-horizon inventory projection, alpha_P is capped at
0.9 / |eta_D| — the maximum value compatible with Euler stability at dt=1.
This is the stability-preserving projection value. Economically justified:
high episode-calibrated alpha_P values capture short-run crisis dynamics;
multi-year stockpile planning requires the long-run price adjustment speed.

Episodes with |alpha_P * eta_D| < 1 (graphite_2008, lithium_2022, nickel_2022)
run with original parameters. Others use stabilized alpha_P.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass
from typing import Optional, Tuple

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.predictability import (
    _GRAPHITE_2008_PARAMS, _GRAPHITE_2022_PARAMS,
    _LITHIUM_2022_PARAMS, _SOYBEANS_2022_PARAMS,
    _COBALT_2016_PARAMS, _COBALT_2022_PARAMS,
    _NICKEL_2022_PARAMS,
    _RARE_EARTHS_2010_PARAMS,
    _URANIUM_2007_PARAMS, _URANIUM_2022_PARAMS,
)
from src.minerals.constants import ODE_DEFAULTS


# ── Stability fix ─────────────────────────────────────────────────────────────

def _projection_alpha_P(params: dict, safety: float = 0.9) -> float:
    """
    Return alpha_P capped at the Euler stability limit for annual timesteps.
    Stability condition: |alpha_P * eta_D| < 1.
    Cap at safety / |eta_D| (default safety=0.9).
    """
    limit = safety / max(abs(params["eta_D"]), 1e-6)
    return min(params["alpha_P"], limit)


def _verify_baseline_stable(params: dict, time_cfg, baseline, label: str) -> bool:
    """Run a no-shock baseline and confirm cover stays bounded."""
    aP = _projection_alpha_P(params)
    cfg = ScenarioConfig(
        name=f"{label}_baseline", commodity="graphite", seed=42,
        time=time_cfg, baseline=baseline,
        parameters=ParametersConfig(
            **ODE_DEFAULTS,
            tau_K=params["tau_K"], eta_D=params["eta_D"],
            demand_growth=DemandGrowthConfig(type="constant", g=params["g"]),
            alpha_P=aP,
        ),
        policy=PolicyConfig(), shocks=[],
        outputs=OutputsConfig(metrics=["avg_price"]),
    )
    df, _ = run_scenario(cfg)
    # Cover should stay in [0, 5] — wild oscillation shows as values > 2 or < 0
    cover_vals = df["cover"].tolist()
    max_cover = max(abs(c) for c in cover_vals)
    stable = max_cover < 3.0
    capped = aP < params["alpha_P"]
    print(f"  Baseline check [{label}]: alpha_P_orig={params['alpha_P']:.3f} "
          f"alpha_P_proj={aP:.3f} {'(CAPPED)' if capped else ''} "
          f"max_cover={max_cover:.2f} {'OK' if stable else 'UNSTABLE'}")
    return stable


# ── Core helpers ──────────────────────────────────────────────────────────────

def _build_cfg(name, params, time_cfg, baseline, shocks, extra=None):
    aP = _projection_alpha_P(params)
    kw = {
        **ODE_DEFAULTS,
        "tau_K": params["tau_K"],
        "eta_D": params["eta_D"],
        "demand_growth": DemandGrowthConfig(type="constant", g=params["g"]),
        "alpha_P": aP,
    }
    if extra:
        kw.update(extra)
    return ScenarioConfig(
        name=name, commodity="graphite", seed=42,
        time=time_cfg, baseline=baseline,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(), shocks=shocks,
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def _cover_traj(params, time_cfg, baseline, shocks, extra=None) -> dict:
    cfg = _build_cfg("_cov", params, time_cfg, baseline, shocks, extra)
    df, _ = run_scenario(cfg)
    return dict(zip(df["year"].tolist(), df["cover"].tolist()))


def _breach_year(cover: dict, cover_star: float, from_year: int) -> Optional[int]:
    for y in sorted(cover):
        if y < from_year:
            continue
        if cover[y] < cover_star:
            return y
    return None


def _min_release(params, time_cfg, baseline, shocks, shock_onset,
                  cover_star=0.20, demand_kt=100.0, extra=None) -> Tuple[Optional[float], Optional[float]]:
    """Binary search for minimum one-period release that prevents breach."""
    cov = _cover_traj(params, time_cfg, baseline, shocks, extra)
    if _breach_year(cov, cover_star, shock_onset) is None:
        return None, None   # no breach without stockpile

    def _ok(kt: float) -> bool:
        s_with = shocks + [ShockConfig(
            type="stockpile_release",
            start_year=shock_onset, end_year=shock_onset, magnitude=kt,
        )]
        c = _cover_traj(params, time_cfg, baseline, s_with, extra)
        return _breach_year(c, cover_star, shock_onset) is None

    lo, hi = 0.0, demand_kt * 2.0
    # Check if ceiling is enough
    if not _ok(hi):
        return float("inf"), float("inf")
    for _ in range(40):
        mid = (lo + hi) / 2
        if _ok(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < 0.5:
            break
    months = (hi / demand_kt) * 12
    return months, hi


# ── Episode runner ────────────────────────────────────────────────────────────

@dataclass
class StockpileResult:
    episode: str
    commodity: str
    alpha_P_orig: float
    alpha_P_proj: float
    shock_onset: int
    breach_year: Optional[int]
    lead_time: Optional[int]
    no_breach: bool
    min_months: Optional[float]
    min_kt: Optional[float]
    demand_kt: float
    cover_traj: dict    # no stockpile


def run_episode(label, commodity, params, time_cfg, baseline,
                shocks, shock_onset, years, extra=None) -> StockpileResult:
    cov = _cover_traj(params, time_cfg, baseline, shocks, extra)
    cov_sub = {y: cov[y] for y in years if y in cov}

    breach = _breach_year(cov, 0.20, shock_onset)
    lead   = (breach - shock_onset) if breach is not None else None

    months, kt = _min_release(
        params, time_cfg, baseline, shocks, shock_onset,
        demand_kt=baseline.D0, extra=extra,
    )

    return StockpileResult(
        episode=label, commodity=commodity,
        alpha_P_orig=params["alpha_P"],
        alpha_P_proj=_projection_alpha_P(params),
        shock_onset=shock_onset,
        breach_year=breach,
        lead_time=lead,
        no_breach=(breach is None),
        min_months=months,
        min_kt=kt,
        demand_kt=baseline.D0,
        cover_traj=cov_sub,
    )


def main():
    # I0 = 50 kt → cover_0 = 50/100 = 0.50 (6 months of consumption).
    # Represents a modest national strategic reserve — realistic starting point.
    base = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=50.0, D0=100.0)

    print("=" * 70)
    print("BASELINE STABILITY VERIFICATION (no shocks, 8-year horizon)")
    print("=" * 70)

    episodes_to_check = [
        ("graphite_2008",   _GRAPHITE_2008_PARAMS),
        ("graphite_2022",   _GRAPHITE_2022_PARAMS),
        ("lithium_2022",    _LITHIUM_2022_PARAMS),
        ("cobalt_2016",     _COBALT_2016_PARAMS),
        ("soybeans_2022",   _SOYBEANS_2022_PARAMS),
        ("nickel_2022",     _NICKEL_2022_PARAMS),
        ("rare_earths_2010",_RARE_EARTHS_2010_PARAMS),
        ("uranium_2007",    _URANIUM_2007_PARAMS),
        ("uranium_2022",    _URANIUM_2022_PARAMS),
    ]
    tc8 = TimeConfig(dt=1.0, start_year=2019, end_year=2027)
    for label, p in episodes_to_check:
        _verify_baseline_stable(p, tc8, base, label)

    print()
    print("=" * 70)
    print("STOCKPILE ANALYSIS (all using stability-corrected alpha_P)")
    print("=" * 70)

    results = []

    # ── 1. Graphite 2023 export controls ─────────────────────────────────────
    r = run_episode(
        "graphite_2023_export_controls", "graphite",
        _GRAPHITE_2022_PARAMS,
        TimeConfig(dt=1.0, start_year=2019, end_year=2028),
        base,
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
            ShockConfig(type="export_restriction", start_year=2023, end_year=2025, magnitude=0.35),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.30),
            ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.05),
        ],
        shock_onset=2023,
        years=list(range(2021, 2028)),
        extra=dict(substitution_elasticity=0.8, substitution_cap=0.6),
    )
    results.append(r)

    # ── 2. Graphite 2008 export quota ─────────────────────────────────────────
    r = run_episode(
        "graphite_2008_export_quota", "graphite",
        _GRAPHITE_2008_PARAMS,
        TimeConfig(dt=1.0, start_year=2004, end_year=2014),
        base,
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2008, end_year=2008, magnitude=0.46),
            ShockConfig(type="macro_demand_shock", start_year=2009, end_year=2009,
                        magnitude=-0.40, demand_destruction=-0.40),
            ShockConfig(type="policy_shock",       start_year=2010, end_year=2011,
                        magnitude=0.35, quota_reduction=0.35),
            ShockConfig(type="capex_shock",        start_year=2010, end_year=2011, magnitude=0.50),
        ],
        shock_onset=2010,
        years=list(range(2006, 2014)),
    )
    results.append(r)

    # ── 3. Lithium 2022 EV demand boom ────────────────────────────────────────
    r = run_episode(
        "lithium_2022_ev_boom", "lithium",
        _LITHIUM_2022_PARAMS,
        TimeConfig(dt=1.0, start_year=2019, end_year=2028),
        base,
        shocks=[
            ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30),
        ],
        shock_onset=2022,
        years=list(range(2020, 2028)),
        extra=dict(fringe_capacity_share=0.4, fringe_entry_price=1.1),
    )
    results.append(r)

    # ── 4. Cobalt 2016 EV speculation ─────────────────────────────────────────
    r = run_episode(
        "cobalt_2016_ev_speculation", "cobalt",
        _COBALT_2016_PARAMS,
        TimeConfig(dt=1.0, start_year=2013, end_year=2022),
        base,
        shocks=[
            ShockConfig(type="demand_surge", start_year=2016, end_year=2018, magnitude=0.25),
            ShockConfig(type="demand_surge", start_year=2019, end_year=2019, magnitude=-0.30),
        ],
        shock_onset=2016,
        years=list(range(2014, 2022)),
    )
    results.append(r)

    # ── 5. Soybeans 2022 Ukraine ──────────────────────────────────────────────
    r = run_episode(
        "soybeans_2022_ukraine", "soybeans",
        _SOYBEANS_2022_PARAMS,
        TimeConfig(dt=1.0, start_year=2018, end_year=2028),
        base,
        shocks=[
            ShockConfig(type="demand_surge", start_year=2021, end_year=2021, magnitude=0.20),
            ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.10),
            ShockConfig(type="capex_shock",  start_year=2022, end_year=2022, magnitude=0.15),
        ],
        shock_onset=2022,
        years=list(range(2020, 2028)),
    )
    results.append(r)

    # ── 6. Nickel 2022 HPAL oversupply crash ─────────────────────────────────
    r = run_episode(
        "nickel_2022_hpal_crash", "nickel",
        _NICKEL_2022_PARAMS,
        TimeConfig(dt=1.0, start_year=2019, end_year=2028),
        base,
        shocks=[
            ShockConfig(type="demand_surge",      start_year=2021, end_year=2021, magnitude=0.20),
            ShockConfig(type="demand_surge",      start_year=2022, end_year=2022, magnitude=0.25),
            ShockConfig(type="stockpile_release", start_year=2023, end_year=2023, magnitude=25.0),
            ShockConfig(type="demand_surge",      start_year=2023, end_year=2023, magnitude=-0.20),
            ShockConfig(type="demand_surge",      start_year=2024, end_year=2024, magnitude=-0.15),
        ],
        shock_onset=2022,
        years=list(range(2020, 2028)),
        extra=dict(fringe_capacity_share=0.45, fringe_entry_price=1.15),
    )
    results.append(r)

    # ── 7. Rare earths 2010 China export quota ────────────────────────────────
    r = run_episode(
        "rare_earths_2010_china_quota", "graphite",  # ODE commodity-agnostic
        _RARE_EARTHS_2010_PARAMS,
        TimeConfig(dt=1.0, start_year=2007, end_year=2017),
        base,
        shocks=[
            ShockConfig(type="demand_surge",  start_year=2010, end_year=2010, magnitude=0.30),
            ShockConfig(type="policy_shock",  start_year=2010, end_year=2012, magnitude=0.40, quota_reduction=0.40),
            ShockConfig(type="capex_shock",   start_year=2010, end_year=2012, magnitude=0.50),
        ],
        shock_onset=2010,
        years=list(range(2008, 2017)),
    )
    results.append(r)

    # ── 8. Uranium 2007 Cigar Lake supply squeeze ─────────────────────────────
    r = run_episode(
        "uranium_2007_cigar_lake", "graphite",
        _URANIUM_2007_PARAMS,
        TimeConfig(dt=1.0, start_year=2003, end_year=2013),
        base,
        shocks=[
            ShockConfig(type="demand_surge",       start_year=2005, end_year=2006, magnitude=0.15),
            ShockConfig(type="capex_shock",        start_year=2006, end_year=2007, magnitude=0.60),
            ShockConfig(type="export_restriction", start_year=2007, end_year=2008, magnitude=0.20),
        ],
        shock_onset=2006,
        years=list(range(2004, 2013)),
    )
    results.append(r)

    # ── 9. Uranium 2022 Russia sanctions ──────────────────────────────────────
    r = run_episode(
        "uranium_2022_russia_sanctions", "graphite",
        _URANIUM_2022_PARAMS,
        TimeConfig(dt=1.0, start_year=2019, end_year=2029),
        base,
        shocks=[
            ShockConfig(type="export_restriction", start_year=2022, end_year=2024, magnitude=0.25),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2025, magnitude=0.15),
        ],
        shock_onset=2022,
        years=list(range(2020, 2029)),
    )
    results.append(r)

    # ── Print ─────────────────────────────────────────────────────────────────
    print()
    for r in results:
        breach = str(r.breach_year) if not r.no_breach else "none"
        lt     = f"{r.lead_time}yr" if r.lead_time is not None else "—"
        capped = " (stabilized)" if r.alpha_P_proj < r.alpha_P_orig else ""
        print(f"\n{r.episode}")
        print(f"  α_P: {r.alpha_P_orig:.3f} → {r.alpha_P_proj:.3f}{capped}")
        print(f"  Shock onset: {r.shock_onset}   Breach: {breach}   Lead time: {lt}")
        if not r.no_breach:
            if r.min_months == float("inf"):
                print(f"  Min stockpile: exceeds 2-year ceiling — pre-positioning required")
            elif r.min_months is not None:
                print(f"  Min stockpile: {r.min_months:.1f} months ({r.min_kt:.0f} kt in year 1)")
        else:
            print(f"  No breach — existing inventory sufficient")
        cov_str = "  Cover: " + "  ".join(f"{y}:{v:.2f}" for y, v in sorted(r.cover_traj.items()))
        print(cov_str)

    # Summary table
    print("\n\n" + "=" * 78)
    print("TABLE 5.1 — LEAD TIME AND MINIMUM STOCKPILE REQUIREMENTS")
    print("(I₀ = 50 kt = 6 months consumption; cover* = 0.20 = 2.4 months)")
    print("=" * 78)
    print(f"{'Episode':<35} {'Onset':>6} {'Breach':>7} {'Lead':>6} {'Min stockpile':>14} {'α_P note'}")
    print("-" * 78)
    for r in results:
        breach = str(r.breach_year) if not r.no_breach else "none"
        lt     = f"{r.lead_time}yr"  if r.lead_time is not None else "—"
        if r.no_breach:
            ms = "—"
        elif r.min_months == float("inf"):
            ms = ">24 mo (pre-pos.)"
        else:
            ms = f"{r.min_months:.1f} mo / {r.min_kt:.0f} kt"
        note = f"stabilized {r.alpha_P_proj:.2f}" if r.alpha_P_proj < r.alpha_P_orig else "original"
        print(f"{r.episode:<35} {r.shock_onset:>6} {breach:>7} {lt:>6} {ms:>14}  {note}")

    print()
    print("Notes:")
    print("  α_P stabilized = capped at 0.9/|η_D| for Euler stability at dt=1yr.")
    print("  Represents long-run price adjustment speed, not short-run episode dynamics.")
    print("  Min stockpile = minimum one-period release at shock onset to prevent breach.")
    print("  'Pre-positioning required' = no one-period release ≤ 24mo prevents breach.")


if __name__ == "__main__":
    main()
