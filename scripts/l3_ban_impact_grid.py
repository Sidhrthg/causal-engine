#!/usr/bin/env python3
"""
L3 Ban Impact Grid — Pearl Layer 3 forward projection.

Research question:
  "Given the actual market trajectory through 2024, how long would the US
   remain affected if China bans X% of exports in 2026 or 2027?"

Why L3 not L2:
  L2 (forward_scenario_2025.py) starts from a hypothetically CLEAN 2024
  baseline — it ignores carry-forward from recent market dynamics:
    - Graphite: export controls ongoing since Oct 2023 (inventory already depleted)
    - Cobalt: deep oversupply in 2023-2024 (U_2024 = -2.128; LFP transition)
    - Lithium: near equilibrium post-2022 crash (U_2024 = +0.154)
    - Nickel: moderately tight from Indonesia ban aftermath (U_2024 = +0.905)

  L3 conditions on the ACTUAL observed trajectory via abducted residuals U_t.
  U_t = log(P_cepii_norm(t)) - log(P_model_norm(t)) captures:
    - Speculative dynamics above/below structural model
    - Inventory depletion / build-up from recent shocks
    - Demand pattern carry-forward

  Starting from the true 2024 market state changes the answer:
    - Cobalt in deep oversupply → a ban first brings market back to balance
      before causing a spike (L3 peak much lower than L2's ~10×)
    - Graphite already tight from 2023 controls → ban compounds on tighter
      starting inventory (L3 lag potentially longer than L2)

Three-step L3 (Abduction → Action → Prediction):
  Step 1: Abduct U_t from CEPII historical prices (2019-2024 per mineral).
          U_t = 0 for 2025+ (no future observations available).
  Step 2: do(export_restriction_2026 = X%) or do(restriction_2027 = X%).
  Step 3: Replay with same U_t under the intervention, compare to no-ban
          reference (also conditioned on same U_t).

Restriction magnitudes: 25%, 50%, 75%, 100% of China's export share.
Start years: 2026, 2027.
Duration: 3 years (then removed).
Metric: 'total_affected_yrs' = norm_yr - start_yr (full disruption window).
"""

import sys, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.minerals.schema import (
    BaselineConfig, DemandGrowthConfig, OutputsConfig,
    ParametersConfig, PolicyConfig, ScenarioConfig,
    ShockConfig, TimeConfig,
)
from src.minerals.simulate import run_scenario
from src.minerals.causal_engine import CausalInferenceEngine
from src.minerals.causal_inference import GraphiteSupplyChainDAG
from src.minerals.predictability import (
    _GRAPHITE_2022_PARAMS, _RARE_EARTHS_2010_PARAMS,
    _COBALT_2016_PARAMS, _LITHIUM_2022_PARAMS,
    _NICKEL_2022_PARAMS, _URANIUM_2022_PARAMS,
    _cepii_series,
)
from src.minerals.knowledge_graph import build_critical_minerals_kg

_KG = build_critical_minerals_kg(data_dir="data/canonical")

def _china_share(mineral: str, year: int = 2022) -> float:
    """Return China's effective processing share from the KG; 0.20 fallback for uranium."""
    r = _KG.effective_control_at("China", mineral, year)
    return r["effective_share"] if r["effective_share"] is not None else 0.20

BASELINE       = BaselineConfig(P_ref=1.0, P0=1.0, K0=108.695652, I0=20.0, D0=100.0)
EULER_SAFETY   = 0.9
NORM_THRESHOLD = 0.10
RESTRICTION_DURATION = 3
PROJECTION_END = 2035


def _stable_alpha_P(alpha_P, eta_D):
    limit = EULER_SAFETY / max(abs(eta_D), 1e-6)
    return min(alpha_P, limit)


# ── Mineral registry ──────────────────────────────────────────────────────────
# Each entry: (params, historical_shocks, history_start, cepii_path, cepii_exporter, extra_params)
# historical_shocks = the actual shocks that occurred 2019-2024 for this mineral
# these form the starting condition for the L3 abduction

_MINERALS = {
    "graphite": dict(
        params=_GRAPHITE_2022_PARAMS,
        history_start=2019,
        cepii_path="data/canonical/cepii_graphite.csv",
        cepii_exporter="China",
        base_year=2021,
        extra=dict(substitution_elasticity=0.8, substitution_cap=0.6),
        historical_shocks=[
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.10),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.30),
            ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.05),
            ShockConfig(type="stockpile_release",  start_year=2023, end_year=2023, magnitude=20.0),
            ShockConfig(type="export_restriction", start_year=2023, end_year=2024, magnitude=0.35),
        ],
        china_processing_share=_china_share("graphite"),
        us_import_reliance=1.00,
    ),
    "cobalt": dict(
        params=_COBALT_2016_PARAMS,
        history_start=2013,
        cepii_path="data/canonical/cepii_cobalt.csv",
        cepii_exporter="Dem. Rep. Congo",
        base_year=2015,
        extra=dict(substitution_elasticity=0.5, substitution_cap=0.4),
        historical_shocks=[
            ShockConfig(type="demand_surge", start_year=2016, end_year=2018, magnitude=0.25),
            ShockConfig(type="demand_surge", start_year=2019, end_year=2019, magnitude=-0.30),
        ],
        china_processing_share=_china_share("cobalt"),
        us_import_reliance=0.76,
    ),
    "lithium": dict(
        params=_LITHIUM_2022_PARAMS,
        history_start=2019,
        cepii_path="data/canonical/cepii_lithium.csv",
        cepii_exporter="Chile",
        base_year=2021,
        extra=dict(substitution_elasticity=0.6, substitution_cap=0.5,
                   fringe_capacity_share=0.4, fringe_entry_price=1.1),
        historical_shocks=[
            ShockConfig(type="demand_surge", start_year=2022, end_year=2022, magnitude=0.30),
        ],
        china_processing_share=_china_share("lithium"),
        us_import_reliance=0.50,
    ),
    "nickel": dict(
        params=_NICKEL_2022_PARAMS,
        history_start=2017,
        cepii_path="data/canonical/cepii_nickel.csv",
        cepii_exporter="Indonesia",
        base_year=2019,
        extra=dict(substitution_elasticity=0.5, substitution_cap=0.4,
                   fringe_capacity_share=0.45, fringe_entry_price=1.15),
        historical_shocks=[
            ShockConfig(type="export_restriction", start_year=2020, end_year=2022, magnitude=0.20),
            ShockConfig(type="demand_surge",       start_year=2021, end_year=2021, magnitude=0.20),
            ShockConfig(type="demand_surge",       start_year=2022, end_year=2022, magnitude=0.25),
            ShockConfig(type="stockpile_release",  start_year=2023, end_year=2023, magnitude=25.0),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2023, magnitude=-0.20),
            ShockConfig(type="demand_surge",       start_year=2024, end_year=2024, magnitude=-0.15),
        ],
        china_processing_share=_china_share("nickel"),
        us_import_reliance=0.40,
    ),
    "rare_earths": dict(
        params=_RARE_EARTHS_2010_PARAMS,
        history_start=2005,
        cepii_path="data/canonical/cepii_rare_earths.csv",
        cepii_exporter="China",
        base_year=2008,
        extra=dict(substitution_elasticity=0.5, substitution_cap=0.4),
        historical_shocks=[
            ShockConfig(type="export_restriction", start_year=2010, end_year=2010, magnitude=0.25),
            ShockConfig(type="export_restriction", start_year=2011, end_year=2011, magnitude=0.40),
            ShockConfig(type="export_restriction", start_year=2012, end_year=2013, magnitude=0.20),
            ShockConfig(type="demand_surge",       start_year=2013, end_year=2013, magnitude=-0.15),
            ShockConfig(type="demand_surge",       start_year=2014, end_year=2014, magnitude=-0.20),
        ],
        china_processing_share=_china_share("rare_earths"),
        us_import_reliance=0.14,
        cepii_note="CEPII data ends ~2016; U_t≈0 for 2017+. L3≈L2 for this mineral.",
    ),
    "uranium": dict(
        params=_URANIUM_2022_PARAMS,
        history_start=2019,
        cepii_path=None,
        cepii_exporter=None,
        base_year=2020,
        extra={},
        historical_shocks=[
            ShockConfig(type="export_restriction", start_year=2022, end_year=2024, magnitude=0.25),
            ShockConfig(type="demand_surge",       start_year=2023, end_year=2025, magnitude=0.15),
        ],
        china_processing_share=_china_share("uranium"),  # Russia SWU proxy; no KG data → 0.20 fallback
        us_import_reliance=0.95,
        cepii_note="No CEPII data for uranium. L3 degenerates to L2.",
    ),
}


def _build_cfg(mineral, m_cfg, extra_shocks, end_year, label):
    """Build scenario config for one mineral with optional forward ban shocks."""
    p = m_cfg["params"]
    alpha_P = _stable_alpha_P(p["alpha_P"], p["eta_D"])
    kw = dict(
        eps=1e-9, u0=0.92, beta_u=0.10, u_min=0.70, u_max=1.00,
        tau_K=p["tau_K"], eta_K=0.40, retire_rate=0.0, eta_D=p["eta_D"],
        demand_growth=DemandGrowthConfig(type="constant", g=p["g"]),
        alpha_P=alpha_P,
        cover_star=0.20, lambda_cover=0.60, sigma_P=0.0,
    )
    kw.update(m_cfg["extra"])
    return ScenarioConfig(
        name=f"{mineral}_{label}",
        commodity="graphite",
        seed=42,
        time=TimeConfig(dt=1.0, start_year=m_cfg["history_start"], end_year=end_year),
        baseline=BASELINE,
        parameters=ParametersConfig(**kw),
        policy=PolicyConfig(),
        shocks=list(m_cfg["historical_shocks"]) + list(extra_shocks),
        outputs=OutputsConfig(metrics=["avg_price"]),
    )


def _load_cepii(m_cfg):
    """Load CEPII prices dict {year: price}. Returns {} if unavailable."""
    if not m_cfg.get("cepii_path"):
        return {}
    try:
        cepii = _cepii_series(m_cfg["cepii_path"], m_cfg["cepii_exporter"])
        return {int(yr): float(cepii.loc[yr, "implied_price"])
                for yr in cepii.index if not __import__("math").isnan(cepii.loc[yr, "implied_price"])}
    except Exception as e:
        print(f"  [CEPII load failed: {e}]")
        return {}


def _norm_year(ban_df, noban_df, base_year, search_from):
    """First year >= search_from where ban price index ≈ no-ban price index (within 10%)."""
    ban   = ban_df.set_index("year")["P"]
    noban = noban_df.set_index("year")["P"]
    ban_base   = ban.loc[base_year]
    noban_base = noban.loc[base_year]
    for yr in sorted(y for y in ban.index if y >= search_from and y in noban.index):
        if abs(ban.loc[yr] / ban_base - noban.loc[yr] / noban_base) < NORM_THRESHOLD:
            return yr
    return None


def run_l3_ban(mineral, m_cfg, magnitude, start_yr):
    """
    L3 forward ban analysis for one mineral × one (magnitude, start_yr) scenario.

    Returns dict with peak, norm_yr, total_affected_yrs.
    """
    end_yr     = start_yr + RESTRICTION_DURATION - 1
    base_year  = m_cfg["base_year"]
    dag        = GraphiteSupplyChainDAG()
    observed   = _load_cepii(m_cfg)

    # ── Step 1: Abduct U_t from ACTUAL historical trajectory ──────────────────
    # Use historical-only config (no forward ban) for abduction
    hist_cfg   = _build_cfg(mineral, m_cfg, [], PROJECTION_END, "hist_only")
    hist_engine = CausalInferenceEngine(dag=dag, cfg=hist_cfg, seed=42)
    abduction  = hist_engine.counterfactual_l3(
        observed_prices=observed,
        do_overrides={},
        cfg=hist_cfg,
        base_year=base_year,
    )
    log_residuals = abduction.abduction.inferred_noise

    # ── Step 2/3: Action + Prediction ─────────────────────────────────────────
    ban_shocks = [ShockConfig(
        type="export_restriction",
        start_year=start_yr, end_year=end_yr,
        magnitude=magnitude,
    )]

    # "With ban" — historical U_t + forward ban
    with_ban_cfg = _build_cfg(mineral, m_cfg, ban_shocks, PROJECTION_END, f"ban{int(magnitude*100)}")
    with_ban_engine = CausalInferenceEngine(dag=dag, cfg=with_ban_cfg, seed=42)
    with_ban_result = with_ban_engine.counterfactual_l3(
        observed_prices={},
        do_overrides={},
        cfg=with_ban_cfg,
        base_year=base_year,
        precomputed_log_residuals=log_residuals,
    )
    with_ban_df = with_ban_result.counterfactual_trajectory

    # "No ban" reference — historical U_t only, no forward ban
    no_ban_cfg    = _build_cfg(mineral, m_cfg, [], PROJECTION_END, "no_ban")
    no_ban_engine = CausalInferenceEngine(dag=dag, cfg=no_ban_cfg, seed=42)
    no_ban_result = no_ban_engine.counterfactual_l3(
        observed_prices={},
        do_overrides={},
        cfg=no_ban_cfg,
        base_year=base_year,
        precomputed_log_residuals=log_residuals,
    )
    no_ban_df = no_ban_result.counterfactual_trajectory

    # ── Metrics ───────────────────────────────────────────────────────────────
    ban_sc   = with_ban_df.set_index("year")["P"]
    ban_base = ban_sc.loc[base_year]

    # Peak from ban start onward
    window   = {yr: ban_sc.loc[yr] / ban_base for yr in ban_sc.index if yr >= start_yr}
    peak_yr  = max(window, key=window.get)
    peak_idx = window[peak_yr]

    # Normalisation relative to no-ban L3 reference (not clean baseline)
    norm_yr = _norm_year(with_ban_df, no_ban_df, base_year, end_yr + 1)
    total_affected = (norm_yr - start_yr) if norm_yr else None

    return {
        "mineral":              mineral,
        "start_yr":             start_yr,
        "end_yr":               end_yr,
        "magnitude":            magnitude,
        "peak_idx":             peak_idx,
        "peak_yr":              peak_yr,
        "norm_yr":              norm_yr,
        "lag_from_end":         (norm_yr - end_yr) if norm_yr else None,
        "total_affected_yrs":   total_affected,
        "has_cepii":            bool(observed),
        "tau_K":                m_cfg["params"]["tau_K"],
        "china_share":          m_cfg["china_processing_share"],
        "us_reliance":          m_cfg["us_import_reliance"],
    }


def print_mineral_section(mineral, m_cfg, rows):
    has_cepii = rows[0]["has_cepii"] if rows else False
    note      = m_cfg.get("cepii_note", "")
    p         = m_cfg["params"]
    print(f"\n{'═' * 78}")
    print(f"MINERAL: {mineral.upper()}")
    print(f"  China processing share: {m_cfg['china_processing_share']:.0%}   "
          f"US import reliance: {m_cfg['us_import_reliance']:.0%}")
    print(f"  τ_K = {p['tau_K']:.2f}yr   "
          f"α_P = {_stable_alpha_P(p['alpha_P'], p['eta_D']):.3f}   "
          f"η_D = {p['eta_D']:.3f}")
    if has_cepii:
        print(f"  L3 mode: abducted U_t from CEPII history → conditioned on actual 2024 state")
    else:
        print(f"  L3 mode: DEGENERATES TO L2 (no CEPII data)")
    if note:
        print(f"  Note: {note}")

    for start_yr in [2026, 2027]:
        end_yr = start_yr + RESTRICTION_DURATION - 1
        mineral_rows = [r for r in rows if r["start_yr"] == start_yr]
        print(f"\n  Ban starts {start_yr}, lifted {end_yr} (3yr restriction)")
        print(f"  {'Ban %':>6}  {'Peak ×':>7}  {'Peak yr':>8}  "
              f"{'Norm yr':>9}  {'Lag from end':>13}  {'Total yrs affected':>19}")
        print("  " + "─" * 6 + "  " + "─" * 7 + "  " + "─" * 8 + "  " +
              "─" * 9 + "  " + "─" * 13 + "  " + "─" * 19)
        for r in mineral_rows:
            ban_pct   = f"{int(r['magnitude']*100)}%"
            peak      = f"{r['peak_idx']:.2f}×"
            peak_y    = str(r["peak_yr"])
            norm_y    = str(r["norm_yr"]) if r["norm_yr"] else "never"
            lag_str   = f"+{r['lag_from_end']}yr" if r["lag_from_end"] is not None else "never"
            total_str = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                         else f">{PROJECTION_END - start_yr} yrs")
            print(f"  {ban_pct:>6}  {peak:>7}  {peak_y:>8}  "
                  f"{norm_y:>9}  {lag_str:>13}  {total_str:>19}")


def print_cross_mineral(all_rows):
    print("\n" + "=" * 78)
    print("CROSS-MINERAL SUMMARY — L3 CONDITIONED, 50% BAN STARTING 2026")
    print("How long does the US remain affected? (conditioned on actual 2024 state)")
    print("=" * 78)
    print(f"  {'Mineral':<14} {'τ_K':>6}  {'US rely':>8}  {'Peak ×':>7}  "
          f"{'Norm yr':>9}  {'Yrs affected':>13}  {'L3/L2':>6}")
    print("  " + "─" * 78)
    for mineral in _MINERALS:
        r = next((x for x in all_rows
                  if x["mineral"] == mineral and x["start_yr"] == 2026 and x["magnitude"] == 0.50), None)
        if not r:
            continue
        norm  = str(r["norm_yr"]) if r["norm_yr"] else "never"
        total = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                 else f">{PROJECTION_END - 2026} yrs")
        mode  = "L3" if r["has_cepii"] else "L2"
        print(f"  {mineral:<14} {r['tau_K']:>6.2f}  {r['us_reliance']:>8.0%}  "
              f"{r['peak_idx']:>7.2f}×  {norm:>9}  {total:>13}  {mode:>6}")
    print()

    print("=" * 78)
    print("FULL BAN (100%) STARTING 2026 — L3")
    print("=" * 78)
    print(f"  {'Mineral':<14} {'China share':>12}  {'Peak ×':>7}  "
          f"{'Norm yr':>9}  {'Yrs affected':>13}")
    print("  " + "─" * 65)
    for mineral in _MINERALS:
        r = next((x for x in all_rows
                  if x["mineral"] == mineral and x["start_yr"] == 2026 and x["magnitude"] == 1.00), None)
        if not r:
            continue
        norm  = str(r["norm_yr"]) if r["norm_yr"] else "never"
        total = (f"{r['total_affected_yrs']} yrs" if r["total_affected_yrs"] is not None
                 else f">{PROJECTION_END - 2026} yrs")
        print(f"  {mineral:<14} {r['china_share']:>12.0%}  {r['peak_idx']:>7.2f}×  "
              f"{norm:>9}  {total:>13}")

    print("\n" + "=" * 78)
    print("2026 vs 2027 START YEAR — 50% BAN, 3yr: does a 1yr delay change exposure?")
    print("=" * 78)
    print(f"  {'Mineral':<14}  {'2026 yrs affected':>20}  {'2027 yrs affected':>20}  {'Difference':>12}")
    print("  " + "─" * 72)
    for mineral in _MINERALS:
        r26 = next((x for x in all_rows if x["mineral"] == mineral
                    and x["start_yr"] == 2026 and x["magnitude"] == 0.50), None)
        r27 = next((x for x in all_rows if x["mineral"] == mineral
                    and x["start_yr"] == 2027 and x["magnitude"] == 0.50), None)
        if not r26 or not r27:
            continue
        t26 = r26["total_affected_yrs"]
        t27 = r27["total_affected_yrs"]
        s26 = (f"{t26} yrs" if t26 is not None else f">{PROJECTION_END - 2026} yrs")
        s27 = (f"{t27} yrs" if t27 is not None else f">{PROJECTION_END - 2027} yrs")
        diff = f"{t27 - t26:+d} yrs" if (t26 is not None and t27 is not None) else "—"
        print(f"  {mineral:<14}  {s26:>20}  {s27:>20}  {diff:>12}")


def main():
    print("=" * 78)
    print("L3 BAN IMPACT GRID — Pearl Layer 3 forward projection")
    print("Conditioned on actual 2024 market state via abducted CEPII residuals")
    print("=" * 78)
    print(f"""
Key difference from L2 (forward_scenario_2025.py):
  L2: starts from clean 2024 baseline (U_t = 0 everywhere)
  L3: abducts U_t from actual 2019-2024 CEPII trajectory, then projects forward
      This captures the CARRY-FORWARD of recent market dynamics into the ban scenario.

Restriction magnitudes: 25%, 50%, 75%, 100% of China's export share
Start years: 2026, 2027
Duration: {RESTRICTION_DURATION} years (then removed)
Projection end: {PROJECTION_END}
Normalisation: price index within 10% of L3 no-ban reference trajectory
""")

    all_rows = []
    magnitudes  = [0.25, 0.50, 0.75, 1.00]
    start_years = [2026, 2027]

    for mineral, m_cfg in _MINERALS.items():
        print(f"\nRunning {mineral}...", flush=True)
        rows = []
        for start_yr in start_years:
            for mag in magnitudes:
                r = run_l3_ban(mineral, m_cfg, mag, start_yr)
                rows.append(r)
                all_rows.append(r)
                print(f"  {mineral} {int(mag*100)}% ban start {start_yr}: "
                      f"peak {r['peak_idx']:.2f}×, "
                      f"yrs affected={'never' if r['total_affected_yrs'] is None else r['total_affected_yrs']}",
                      flush=True)
        print_mineral_section(mineral, m_cfg, rows)

    print_cross_mineral(all_rows)

    print("\n" + "=" * 78)
    print("L3 vs L2 COMPARISON NOTE")
    print("=" * 78)
    print("""
  L3 results differ from L2 (forward_scenario_2025.py) because they condition
  on the actual 2024 market state:

  Cobalt L3 < L2: cobalt is in deep 2024 oversupply (U_t = -2.128).
    L3 starts from this oversupply — the ban first brings the market back to
    balance before causing a spike. L2's ~10× peak is unrealistic; L3's peak
    reflects the actual starting point.

  Graphite L3 ≥ L2: export controls already depleted inventory through 2024.
    L3 starts from a tighter market. A 2026 ban compounds on existing inventory
    depletion, extending the affected period vs. a clean-start L2.

  Lithium/Nickel: moderate differences reflecting recent demand/supply dynamics.

  Rare earths: CEPII data ends ~2016. L3 ≈ L2 for this mineral (stale U_t).
  Uranium: no CEPII. L3 = L2 (degenerate case).
""")


if __name__ == "__main__":
    main()
