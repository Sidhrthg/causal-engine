#!/usr/bin/env python3
"""
End-to-end demo: Pearl's three layers (L1 Association, L2 Intervention, L3 Counterfactual).

Run from project root (with venv activated):
  source .venv/bin/activate
  python scripts/run_three_layers_demo.py

Uses: graphite_baseline.yaml, graphite_export_restriction.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.minerals.schema import load_scenario
from src.minerals.simulate import run_scenario
from src.minerals.pearl_layers import (
    observational_summary,
    interventional_identifiability,
    three_layers_summary,
)
from src.minerals.causal_inference import GraphiteSupplyChainDAG
from src.minerals.causal_engine import CausalInferenceEngine


def main() -> None:
    print("=" * 70)
    print("THREE-LAYER CAUSAL ENGINE — END-TO-END DEMO")
    print("=" * 70)
    print(three_layers_summary())
    print()

    # -------------------------------------------------------------------------
    # Setup: load scenarios and run baseline + policy
    # -------------------------------------------------------------------------
    path_baseline = PROJECT_ROOT / "scenarios" / "graphite_baseline.yaml"
    path_policy = PROJECT_ROOT / "scenarios" / "graphite_export_restriction.yaml"
    if not path_baseline.exists() or not path_policy.exists():
        print("Missing scenario files. Need graphite_baseline.yaml and graphite_export_restriction.yaml")
        sys.exit(1)

    cfg_baseline = load_scenario(str(path_baseline))
    cfg_policy = load_scenario(str(path_policy))

    print("Running baseline scenario ...")
    df_baseline, metrics_baseline = run_scenario(cfg_baseline)
    print("Running export-restriction scenario ...")
    df_policy, metrics_policy = run_scenario(cfg_policy)
    print()

    # =========================================================================
    # LAYER 1 — Association (Seeing): P(Y|X)
    # =========================================================================
    print("=" * 70)
    print("LAYER 1 — ASSOCIATION (Seeing): P(Y|X)")
    print("  Question: 'What if I see X?' — observational summaries only, no do(·)")
    print("=" * 70)

    # Observational summary of Price and shortage from baseline run
    summary_P = observational_summary(df_baseline, "P")
    summary_shortage = observational_summary(df_baseline, "shortage")
    print("\n  Observational summary (baseline run):")
    print(f"    P (price):       mean = {summary_P['mean'].iloc[0]:.4f}, std = {summary_P['std'].iloc[0]:.4f}")
    print(f"    shortage:       mean = {summary_shortage['mean'].iloc[0]:.4f}, std = {summary_shortage['std'].iloc[0]:.4f}")

    # Correlation matrix (L1 only — association, not causation)
    dag = GraphiteSupplyChainDAG()
    engine = CausalInferenceEngine(dag=dag, cfg=cfg_baseline, seed=42)
    corr = engine.correlate(df_baseline, variables=["P", "shortage", "cover", "D"], method="pearson")
    print("\n  Correlation matrix (Layer 1):")
    print(f"    Variables: {corr.variables}")
    for i, vi in enumerate(corr.variables):
        for j, vj in enumerate(corr.variables):
            if i < j:
                print(f"      {vi} vs {vj}: r = {corr.pearson[i, j]:.3f}, p = {corr.p_values[i, j]:.3f}")
    print()

    # =========================================================================
    # LAYER 2 — Intervention (Doing): P(Y|do(X))
    # =========================================================================
    print("=" * 70)
    print("LAYER 2 — INTERVENTION (Doing): P(Y|do(X))")
    print("  Question: 'What if I do X?' — causal effect via do-calculus + scenario runs")
    print("=" * 70)

    # Identifiability: is P(Price | do(ExportPolicy)) identifiable?
    ident = interventional_identifiability("ExportPolicy", "Price", dag=dag)
    print(f"\n  Identifiability: P(Price | do(ExportPolicy))")
    print(f"    Identifiable: {ident.identifiable}")
    print(f"    Strategy: {ident.strategy}")
    print(f"    Adjustment set: {ident.adjustment_set}")

    # Causal effect = compare baseline vs export-restriction run (do(policy))
    print("\n  Causal effect (baseline vs export restriction):")
    print(f"    Baseline total_shortage: {metrics_baseline['total_shortage']:.4f}")
    print(f"    Policy   total_shortage: {metrics_policy['total_shortage']:.4f}")
    print(f"    ATE (total_shortage):    {metrics_policy['total_shortage'] - metrics_baseline['total_shortage']:.4f}")
    print(f"    Baseline avg_price:      {metrics_baseline['avg_price']:.4f}")
    print(f"    Policy   avg_price:      {metrics_policy['avg_price']:.4f}")
    print(f"    ATE (avg_price):         {metrics_policy['avg_price'] - metrics_baseline['avg_price']:.4f}")
    print()

    # =========================================================================
    # LAYER 3 — Counterfactual (Imagining): P(Y_x | X', Y')
    # =========================================================================
    print("=" * 70)
    print("LAYER 3 — COUNTERFACTUAL (Imagining): P(Y_x | X', Y')")
    print("  Question: 'What if X had been x, given what I saw?'")
    print("  Example: We SAW the export restriction (factual). What would have happened")
    print("           if we had NOT had the restriction (counterfactual)?")
    print("=" * 70)

    # Use the policy run as "factual" trajectory; counterfactual = no restriction in shock years
    shock_years = [s.start_year for s in cfg_policy.shocks] + [s.end_year for s in cfg_policy.shocks]
    shock_years = sorted(set(y for s in cfg_policy.shocks for y in range(s.start_year, s.end_year + 1)))
    do_overrides = {y: {"export_restriction": 0.0} for y in shock_years}

    print(f"\n  Factual: export_restriction scenario (shock in {shock_years})")
    print(f"  Counterfactual: same run but do(export_restriction=0) in those years")
    print("  Running Pearl 3-step (abduction → action → prediction) ...")

    engine_policy = CausalInferenceEngine(dag=dag, cfg=cfg_policy, seed=42)
    cf_result = engine_policy.counterfactual(
        observed_data=df_policy,
        do_overrides=do_overrides,
        cfg=cfg_policy,
    )

    print("\n  Counterfactual result (summary):")
    for k, v in cf_result.summary.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Per-year effect (counterfactual − factual) for Price and shortage:")
    eff = cf_result.effect
    for col in ["P", "shortage"]:
        if col in eff.columns:
            print(f"    {col}: mean delta = {eff[col].mean():.4f}, max |delta| = {eff[col].abs().max():.4f}")
    print()

    print("=" * 70)
    print("DONE — Three layers demonstrated.")
    print("=" * 70)


if __name__ == "__main__":
    main()
