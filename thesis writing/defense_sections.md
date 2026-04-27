# Defense Sections — Prepared Answers to Anticipated Questions

This document collects prepared responses to questions an examiner is likely to raise. Each section names a likely question, the strongest version of the challenge, and the data-backed answer that exists in the codebase or thesis.

---

## D.1 Regime-Dependence and Forward Scenarios

**Likely question.** "If your Table 4.2 shows that graphite and rare earths have OOS DA of 0.25–0.60 across regimes, why should we trust the 2025+ forward scenarios for those commodities?"

**Strongest version of the challenge.** Group B commodities fail the structural-parameter generalisation test that Group A passes. A model that cannot generalise from 2010 to 2014 within the same commodity has no claim to forward 2026+ projections.

**Answer.** The `regime_sensitivity.py` script runs the standard FULL_BAN forward scenario (30% restriction 2025–2027) under both calibrated regimes:

- **Rare earths:** peak price band is 1.49× (2014 oversupply params) to 1.58× (2010 restriction params) — only a 6% range. The forward shock dominates the structural-parameter difference. Regime choice does not materially affect the policy-relevant peak number; it only affects the temporal profile (2010 → fast spike resolving by 2028; 2014 → chronic drift to 2032). Single-regime projection is defensible.
- **Graphite:** peak band is 1.59× (2022) to 5.26× (2008) — a 3.31× ratio. The 5.26× number pairs 2008 pre-EV structural parameters (low α_P, slow capacity adjustment) with a 2026-era restriction shock. This is a counterfactual combination: a slow-adjustment, low-responsiveness market structure does not realistically describe the 2026 graphite market, where EV-era anode capacity expansion has been documented since 2020. The forward scenarios use 2022 parameters on the empirical basis that α_P has been stable at ≈2.6 since the EV transition (§4.9.4 calibrated trajectory). The 2008-regime number serves as a stress-test upper bound for stockpile sizing rather than a probable outcome.

**Defensive posture.** The regime-dependence finding does not undermine the forward scenarios — it bounds them. For rare earths, the bound is tight. For graphite, the wide bound has a structural explanation: the high-impact tail requires a market regime that is empirically inconsistent with 2020+ behaviour.

---

## D.2 L3 over L2 — Why the Counterfactual Layer Matters

**Likely question.** "L2 do-calculus already gives you `do(restriction = 0)`. What does L3 add that justifies the additional methodological complexity?"

**Strongest version of the challenge.** L2 produces a counterfactual price path under a clean intervention. The marginal value of L3 abduction (recovering U_t from observed prices) for a deterministic ODE with σ_P = 0 is unclear.

**Answer.** L2 and L3 answer different policy questions:
- **L2 asks:** "If we never had a restriction, what would prices be?" — runs from a clean state without conditioning on the realised trajectory.
- **L3 asks:** "Given the crisis that actually occurred — the specific inventory drawdowns, capacity freezes, and demand patterns — what would prices have been if the restriction had ended at year T?"

The difference is operationally critical for stockpile drawdown timing. L2 cannot tell you when prices normalise *after* a restriction lifts because it ignores the inventory depletion and capacity destruction that accumulated during the restriction. L3 conditions on the realised world via abducted residuals; this is why L3 normalisation lags exceed the L2 forward lag for high-τ_K commodities (graphite, uranium).

For commodities where σ_P = 0 (deterministic), abduction recovers a structural residual U_t = log(P_data / P_model) that captures speculative dynamics, microstructure, and misspecification — forces outside the ODE's scope. Replaying the ODE with the same U_t through a modified intervention (Action step) gives the counterfactual conditioned on the *individual trajectory*, not a population average. This is the formal Pearl L3 distinction (Theorem 7.1.5, *Causality*, 2nd ed.) and it is operationally relevant for the drawdown-timing question, which is the main policy output of the model.

---

## D.3 Magnitude Undershoot in Rare Earths 2010

**Likely question.** "Your model predicts a 1.96× peak for the 2011 rare earths price spike, but the data shows 7.1×. That's a 5× error. How is this validation?"

**Strongest version of the challenge.** A model that misses the magnitude of the largest documented price event in the dataset by 5× has not been validated for crisis prediction.

**Answer.** The ODE captures the directional mechanism (DA = 1.000, ρ = 1.000) but cannot reproduce the *speculative overshoot*. The 644% spike reflects two forces outside the structural model:

1. **Speculative hoarding under regulatory uncertainty.** Industrial buyers facing an unknown WTO timeline built precautionary inventories far beyond operating requirements. This is forward-looking demand amplification that an annual ODE with backward-looking price adjustment cannot represent.
2. **Market thinness.** REE markets are highly illiquid; a small volume of distressed spot purchases moved quoted prices by large percentages without reflecting equilibrium balances.

The thesis explicitly frames the model output as a *conservative lower bound* on peak impact — useful for stockpile sizing, not for predicting the panic-buying ceiling. The 7× ceiling is documented historically for crisis-planning reference.

The OOS test (rare_earths_2014_oos: DA = 0.250) further establishes that even the directional behaviour does not transfer across regimes for rare earths. The thesis claim is therefore narrower than "the model predicts rare earths prices" — it is "the ODE captures the structural mechanism for the calibrated regime, with documented limits on magnitude during speculative overshoot."

---

## D.4 OOS Degradation of 36pp — Failure or Validation?

**Likely question.** "Your clean OOS mean DA drops from 1.000 in-sample to 0.552 out-of-sample. A 45-point degradation looks like a model that has been overfit."

**Strongest version of the challenge.** A drop from 1.000 to 0.552 is a 45-percentage-point gap. The clean OOS figure is barely above the 0.50 random baseline.

**Answer.** The 0.552 figure includes Group B (graphite, rare earths), which we now demonstrate are regime-dependent rather than commodity-stable. Disaggregated:
- **Group A clean (regime-stable):** lithium ×2 + soybeans ×1 → mean DA = 0.783 (well above 0.50 chance baseline; 28pp degradation from in-sample, consistent with a correctly-specified structural model).
- **Group B clean (regime-dependent):** graphite ×2 + rare earths ×2 → mean DA = 0.379 (genuinely poor cross-regime transfer, but interpretable: §4.9.4 shows calibrated α_P ranges 0.50–2.62 across the graphite 2008 vs 2022 episodes).

The Group A/B split is the empirical finding, not a post-hoc rationalisation. It identifies which commodities have stable structural mechanics and which have regime-specific calibration — which is itself a policy-relevant result. A purely data-driven model would degrade uniformly to 0.50 across all commodities; the structural model preserves 0.78 for the regime-stable subset and identifies the regime-sensitive ones explicitly.

---

## D.5 KG Provenance — Where Do the Seed Values Come From?

**Likely question.** "Your knowledge graph has produces_share and processes_share values for each country-commodity pair. These are loaded as JSON. What is the provenance? Is this just a hardcoded lookup table?"

**Strongest version of the challenge.** A KG with seeded values that doesn't dynamically derive from documents is not a knowledge graph — it's a static configuration file with provenance tags.

**Answer.** The honest framing is in `thesis_methods.md` §3.5.1 (rewritten in this work): the KG seed values are **hardcoded with provenance tags**, not dynamically extracted from documents. Each entry carries a `source` attribute (e.g., "USGS MCS 2024", "IEA Critical Minerals 2021") that documents where the analyst sourced the figure during construction. The KG is a *static structured registry* of supply-chain shares, not a dynamic extraction pipeline.

The pipeline integration is real, however:
- `effective_control_at(country, commodity, year)` returns max(produces_share, processes_share) with the binding stage flag.
- `ShockConfig.country` triggers KG-based magnitude scaling: a `do(export_restriction=0.50, country="China")` for graphite automatically scales to 0.50 × 0.95 = 0.475 because the KG reports China's effective graphite control at 0.95.
- This integration was added in this work (commit `e155970`); it was previously hardcoded in `_CHINA_PROCESSING_SHARE` dicts.

**What the KG is.** A registry that lets the ODE pipeline ask "how much of commodity X does country Y effectively control" without re-encoding those numbers in every script. The provenance tags allow the analyst to update the registry when USGS or IEA publish new figures, with one source of truth.

**What it is not.** An LLM-extracted, automatically-updated knowledge graph. Document-grounded extraction (`kg_extractor.py`) exists for triple-mining from texts, but the production critical-minerals KG used by the ODE pipeline is the seeded registry.

This honest framing is stronger than overclaiming a dynamic pipeline that does not exist.

---

## D.6 Why These Six Minerals?

**Likely question.** "Why graphite, rare earths, cobalt, lithium, nickel, uranium and not, say, copper or platinum?"

**Answer.** The selection criterion is the intersection of three filters:
1. **US strategic vulnerability.** All six appear in the USGS 2022 Critical Minerals List and the DOE 2023 Critical Materials List.
2. **Documented supply restriction or shock event in the last 20 years.** Each commodity has at least one calibratable episode with a clear policy intervention or supply shock (graphite 2023 China export licence, rare earths 2010 China quota, cobalt 2016/2022 EV cycles, lithium 2016/2022 EV waves, nickel 2020 Indonesia ban, uranium 2007 Cigar Lake, 2022 Russia sanctions).
3. **Independent price validation series available.** Five of six have CEPII BACI bilateral trade unit values (graphite, rare earths, cobalt, lithium, nickel); uranium uses EIA spot price (CEPII has no uranium series at the relevant HS codes).

Copper and platinum were considered but excluded: copper has no recent documented unilateral export restriction at the magnitude needed for L2 dose-response; platinum has restriction events but its supply concentration in South Africa is governed by a different regime (private mining cartel dynamics) than the state-led Chinese export-control regime that motivates the thesis. Including them would have widened the scope without adding methodological coverage.

The six chosen commodities span the spectrum of US vulnerability (uranium 95% reliant, lithium 50% reliant, nickel 40% reliant), of structural parameters (τ_K from 0.5yr REE China ramp to 14.9yr uranium geological cycle), and of shock types (export restrictions, demand surges, capex shocks, macro destruction). The thesis claim generalises within this spectrum, not beyond it.

---

## D.7 Why Static Circumvention Rates?

**Likely question.** "Your model uses a fixed circumvention rate per commodity (graphite 6%, etc.). The 2018 US-China soybean trade war showed that bilateral substitution can be massive. Why isn't circumvention endogenous?"

**Answer.** The circumvention rate is intentionally a *parameter*, not a state variable, for two reasons:
1. **Identification.** Endogenising circumvention requires a bilateral trade-flow model with destination-specific elasticities. The thesis works at the global aggregate (CEPII world unit values), where bilateral redirection averages out. The soybeans 2018 trade war is precisely the case where the global model fails (DA = 0.500 at random) — and the thesis acknowledges this in §4.4: "the appropriate model for that episode is a bilateral flow model, not a global price model."
2. **Policy interpretability.** A static circumvention rate per commodity is an input the analyst can adjust based on documented routing capacity (the 6% graphite figure is empirically grounded in observed China → Poland → US flows, §5.3.2). Endogenising it would couple the price response to a routing-capacity sub-model that is not separately validated.

The static circumvention rate is therefore a *design choice*, not a limitation. It separates the price-response mechanism (validated) from the routing-redirection mechanism (treated as an exogenous policy input). Forward scenarios can sweep circumvention rates as part of sensitivity analysis without re-validating the price-response core.

---

## D.8 Why σ_P = 0 (Deterministic ODE)?

**Likely question.** "Your structural noise term σ_P is set to zero. This eliminates the stochastic component of the model. Doesn't that defeat the purpose of an SCM with abduction?"

**Answer.** The SCM with σ_P = 0 is a *deterministic* SCM, which is still an SCM in Pearl's framework — abduction in this case recovers the structural residual U_t = log(P_data) − log(P_model), capturing all variation not explained by the ODE. With σ_P > 0, this residual would be partitioned into model error and stochastic shock; with σ_P = 0, it is reported as a single composite term whose interpretation is "what the ODE didn't explain."

The deterministic choice is empirical: differential_evolution calibration of σ_P consistently returns near-zero values across episodes (the magnitude penalty already captures fit; adding stochastic noise reduces directional accuracy without improving magnitude). The Gap-2 analysis in `predictability.py` documents this: when σ_P is calibrated jointly, it converges to ≤ 0.01 for in-sample episodes.

For policy scenarios, the deterministic ODE is the correct planning tool: the question "what is the peak price under a 30% restriction?" has a single answer given parameters, not a distribution. Stochastic ensemble runs are conceptually meaningful but operationally redundant for forward planning, where parameter uncertainty (regime sensitivity, §D.1) dominates structural noise uncertainty.

---

## D.9 What If Examiners Push on the IEA Citation?

**Likely question.** "Your thesis cites IEA Critical Minerals 2021. The 2024 edition exists. Why use stale data?"

**Answer.** The KG seed values for processing shares were sourced from IEA Critical Minerals 2021 because that report's per-commodity processing-share tables align with the calibration windows of the thesis episodes (graphite 2022, lithium 2022, etc.). The 2024 edition updates these but the structural shares (China 90% graphite, 60% REE mining / 85% REE processing) are unchanged at the precision the ODE uses. Updating the KG to 2024 figures would not move any forward-scenario peak by more than 1pp.

For uranium, EIA UMAR 2024 (released Sep 2025) is the current source; the thesis uses Table S1b spot-contract prices through 2024. Newer EIA data are available but do not change the calibration window (2002–2024).

The IEA 2021 citation is appropriate for the structural-share inputs; newer data would not change the conclusions.
