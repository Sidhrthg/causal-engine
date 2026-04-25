# Chapter 9: Forward Projections — Cross-Mineral Shock Scenarios 2025–2031

## 9.1 Introduction

The preceding four case study chapters established causal models for graphite, rare earth elements, lithium, cobalt, and nickel, validated against historical price episodes using CEPII bilateral trade data as an independent benchmark. This chapter turns from historical validation to prospective intervention analysis.

Pearl's causal hierarchy distinguishes sharply between observational prediction (L1), interventional prediction (L2), and counterfactual reasoning (L3). The forward projections in this chapter are L2 claims — do-calculus interventions of the form "what would prices be if a restriction of magnitude X were imposed from 2025?" L3 counterfactual reasoning (conditioning on a specific realised trajectory) requires an observed post-restriction trajectory for abduction; since the restrictions are hypothetical, L3 is inapplicable, and the L2 framework is the appropriate tool.

This is not a weakness of the method — it is the correct epistemic positioning. An L2 projection states what the structural model predicts under a specified intervention, given the calibrated parameters from historical episodes. It makes no claim to predict the precise magnitude or timing of speculative financial dynamics, ASM supply volatility, or technology adaptation responses that would appear as U_t residuals in an ex-post L3 analysis. The appropriate use of these projections is to establish structural price bounds — upper and lower envelopes within which the actual market would move — rather than point forecasts.

Three classes of forward scenarios are presented:
1. **Cross-mineral full-ban comparison**: a uniform 30% restriction for three years applied to each mineral independently, to permit structural comparison
2. **Severity gradient**: mild (30%, 2yr), full (30%, 3yr), and severe (50%, 4yr) restrictions for each mineral
3. **Structural vulnerability ranking**: integration of τ_K, peak magnitude, normalisation lag, and US import reliance into a composite risk score

All projections start from the 2024 calibrated state and run through 2031. Background demand growth at the calibrated episode rate (g) continues throughout. Euler-stabilised α_P (capped at 0.9/|η_D|) is used to prevent numerical blow-up in long forward runs.

---

## 9.2 Baseline Trajectories

Before examining restriction scenarios, the baseline trajectory — no new restriction, calibrated demand growth continuing — provides the structural reference for each mineral.

| Mineral | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | Growth driver |
|---------|------|------|------|------|------|------|------|------|--------------|
| graphite | 1.000 | 1.000 | 0.946 | 0.916 | 0.884 | 0.856 | 0.829 | 0.803 | Negative: supply enters, demand growth below supply ramp |
| rare earths | 1.000 | 1.000 | 1.138 | 1.249 | 1.158 | 1.324 | 1.441 | 1.504 | EV magnet demand growth |
| cobalt | 1.000 | 1.000 | 1.570 | 2.082 | 2.572 | 3.179 | 3.924 | 4.795 | Steep: high g, τ_K = 5.75yr |
| lithium | 1.000 | 1.000 | 1.326 | 1.831 | 1.707 | 1.673 | 2.119 | 2.605 | EV demand; near-zero demand elasticity |
| nickel | 1.000 | 1.000 | 1.493 | 1.496 | 2.012 | 1.721 | 2.731 | 2.606 | EV + stainless demand |
| uranium | 1.000 | 1.000 | 1.056 | 1.187 | 1.424 | 1.725 | 2.101 | 2.560 | Nuclear renaissance, slow supply |

Three structural observations emerge from the baseline:

**Graphite is the only mineral with a declining baseline.** The post-2022 export licence overhang combined with background supply ramp produces a slightly declining price trajectory even without new restrictions. This reflects the model's calibration from the 2022 episode where fringe supply and demand substitution (silicon anodes, LFP) are already entering. A new restriction would impose additional tightening on top of a structurally declining baseline.

**Cobalt has the steepest baseline growth** (reaching 4.8× 2024 prices by 2031 with no restriction). This is driven by the compound effect of high background demand growth (g = 1.187/yr) and moderate τ_K (5.75yr): supply simply cannot keep pace with demand growth under normal conditions. If this trajectory is accurate, cobalt faces a structural supply deficit even without any geopolitical shock.

**Uranium and nickel both show oscillatory baseline patterns.** This reflects the ODE's inventory-rebuild dynamics for high-τ_K minerals: the system overshoots and undershoots equilibrium on a cycle of approximately τ_K years even without external shocks.

---

## 9.3 Cross-Mineral Full-Ban Comparison

The FULL_BAN scenario applies do(restriction = 0.30) for three years (2025–2027) to each mineral in isolation. This is a structural comparison scenario — it does not represent a realistic simultaneous multi-mineral ban, but rather isolates each mineral's structural response to an equivalent intervention.

| Mineral | τ_K | Peak × baseline | Peak yr | Norm yr | Lag | US reliance |
|---------|-----|----------------|---------|---------|-----|------------|
| graphite | 7.83yr | **1.590×** | 2026 | 2030 | +3yr | 100% |
| rare earths | 0.51yr | **1.580×** | 2026 | never | — | 14% |
| cobalt | 5.75yr | **4.476×** | 2031 | 2032 | +5yr | 76% |
| lithium | 1.34yr | **2.476×** | 2031 | 2032 | +5yr | 50% |
| nickel | 7.51yr | **2.896×** | 2030 | 2031 | +4yr | 40% |
| uranium | 14.89yr | **4.740×** | 2031 | never | — | 95% |

The FULL_BAN cross-mineral comparison reveals three structural clusters:

**Cluster 1 — Low peak, short lag (graphite, rare earths):** Both minerals peak near 1.58–1.59× baseline. Graphite's low peak reflects near-zero demand elasticity (η_D = −0.777) combined with the declining baseline absorbing some of the shock; rare earths' low peak reflects τ_K = 0.51yr (China can ramp production rapidly if the political decision to lift the restriction is made). The long "never normalises" result for rare earths reflects the structural demand growth in the baseline rather than a genuine non-convergence at the physical level.

**Cluster 2 — Moderate peak, medium lag (lithium, nickel):** Both minerals peak in the 2.5–2.9× range. Lithium's τ_K = 1.34yr limits the restriction impact, but demand growth means normalisation is delayed to 2032 — five years after restriction end. Nickel's τ_K = 7.51yr produces a longer structural lag despite a lower peak.

**Cluster 3 — High peak, long or no normalisation (cobalt, uranium):** Both minerals reach 4.5–4.7× baseline and either barely normalise (cobalt: 2032, +5yr) or never normalise (uranium: within the 2031 projection window). These are the highest structural vulnerability minerals. Cobalt's high peak reflects the steep background demand growth trajectory; uranium's reflects τ_K = 14.89yr — no new mine capacity can enter within a four-year restriction window regardless of price signal.

---

## 9.4 Severity Gradient Analysis

### 9.4.1 Graphite

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 1.000 | 2025 | 2025 | — |
| mild_ban | 1.590 | 2026 | 2027 | +1yr |
| full_ban | 1.590 | 2026 | 2030 | +3yr |
| severe_ban | 2.018 | 2026 | 2029 | +1yr |

Key structural insight: the MILD_BAN and FULL_BAN both peak at 1.590 because both impose a 30% restriction — the peak is determined by restriction magnitude, not duration. Duration determines the normalisation lag (MILD normalises in 1 year; FULL takes 3 years). The SEVERE_BAN (50% restriction) raises the peak to 2.018 but normalises faster because the larger price signal attracts fringe supply at lower multiples. This counter-intuitive result reflects the ODE's fringe entry mechanism: a larger price spike triggers earlier and more aggressive non-Chinese anode sourcing (Poland, Mozambique processing), which actually accelerates recovery relative to a moderate restriction that does not breach the fringe entry threshold.

### 9.4.2 Rare Earths

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 1.504 | 2031 | 2025 | — |
| mild_ban | 1.580 | 2026 | never | — |
| full_ban | 1.580 | 2026 | never | — |
| severe_ban | 1.888 | 2026 | never | — |

The rare earths baseline already reaches 1.504 by 2031 due to background demand growth from EV permanent magnets and wind turbine generators. All restriction scenarios peak modestly (1.58–1.89×) but never normalise relative to the growing baseline. The structural interpretation is that the baseline demand growth trajectory is the dominant factor: even after the restriction ends, the growing no-restriction baseline keeps pulling above the post-restriction price level. This does not represent a genuine supply failure — it reflects that rare earth demand is growing faster than supply under any restriction scenario.

The τ_K = 0.51yr "China ramp speed" means that if the restriction is lifted, China can restore supply within months. The "never normalises" result is a baseline trajectory artefact, not a supply incapacity finding.

### 9.4.3 Cobalt

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 4.795 | 2031 | 2025 | — |
| mild_ban | 4.755 | 2031 | 2031 | +5yr |
| full_ban | 4.476 | 2031 | 2032 | +5yr |
| severe_ban | 4.869 | 2029 | never | — |

The striking feature of cobalt's severity gradient is that MILD_BAN (4.755×) produces a higher final-year price than FULL_BAN (4.476×) in 2031. This counterintuitive result reflects the FULL_BAN's stronger price spike in 2027–2028, which triggers more demand substitution (LFP transition acceleration) in the ODE, moderating the 2031 outcome. The SEVERE_BAN hits a "never normalises" regime because four years of 50% restriction exhausts the model's fringe supply capacity and the inventory rebuild compounds beyond the projection window.

### 9.4.4 Lithium

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 2.605 | 2031 | 2025 | — |
| mild_ban | 2.557 | 2031 | 2027 | +1yr |
| full_ban | 2.476 | 2031 | 2032 | +5yr |
| severe_ban | 3.038 | 2026 | never | — |

Lithium's short τ_K (1.34yr) makes the MILD_BAN self-correcting within one year of restriction end — the fastest recovery of any mineral in the study. The FULL_BAN's five-year lag is driven by demand growth compounding, not structural supply incapacity. The SEVERE_BAN (50%, 4yr) exceeds the fringe supply capacity and enters the "never normalises" regime, showing that even a fast-responding mineral can be overwhelmed by a sufficiently large and prolonged restriction.

### 9.4.5 Nickel

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 2.731 | 2030 | 2025 | — |
| mild_ban | 2.776 | 2031 | never | — |
| full_ban | 2.896 | 2030 | 2031 | +4yr |
| severe_ban | 3.070 | 2026 | 2032 | +4yr |

Nickel's MILD_BAN produces a "never normalises" result despite a lower peak than FULL_BAN. This reflects the oscillatory dynamics of high-τ_K minerals under short restrictions: the two-year MILD_BAN creates an inventory trough that feeds into the ODE's oscillatory cycle, producing a trajectory that perpetually stays just above the growing no-restriction baseline through 2031 without converging. The FULL_BAN and SEVERE_BAN, paradoxically, normalise because the larger shock triggers a larger price response that overshoots the fringe entry threshold, pulling in more supply and dampening the oscillation.

### 9.4.6 Uranium

| Scenario | Peak × | Peak yr | Norm yr | Lag |
|----------|--------|---------|---------|-----|
| baseline | 2.560 | 2031 | 2025 | — |
| mild_ban | 3.738 | 2031 | never | — |
| full_ban | 4.740 | 2031 | never | — |
| severe_ban | 10.395 | 2031 | never | — |

Uranium's severity gradient is qualitatively different from all other minerals. The SEVERE_BAN (50%, 4yr) reaches 10.4× baseline — more than double the next highest (cobalt SEVERE_BAN 4.87×). This extreme sensitivity reflects τ_K = 14.89yr: uranium mine development takes approximately 15 years from discovery to production. No restriction of any duration within the 2025–2031 window can be offset by new mine supply; the entire adjustment must come through demand destruction (η_D = −0.001, essentially zero) or existing contract coverage (not modelled explicitly but estimated to cover approximately 2–3 years). All uranium restriction scenarios produce "never normalises" results because the projection window (through 2031) is shorter than τ_K itself.

The uranium forward analysis is presented in the policy chapter (Chapter 10) with the additional context of the Cigar Lake and Russia sanctions historical episodes.

---

## 9.5 Multi-Mineral Compound Scenario

### 9.5.1 Motivation

The preceding analysis examined each mineral in isolation. However, many geopolitical restriction scenarios would affect multiple minerals simultaneously. A China technology export control targeting EV battery materials would restrict graphite, rare earths, and potentially cobalt (via Chinese-owned DRC processing) simultaneously. A broader conflict scenario could disrupt Indonesian nickel, Kazakh uranium, and DRC cobalt in parallel.

The ODE model is a single-mineral model and cannot directly simulate compound shocks. However, the L2 interventional framework can bound the compound scenario: if minerals A and B are both shocked simultaneously, and their supply chains are upstream-independent (graphite anodes and REE magnets feed separate parts of the EV, not a common intermediate), then the compound price impact is bounded above by the product of the individual impacts (perfect complementarity) and below by the sum (perfect substitutability). For battery materials, the correct bound is closer to the sum, since battery manufacturers can substitute across cathode chemistries but cannot produce a battery without both an anode and a cathode.

### 9.5.2 Simultaneous China Battery Material Restriction (Illustrative)

Scenario: China imposes simultaneous 30% export restrictions on graphite (anode) and rare earths (permanent magnets for EV motors) from 2025.

Individual FULL_BAN peaks: graphite 1.59×, rare earths 1.58×.

Combined effect on EV battery supply chain (additive approximation):
- Anode cost impact: +59% on anode materials (graphite)
- Motor magnet cost impact: +58% on permanent magnet materials (REE)
- Combined battery pack cost impact: approximately +8–12% (graphite ~15% of cell cost, REE magnets ~5% of motor cost — small fractions of total vehicle cost)

The compound cost impact on EV affordability is therefore moderate — a 8–12% increase in battery pack cost — rather than catastrophic. This calculation bounds the macroeconomic impact: individual mineral price spikes, while large in commodity market terms, translate to relatively modest vehicle cost increases because minerals are a small fraction of finished product cost. The strategic significance of mineral supply shocks lies primarily in the supply chain disruption and production halt risk (no graphite = no anode production, regardless of cost) rather than in price transmission to consumer prices.

---

## 9.6 Structural Vulnerability Ranking

Integrating τ_K, FULL_BAN peak, normalisation lag, and US import reliance into a composite structural vulnerability assessment:

| Mineral | τ_K rank | Peak rank | Lag rank | US reliance rank | Composite | Classification |
|---------|----------|-----------|----------|-----------------|-----------|---------------|
| uranium | 1st (14.89yr) | 1st (4.74×) | 1st (never) | 1st (95%) | **Tier 1 Critical** | Highest |
| graphite | 3rd (7.83yr) | 5th (1.59×) | 3rd (+3yr) | 1st (100%) | **Tier 1 Critical** | Highest |
| cobalt | 2nd (5.75yr) | 2nd (4.48×) | 2nd (+5yr) | 2nd (76%) | **Tier 1 Critical** | High |
| nickel | 3rd (7.51yr) | 4th (2.90×) | 3rd (+4yr) | 4th (40%) | **Tier 2** | Moderate-High |
| rare earths | 6th (0.51yr) | 5th (1.58×) | 5th (never*) | 5th (14%) | **Tier 2** | Moderate |
| lithium | 5th (1.34yr) | 4th (2.48×) | 2nd (+5yr) | 3rd (50%) | **Tier 2** | Moderate |

*REE "never normalises" is a baseline growth artefact; structural recovery is fast due to low τ_K.

The composite ranking places uranium, graphite, and cobalt in the highest-vulnerability tier. Uranium's vulnerability is structural and geological: no new mine supply can enter within any plausible restriction window. Graphite's vulnerability is geopolitical: 100% US import reliance, 95% China processing concentration, no strategic reserve, no domestic alternative. Cobalt's vulnerability is compound: DRC mine concentration plus Chinese refining concentration, with a steep background demand growth trajectory.

Lithium, despite its short τ_K and allied supply base, scores moderate vulnerability because its near-zero demand elasticity (η_D = −0.062) and steep demand growth create severe short-run price spikes even under moderate restrictions. Rare earths, counter-intuitively, score lower on the composite: China's fast ramp speed (τ_K = 0.51yr) means that supply can be restored quickly once the political decision to lift a restriction is made, and US import reliance (14%) is the lowest in the study.

---

## 9.7 Cross-Mineral Policy Signal

The forward projections, taken together, identify three distinct policy imperatives that map to the structural parameters:

**Imperative 1: Stockpile high-τ_K minerals proportional to τ_K.** Uranium (τ_K = 14.89yr), graphite (7.83yr), and nickel (7.51yr) require multi-year strategic reserves because no new supply can enter within a restriction window. The NDS targets for these minerals should reflect τ_K/2 as a minimum reserve horizon: ~7 years for uranium, ~4 years for graphite, ~4 years for nickel.

**Imperative 2: Invest in processing diversification for processing-bound minerals.** Graphite (95% China processing), rare earths (85–97% China processing), and cobalt (65–78% China processing) are all processing-bound. Mine diversification without processing diversification does not resolve the supply vulnerability. Policy instruments — IRA 45X credits, DoD Section 232 processing offtake agreements, allied investment guarantees — should target the processing stage specifically.

**Imperative 3: Monitor LFP and technology substitution.** Cobalt's future demand trajectory depends critically on whether high-energy-density NMC chemistry retains market share in the EV transition. If LFP penetration reaches 60%+ of EV deployment, the effective cobalt demand elasticity rises substantially above the calibrated η_D = −0.542, and the forward projections overstate cobalt vulnerability. Policy should monitor cathode chemistry evolution as a leading indicator of effective demand elasticity change.
