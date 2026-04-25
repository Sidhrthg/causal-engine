# Chapter 7: Cobalt — Dual Concentration and the EV Demand Surge

## 7.1 Introduction

Cobalt presents the most structurally complex supply chain among the six minerals studied in this thesis. Unlike graphite or rare earth elements — where the vulnerability originates at a single dominant stage (Chinese processing) — cobalt faces a dual concentration problem: approximately 70% of global mine output originates in the Democratic Republic of Congo, and approximately 65–80% of cobalt hydroxide refining capacity is located in China. These two concentrations operate at separate stages of the supply chain and involve distinct actors, mechanisms, and policy levers. A disruption at either stage is sufficient to tighten the global supply balance; a simultaneous disruption at both would be severely amplifying.

The 2016–2018 EV demand surge episode is the most instructive case study for cobalt dynamics. Unlike the graphite or REE episodes — which were triggered by a specific policy action (export licence, export quota) with a dateable onset — the cobalt shock emerged gradually from demand-side acceleration. EV battery manufacturers had not contracted sufficient cobalt supply ahead of the 2016–2018 demand step. The price response was amplified by the artisanal small-scale mining (ASM) sector, which supplies approximately 15–20% of Congolese cobalt and responds to price signals with lags and volatility not present in large-scale industrial mines.

The 2016 episode is also notable for its post-shock trajectory: cobalt prices collapsed approximately 73% between the March 2018 peak (~$95,000/tonne CIF) and the 2019 trough, one of the fastest reversals in critical minerals markets. The reversal was driven by simultaneous supply entry (Glencore's Mutanda mine resumption, Chinese merchant refiners building inventory) and demand-side substitution: battery manufacturers accelerated the transition to low-cobalt (NMC 622, NMC 811) and cobalt-free (LFP) cathode chemistries in response to the price spike. This demand-side structural response is visible in the L3 residual analysis and creates one of the most complex U_t abduction profiles in this study.

The chapter proceeds as follows. Section 7.2 establishes the market context and US strategic exposure. Section 7.3 presents the causal knowledge graph. Section 7.4 covers the 2016 EV demand surge episode. Section 7.5 presents the L3 duration analysis. Section 7.6 discusses the LFP transition as a structural demand shift. Section 7.7 presents forward projections. Section 7.8 summarises findings.

---

## 7.2 Market Context and US Strategic Exposure

### 7.2.1 Supply Structure

The DRC's cobalt dominance reflects the geological coincidence of cobalt with copper in the Central African Copperbelt. Cobalt is not mined as a primary product anywhere at scale — it is always a byproduct, principally of copper mining in the DRC and nickel mining in Russia, Australia, and the Philippines. This byproduct dependency means cobalt supply cannot be independently expanded: it is constrained by the economics of the primary metal (copper or nickel), making cobalt uniquely inelastic to its own price signal.

Chinese refining dominance developed through direct investment: from approximately 2005 onward, Chinese state-owned enterprises (CMOC, China Molybdenum) and private trading houses (Zhejiang Huayou Cobalt) acquired DRC copper/cobalt assets and built out hydrometallurgical refining capacity in China. By 2022, Chinese facilities account for approximately 65–80% of global cobalt hydroxide-to-sulphate conversion — the processing step required before cobalt enters the battery cathode manufacturing chain.

The effective supply control for the United States, computed from CEPII bilateral trade data and USGS processing data via the causal knowledge graph:

| Year | DRC PRODUCES share | China PROCESSES share | Effective control | Binding stage |
|------|-------------------|----------------------|-------------------|---------------|
| 2015 | ~60% | ~55% | **60%** | Mine (DRC) |
| 2018 | ~70% | ~65% | **70%** | Mine (DRC) |
| 2022 | ~72% | ~78% | **78%** | Processing (China) |

The binding constraint shifts over the study period. In 2015–2018, DRC mine concentration was the primary structural vulnerability. By 2022, Chinese refining concentration had grown to the point where it becomes the binding stage — consistent with the graphite pattern, where processing dominance exceeded mine-level concentration.

### 7.2.2 US Demand and Strategic Exposure

US net import reliance for cobalt is approximately 76% (USGS MCS 2024). The United States has no commercially significant domestic cobalt production; the Freeport Cobalt refinery in Kokkola, Finland (US-owned but operated in Finland) and Idaho Cobalt Operations represent the most credible allied-supply pathway but are small relative to total demand.

The strategic exposure is compound: DRC political instability (multiple armed group disruptions to mining logistics since 2016), Chinese processing concentration, and ASM volatility all feed into price risk independently. A coordinated risk scenario — DRC logistics disruption coinciding with Chinese refinery export restriction — has no credible short-run mitigation without pre-positioned stockpiles.

The National Defense Stockpile (NDS) has historically held cobalt; the current stockpile target and drawdown history is a subject of the policy chapter (Chapter 10).

---

## 7.3 Causal Knowledge Graph

The knowledge graph for cobalt (Figure 7.1) is constructed by querying the HippoRAG-indexed document corpus with the Claude API-generated query: *"DRC Congo cobalt supply EV battery demand surge China refining artisanal mining price volatility."*

HippoRAG retrieves the six most relevant document chunks, the KGExtractor extracts causal triples, and the subgraph is snapshotted at 2016 and 2018 using `query_at_time(year)`.

**Figure 7.1: Cobalt 2016 Knowledge Graph Snapshot**

*(See `outputs/kg_scenarios/validation/cobalt_2016.png`)*

The 2016 snapshot shows:
- **Dual shock origin nodes**: DRC (mine concentration, 60% PRODUCES share) and China (refining, 55% PROCESSES share) — the dual concentration structure visible as two high-degree nodes
- **Blue node** (cobalt): focal commodity
- **Heat-coloured downstream nodes**: EV batteries, battery supply chain, cathode materials (red); consumer electronics, defence applications (orange)
- **ASM edge**: artisanal small-scale mining node with high-variance supply relationship edge
- **Propagation**: DRC mine → cobalt → China refinery → cathode\_materials → ev\_batteries

The 2018 snapshot (`outputs/kg_scenarios/validation/cobalt_2018.png`) shows elevated edge widths on the China PROCESSES relationship (65% share) and a new LFP substitution node beginning to appear — representing the battery chemistry substitution response triggered by the price spike, which the 2016 snapshot does not yet contain.

---

## 7.4 Episode: The 2016–2018 EV Demand Surge

### 7.4.1 Historical Context

Global EV sales crossed 1 million units in 2017 and approximately 2 million in 2018. The dominant battery chemistry for premium EVs (Tesla, BMW, Audi E-tron) was NMC 622 (Nickel-Manganese-Cobalt, 6:2:2 ratio), requiring approximately 6–15 kg of cobalt per vehicle. Battery manufacturers had not pre-positioned cobalt supply contracts sufficient to cover this demand step — cobalt was underweighted in supply chain planning because it had been stable and cheap through 2015.

The price spike was initiated in 2016 by a combination of demand acceleration and a DRC supply disruption: Glencore placed its Mutanda mine (approximately 25,000 tonnes/year, ~25% of global supply) on care-and-maintenance in August 2019 in anticipation of the price run-up, but the announcement itself tightened 2017–2018 forward expectations. Financial traders, observing the supply-demand imbalance, accumulated spot cobalt positions, amplifying the physical market signal.

The collapse was equally rapid: Glencore restarted Mutanda in December 2019 after the price peak; Chinese merchant refiners, who had built inventory at low 2015–2016 prices, liquidated positions; and battery manufacturers simultaneously began transitioning to NMC 811 (reducing cobalt content per kWh by approximately 30%) and LFP (zero cobalt). The market moved from scarcity to surplus within 18 months.

### 7.4.2 Calibrated Parameters

| Parameter | Value | Identification |
|-----------|-------|----------------|
| α_P | 1.661 | Price signal amplification; amplification regime (>1.5) |
| η_D | −0.542 | Moderate demand elasticity (battery substitution possible) |
| τ_K | 5.75yr | Copper/cobalt mine development timeline (DRC greenfield) |
| σ_P | 0.9060 | Largest residual noise among all episodes (ASM + financial volatility) |
| g | 1.1874/yr | EV demand background growth |

The α_P = 1.661 confirms cobalt operates in the amplification regime — a 1% inventory tightening generates a 1.66% price response. This amplification reflects both the financial market participation (commodity funds treating cobalt as an EV proxy) and the ASM supply volatility (ASM output is lumpy and price-responsive with a lag, creating inventory overshoot/undershoot cycles).

The η_D = −0.542 is notably different from lithium (η_D = −0.062). The larger demand elasticity reflects the fact that cobalt demand is partially price-responsive: battery manufacturers can and do switch cathode chemistries (NMC 622 → NMC 811 → LFP) when cobalt prices rise sufficiently. This elasticity is the supply chain's primary defensive mechanism against cobalt price shocks — it is absent for lithium (no substitute) and graphite (no substitute for anode material).

The σ_P = 0.9060 is the largest calibrated residual noise in this study, reflecting the genuine unpredictability of cobalt price dynamics: ASM output fluctuations, DRC political disruptions, and financial speculation all contribute to residuals that the structural ODE cannot represent.

### 7.4.3 Model Performance

From `outputs/predictability_run.txt`, formal validation against the World Bank LME cobalt spot price series (BACI unit values are unreliable for cobalt due to heterogeneous export forms — hydroxide, sulphate, and matte at varying cobalt content):

**cobalt_2016_ev_hype_and_crash (2015–2019)**

**DA = 1.000 | Spearman ρ = 1.000 | RMSE = 0.26 | MagR = 2.33 | Grade: A**

| Year | Model | World Bank | Δ Model | Δ WB | Agree |
|------|-------|-----------|---------|------|-------|
| 2015 | 1.000 | 1.000 | — | — | |
| 2016 | 0.681 | 0.985 | −0.319 | −0.015 | ✓ |
| 2017 | 1.628 | 2.101 | +0.947 | +1.116 | ✓ |
| 2018 | 4.606 | 3.218 | +2.978 | +1.117 | ✓ |
| 2019 | 1.164 | 1.248 | −3.443 | −1.969 | ✓ |

All four directional transitions correct (DA = 1.000). Spearman ρ = 1.000 confirms the model captures the full up–up–down rank ordering of the episode. MagR = 2.33 reflects the model's tendency to over-predict the 2018 peak amplitude relative to the World Bank annual average — a consequence of the LME spot series being smoothed across the year while the structural ODE builds peak inventory tightness instantaneously in the annual step.

**cobalt_2022_ev_demand_and_lfp_crash (2020–2024)**

**DA = 1.000 | Spearman ρ = 1.000 | RMSE = 0.84 | MagR = 1.89 | Grade: A**

| Year | Model | World Bank | Δ Model | Δ WB | Agree |
|------|-------|-----------|---------|------|-------|
| 2020 | 1.000 | 1.000 | — | — | |
| 2021 | 1.578 | 1.689 | +0.578 | +0.689 | ✓ |
| 2022 | 2.765 | 2.128 | +1.187 | +0.438 | ✓ |
| 2023 | 1.498 | 1.018 | −1.268 | −1.110 | ✓ |
| 2024 | 0.126 | 0.776 | −1.372 | −0.241 | ✓ |

All four directional transitions correct. The RMSE = 0.84 is the highest among cobalt episodes, driven by the 2024 model–data gap: the model predicts a very deep 2024 price fall (0.13×) while the World Bank annual average sits at 0.78×. This reflects the model's representation of the LFP structural demand shift — the model encodes LFP adoption as a permanent demand destruction that fully propagates by 2024, whereas actual cobalt prices were partially supported by Chinese battery manufacturers maintaining contractual cobalt procurement above spot-market-implied demand levels.

The extended trajectory below spans both episodes (duration analysis base year = 2015), presented to show the full 2015–2024 arc and the transition between the two structural regimes:

| Year | Model (nsr-pure) | CEPII (factual) | Δ Model | Δ CEPII | Agree |
|------|-----------------|-----------------|---------|---------|-------|
| 2015 | 1.000 | 1.000 | — | — | |
| 2016 | 1.326 | 1.326 | +0.326 | +0.326 | ✓ |
| 2017 | 1.639 | 2.279 | +0.313 | +0.953 | ✓ |
| 2018 | 2.026 | 3.026 | +0.387 | +0.747 | ✓ |
| 2019 | 2.500 | 3.518 | +0.474 | +0.493 | ✓ |
| 2020 | 0.980 | 0.464 | −1.520 | −3.054 | ✓ |
| 2021 | 2.644 | 1.673 | +1.664 | +1.209 | ✓ |
| 2022 | 4.633 | 3.587 | +1.989 | +1.914 | ✓ |
| 2023 | 6.183 | 5.653 | +1.550 | +2.066 | ✓ |
| 2024 | 7.027 | 7.200 | +0.844 | +1.547 | ✓ |

The model achieves perfect directional accuracy across all nine transitions in this extended table. The structural model correctly captures the 2020 price collapse (COVID demand destruction), the 2021 recovery (post-COVID EV restocking), and the continued demand growth through 2024. The model undershoots the magnitude of the 2016–2019 price spike (model 2.5× vs CEPII 3.5× at peak) — this magnitude gap reflects the financial speculation and ASM volatility captured in σ_P = 0.9060, which the structural ODE compresses.

---

## 7.5 L3 Duration Analysis

### 7.5.1 Abduction of Residual Trajectory

| Year | U_t (abducted) | Status |
|------|---------------|--------|
| 2015 | +0.000 | Baseline |
| 2016 | +0.528 | ⚠ shock-active |
| 2017 | −0.315 | ⚠ shock-active |
| 2018 | −0.131 | Post-peak |
| 2019 | −1.063 | Price collapse period |
| 2020 | +1.128 | Post-collapse recovery |
| 2021 | +0.356 | Restocking phase |
| 2022 | −0.206 | Moderate |
| 2023 | −1.384 | Market oversupply |
| 2024 | −2.128 | Deep oversupply |

The cobalt U_t profile is the most erratic of all episodes. The sign reversals — positive in 2016 (speculation drives CEPII above model), strongly negative in 2019 (collapse overshoots model), strongly positive in 2020 (COVID recovery spike), then deeply negative in 2023–2024 (LFP transition oversupply) — reflect the multiple causal channels simultaneously active in cobalt markets that the structural ODE cannot represent in a single set of time-invariant parameters.

The large negative U_t in 2023–2024 (−1.384, −2.128) reflects the structural demand reduction from LFP battery adoption: CEPII-measured cobalt prices have fallen well below what the ODE predicts from supply-demand fundamentals alone. This is the LFP transition embedded in the residuals — not a cyclical oversupply but a structural demand contraction as high-volume EV segments switch cathode chemistry.

### 7.5.2 Counterfactual Duration Table

Applying Pearl's L3 do-calculus — do(demand_surge ends year T) — for the 2016–2018 episode:

| | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | Norm yr | (corr) |
|---|------|------|------|------|------|------|------|------|------|------|---------|--------|
| nsr-l3 | 1.000 | 3.125 | 0.801 | 2.666 | 1.494 | 22.202 | 0.594 | 1.328 | 1.303 | 2.221 | (ref) | |
| nsr-pure | 1.000 | 1.326 | 1.639 | 2.026 | 2.500 | 0.980 | 2.644 | 4.633 | 6.183 | 7.027 | | |
| factual | 1.000 | 1.326 | 2.279 | 3.026 | 3.518 | 0.464 | 1.673 | 3.587 | 5.653 | 7.200 | | |
| T=2016 | 1.000 | 3.125 | 1.572 | 3.299 | 1.454 | 32.030 | 0.198 | 0.560 | 0.721 | 1.434 | 2019 (+3yr) | 2019 (+3yr) |
| T=2017 | 1.000 | 3.125 | 1.572 | 4.281 | 1.244 | 29.393 | 0.211 | 0.592 | 0.748 | 1.473 | never | never |
| T=2018 | 1.000 | 3.125 | 1.572 | 4.281 | 2.230 | 35.330 | 0.105 | 0.322 | 0.467 | 1.019 | never | never |
| T=2019 | 1.000 | 3.125 | 1.572 | 4.281 | 2.230 | 18.277 | 0.478 | 1.132 | 1.176 | 2.059 | never | never |

### 7.5.3 Structural Interpretation

The dominant feature of the cobalt L3 table is the explosion of the nsr-l3 reference at 2020: 22.202 under the base nsr-l3, and 35.330 for the T=2018 scenario. This reflects the large positive U_t = +1.128 at 2020 (post-collapse recovery spike) being injected into the ODE, which compounds through the inventory dynamics. The "never normalises" result for T ≥ 2017 is structurally analogous to the lithium 2022 case: the reference trajectory becomes explosive due to large abducted residuals, and the factual trajectory cannot converge to it.

The T=2016 scenario is the only one showing normalisation (+3yr). This is because cutting the shock at its first year avoids the inventory depletion that propagates through 2017–2018, producing a faster supply response and a reference trajectory that does not diverge as severely.

The policy-relevant finding is the three-year lag under T=2016: even if the demand surge had been fully absorbed by pre-positioned strategic reserve releases in 2016 (preventing the supply tightness), prices would have remained elevated for approximately three years — 2016–2019 — before normalising. This directly motivates the NDS cobalt stockpile recommendation: the reserve must cover not just the surge year but three years of inventory rebuild.

---

## 7.6 The LFP Transition: Structural Demand Shift

The 2023–2024 U_t residuals (−1.384, −2.128) are not noise — they are a structural signal. Lithium iron phosphate (LFP) batteries, which contain zero cobalt, have captured approximately 40% of global EV battery deployments by volume as of 2023, up from approximately 10% in 2019. CATL and BYD's Cell-to-Pack LFP designs have reached sufficient energy density for mass-market EVs; LFP chemistry has also penetrated grid storage at scale.

This structural transition is not captured in the ODE's time-invariant parameters. The η_D = −0.542 was calibrated from the 2016–2019 price spike, when battery manufacturers were already beginning the transition. By 2023–2024, the effective demand elasticity has become much higher — large price movements have already occurred, and battery designers have permanently redesigned products around lower-cobalt or cobalt-free cathodes.

The practical implication for the forward projections is important: the calibrated η_D = −0.542 is likely an underestimate of the current demand elasticity. If cobalt prices spike again in the 2025–2031 window, the demand-side substitution response would be faster and larger than the 2016–2018 calibration implies — battery manufacturers have already done the engineering work to reduce cobalt loading and would accelerate the transition further. The forward projections in Section 7.7 should therefore be read as upper bounds on cobalt price sensitivity, with the caveat that structural demand substitution may limit the spike to below the ODE's prediction.

---

## 7.7 Forward Projections

### 7.7.1 Scenario Design

All forward scenarios use the 2022 episode calibration (most recent observable regime): τ_K = 5.75yr, α_P = 1.661, η_D = −0.542, background demand growth g = 1.1874/yr. The hypothetical restriction posits a coordinated DRC export ban or Chinese refinery processing restriction.

### 7.7.2 Price Trajectory Table

Price index (P / P_2024):

| Scenario | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | Peak | Peak yr | Norm yr | Lag |
|----------|------|------|------|------|------|------|------|------|------|---------|---------|-----|
| baseline | 1.000 | 1.000 | 1.570 | 2.082 | 2.572 | 3.179 | 3.924 | 4.795 | 4.795 | 2031 | 2025 | — |
| mild_ban | 1.000 | 1.000 | 2.413 | 2.800 | 1.883 | 2.763 | 3.742 | 4.755 | 4.755 | 2031 | 2031 | +5yr |
| full_ban | 1.000 | 1.000 | 2.413 | 2.800 | 3.372 | 2.294 | 3.209 | 4.476 | 4.476 | 2031 | 2032 | +5yr |
| severe_ban | 1.000 | 1.000 | 3.192 | 3.771 | 4.213 | 4.869 | 1.881 | 3.447 | 4.869 | 2029 | never | — |

### 7.7.3 Key Findings

**Finding 1: Baseline trajectory is steeply upward.** Even without any restriction, cobalt prices reach 4.8× the 2024 level by 2031. This reflects the combination of high background demand growth (g = 1.187/yr) and moderate τ_K (5.75yr): supply cannot keep pace with demand growth even under normal conditions. This is qualitatively different from lithium, where supply response is faster, or graphite, where the baseline is flat.

**Finding 2: FULL_BAN peaks at 4.5× in 2031 — five-year lag to normalisation.** The five-year post-restriction lag reflects the τ_K = 5.75yr mine development cycle: new cobalt supply from DRC greenfield or Indonesian nickel-cobalt HPAL cannot enter within the restriction window, so the inventory drawdown compounds into a sustained price scar. Stockpile releases must be sustained approximately three years beyond the restriction end to suppress the rebound.

**Finding 3: LFP substitution acts as a natural ceiling.** Though the ODE does not model the LFP transition directly (it uses time-invariant η_D = −0.542), the historical record shows that cobalt prices above approximately $65,000–70,000/tonne trigger accelerated LFP adoption. If the forward shock drives prices to that level, the structural demand response would be faster than the model implies, limiting the actual peak below the 4.5× projection. This represents the primary downside risk to the forward projection — it may overstate cobalt vulnerability for high-energy-density applications while underweighting the LFP substitution pathway.

### 7.7.4 Stockpile Policy Implication

The NDS cobalt target of 3–5 years supply (as recommended in forward_run.txt) is justified by:
1. τ_K = 5.75yr mine development cycle: no new production enters within a five-year window
2. US import reliance of 76%: no significant domestic substitution
3. LFP transition partially mitigates risk for passenger EV applications but does not eliminate cobalt need for high-energy-density applications (aviation, defence, grid-scale long-duration storage)
4. Allied sourcing from Zambia (Chambishi Metals, Mopani Copper Mines) represents a credible partial diversification that reduces DRC concentration but does not address Chinese refining dependency

---

## 7.8 Summary

Cobalt represents the dual-concentration case study: mine-level (DRC, ~70%) and processing-level (China, ~65–80%) vulnerabilities coexist and can be independently triggered. The 2016–2018 demand surge episode demonstrates that EV demand acceleration creates price dynamics comparable in severity to supply-side export restrictions, with α_P = 1.661 placing cobalt in the amplification regime.

The L3 duration analysis reveals a "never normalises" result for restriction periods beyond one year — driven by the interaction of large abducted residuals (reflecting ASM volatility and financial speculation) with the structural ODE's inventory dynamics. The T=2016 scenario normalises within three years, directly calibrating the required stockpile duration.

The LFP battery transition creates a structural demand-side escape valve that the ODE does not represent. Forward projections should be treated as upper bounds for cobalt price sensitivity, with the understanding that demand elasticity above the 2016-calibrated η_D = −0.542 will dampen actual price peaks for passenger EV applications. High-energy-density applications (aviation, defence) face full exposure with no near-term cobalt-free substitute.
