# Chapter 10: Policy and Structural Implications

## 10.1 Introduction

The preceding chapters have established, through five case study minerals and six historical episodes, that the causal engine framework can predict the direction and sequence of commodity price responses to supply shocks with high accuracy (median DA = 1.000 across all Grade A episodes), and can distinguish the structural parameters that govern recovery speed (τ_K), amplification (α_P), demand flexibility (η_D), and shock persistence. This chapter synthesises those findings into policy recommendations for US critical mineral supply security.

The policy analysis proceeds in three sections. Section 10.2 presents the uranium case study — folded into this chapter rather than a standalone chapter because uranium's L3 degeneracy (no CEPII data available) limits the depth of causal analysis, but the two uranium episodes provide important structural benchmarks for the stockpile policy framework. Section 10.3 develops the cross-mineral stockpile policy framework, using L3 normalisation lags and τ_K to derive mineral-specific strategic reserve targets. Section 10.4 examines the processing concentration problem — the structural vulnerability that mine-level supply diversification does not address. Section 10.5 presents alliance-based supply chain restructuring options calibrated to the structural parameters. Section 10.6 considers the limitations of the causal model as a policy tool.

---

## 10.2 Uranium: The Geological Ceiling Case

### 10.2.1 Market Context

Uranium occupies a structurally unique position among the six minerals studied: its τ_K = 14.89yr is the longest mine development cycle in this study, and its demand elasticity is essentially zero (η_D = −0.001). Nuclear power plants cannot substitute uranium as a fuel input; they cannot rapidly reduce consumption; and they typically hold 12–24 months of inventory under normal conditions via long-term supply contracts. These structural features make uranium the mineral most sensitive to sustained supply disruption.

US import reliance for uranium is approximately 95% (USGS MCS 2024) — the highest in this study. Kazakhstan (43% of global production), Russia (15% pre-2023, through TENEX/Rosatom), and Canada (12%) dominate supply. The Prohibiting Russian Uranium Imports Act (PRIA, signed May 2024) formalises the restriction on Russian enriched uranium, directly removing approximately 20% of US uranium supply enrichment services (SWU). This is the forward restriction scenario that the causal engine's uranium_2022 episode captures.

### 10.2.2 The Two Uranium Episodes

**Episode 1: Cigar Lake flood (2007)**

| Parameter | Value |
|-----------|-------|
| α_P | 2.064 |
| η_D | −0.436 |
| τ_K | 19.97yr |
| DA | 1.000 |
| ρ | 1.000 |
| RMSE | 0.18 |
| MagR | 0.91 |

The October 2006 flood at Cigar Lake mine — then under construction as the world's second-largest high-grade uranium deposit — removed approximately 15% of global future uranium supply in a single event. Combined with Nuclear Renaissance demand (30+ US reactor orders in 2006–2007), the EIA spot price rose approximately 6× from $14.77/lb in 2004 to $88.25/lb in 2007. The causal model calibrates τ_K = 19.97yr, consistent with uranium's known geological constraint: Cigar Lake itself took 20 years from discovery (1981) to first production (2014). DA = 1.000, ρ = 1.000 confirms the model correctly captures all price directional transitions including the 2008 GFC collapse.

L3 note: No CEPII bilateral trade data is available for uranium (nuclear materials are excluded from standard customs reporting for non-proliferation reasons). The L3 analysis degenerates to L2 for both uranium episodes — the U_t residuals cannot be abducted from CEPII, so σ_P = 0. This is documented in the methodology as a known limitation.

**Episode 2: Russia sanctions and PRIA (2022)**

| Parameter | Value |
|-----------|-------|
| α_P | 0.890 |
| η_D | −0.001 |
| τ_K | 14.89yr |
| DA | 1.000 |
| ρ | 1.000 |
| RMSE | 0.04 |
| MagR | 1.12 |

The 2022 episode reflects the cumulative effect of Russia/Ukraine war (reducing confidence in Rosatom supply contracts), Kazatomprom production cuts (~10%), and Sprott Physical Uranium Trust speculative demand. The PRIA formal ban took effect in May 2024. The calibrated α_P = 0.890 is notably below the amplification threshold (1.0), reflecting the utility market structure: nuclear fuel buyers are utilities operating under long-term contracts, not speculative traders. The price response is more muted per unit of supply tightness than in financial-market-traded commodities like cobalt.

The model achieves RMSE = 0.04 — the best fit in this study — reflecting the monotone rising trajectory (2020→2024 all positive) making the uranium episode structurally the most predictable of any mineral.

Year-by-year comparison (model vs CEPII, base 2020):

| Year | Model | CEPII | Agree |
|------|-------|-------|-------|
| 2020 | 1.000 | 1.000 | |
| 2021 | 1.124 | 1.065 | ✓ |
| 2022 | 1.349 | 1.418 | ✓ |
| 2023 | 1.861 | 1.799 | ✓ |
| 2024 | 2.471 | 2.506 | ✓ |

### 10.2.3 Uranium Forward Projections

| Scenario | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | Peak | Norm yr |
|----------|------|------|------|------|------|------|------|------|------|---------|
| baseline | 1.000 | 1.000 | 1.056 | 1.187 | 1.424 | 1.725 | 2.101 | 2.560 | 2.560 | 2025 |
| mild_ban | 1.000 | 1.000 | 1.486 | 2.199 | 2.477 | 2.818 | 3.235 | 3.738 | 3.738 | never |
| full_ban | 1.000 | 1.000 | 1.486 | 2.199 | 3.223 | 3.666 | 4.171 | 4.740 | 4.740 | never |
| severe_ban | 1.000 | 1.000 | 1.764 | 3.079 | 5.363 | 9.340 | 10.192 | 10.395 | 10.395 | never |

All restriction scenarios produce "never normalises" results within the 2031 projection window. The SEVERE_BAN (50%, 4yr) reaches 10.4× baseline — reflecting τ_K = 14.89yr: no new mine supply can enter within four years regardless of price level. The full demand inelasticity (η_D ≈ 0) eliminates the primary self-correcting mechanism that limits spikes in other minerals. Uranium has no effective short-run demand-side escape valve.

### 10.2.4 Uranium Policy Recommendation

The forward projections directly support the policy recommendations in the ADVANCE Act 2024 and DOE Uranium Reserve:
- **Expand DOE Uranium Reserve** from the current ~700,000 lb to a level covering approximately τ_K/2 ≈ 7 years of strategic consumption — far beyond the current 18-month contract buffer that utilities hold
- **Accelerate Centrus HALEU plant** (American Centrifuge plant, Piketon Ohio): HALEU (high-assay low-enriched uranium) for advanced reactors is 100% dependent on non-Russian enrichment post-PRIA; Centrus is the only licensed US enricher
- **Maintain Kazatomprom relationship**: Kazakhstan provides ~43% of global supply and is not subject to PRIA; long-term contract diversification to Kazatomprom and Cameco reduces Rosatom dependency without requiring new mine production

---

## 10.3 Cross-Mineral Stockpile Policy Framework

### 10.3.1 Stockpile Duration Targeting

The L3 normalisation lag analysis provides a principled basis for strategic reserve duration targets. The operative policy question is: for how long must a strategic reserve sustain supply releases before the market self-corrects? The answer is the L3 normalisation lag at the factual restriction end — the number of years after which prices return to within 10% of the no-shock baseline.

From the cross-mineral duration ranking:

| Mineral | τ_K | Normalisation lag (factual T) | Recommended reserve duration | Priority |
|---------|-----|------------------------------|------------------------------|---------|
| uranium | 14.89yr | never (within 7yr window) | **7yr** (τ_K/2) | Tier 1 |
| graphite | 7.83yr | +3yr | **4yr** (τ_K/2 + buffer) | Tier 1 |
| cobalt | 5.75yr | never (within 5yr window) | **3–5yr** | Tier 1 |
| nickel | 7.51yr | +3yr | **4yr** | Tier 2 |
| lithium | 1.34yr | never* | **90-day** (IEA buffer) | Tier 2 |
| rare earths | 0.51yr | +1yr | **1–2yr** (processing focus) | Tier 2 |

*Lithium "never" is a model boundary condition (Section 6.6.3), not a structural supply failure.

The Reserve Duration formula is: `Reserve_target = max(τ_K / 2, Norm_lag) + 1yr buffer`

For uranium and cobalt, where normalisation does not occur within the projection window, the formula defaults to τ_K/2 as the minimum reserve horizon.

### 10.3.2 Current Reserve Status and Gap

Comparing the recommended targets to known current strategic holdings (NDS, DOE Reserve, USGS assessments):

**Uranium**: DOE Uranium Reserve holds approximately 700,000 lb (≈18 months equivalent), vs recommended 7yr target. Gap: approximately 5 years of strategic consumption. The ADVANCE Act 2024 authorises Reserve expansion but funding is outstanding.

**Graphite**: No US strategic reserve exists. The 4yr reserve target represents approximately 360,000 tonnes of natural graphite (based on current consumption trajectory). Zero strategic buffer means any restriction takes immediate effect on US battery manufacturing. Gap: 4 years from zero.

**Cobalt**: NDS holds cobalt but the current target is classified. Public reporting suggests holdings equivalent to approximately 1–2 years of consumption, vs a 3–5yr target. Gap: approximately 2–3 years.

**Rare earths**: DoD NDS holds limited REE stockpile primarily in separated oxide form. The 1–2yr reserve target is partially met, but the processing bottleneck (separation and alloying outside China) means raw oxide holdings do not translate to finished alloy availability. Gap: processing capacity, not material quantity.

**Lithium**: No strategic reserve. IEA 90-day target is modest; at current US consumption growth, a 90-day reserve would require approximately 30,000 tonnes lithium carbonate equivalent. No formal US lithium reserve mechanism exists.

**Nickel**: No strategic reserve. Given 40% import reliance and Canadian allied supply, the priority is lower; a 4yr target may be reduced to 2yr given allied supply resilience.

---

## 10.4 Processing Concentration: The Hidden Structural Vulnerability

The case study chapters reveal a consistent structural pattern: for five of the six minerals studied, the binding supply constraint is at the processing stage, not the mining stage. This section synthesises that finding and derives its policy implications.

### 10.4.1 Processing Concentration Summary

| Mineral | Mine concentration | Processing concentration | Binding stage | Processing country |
|---------|------------------|------------------------|---------------|------------------|
| graphite | China 32% (ore) | China **95%** (anode) | Processing | China |
| rare earths | China 60% (ore) | China **85–97%** (separation/refining) | Processing | China |
| cobalt | DRC 72% (ore) | China **65–78%** (hydroxide→sulphate) | Processing (post-2022) | China |
| lithium | Australia 55% | China **65%** (LiOH/Li₂CO₃) | Processing | China |
| nickel | Indonesia 37% | China **40%** Class I refined; Indonesia **50%** HPAL/NPI (Chinese-invested) | Processing | China/Indonesia |
| uranium | Kazakhstan 43% | Russia **20% SWU** (enrichment) | Processing | Russia/US |

China is the dominant processor across five of the six minerals. This processing concentration is not coincidental — it reflects a deliberate strategic investment in mid-stream processing capacity over the 2005–2020 period, facilitated by state industrial policy, subsidised capital, and tolerance of environmental externalities that raised the cost of processing in Western markets.

For uranium, the analogous processing concentration is Russian SWU (separative work unit) enrichment services — the same pattern of state-controlled mid-stream capacity, now being disrupted by PRIA.

### 10.4.2 Why Mine Diversification Does Not Solve the Problem

Consider the case of graphite. If a US battery manufacturer sources raw graphite ore from Mozambique instead of China, the supply chain reads: Mozambique ore → Chinese anode processing facility → US battery manufacturer. The upstream mine concentration is resolved; the processing concentration is not. The ore is still entering Chinese value-added processing, and a Chinese export restriction on processed anodes (the actual instrument used in 2023) would still take immediate effect.

The same structural logic applies to:
- REE: Mountain Pass mine (California) ships ore concentrate to China for separation, because no US commercial separation facility operates at scale
- Cobalt: DRC cobalt hydroxide feeds Chinese refineries producing battery-grade sulphate; no US or allied cobalt refinery operates at commercial scale
- Lithium: Australian spodumene feeds Chinese lithium hydroxide conversion facilities; US automakers buying from "Australian" supply are still dependent on Chinese refining

The policy implication is direct: supply security initiatives that fund mine development without simultaneously funding processing infrastructure address the wrong bottleneck. The US mining sector is not the primary vulnerability in any of the five minerals where China dominates — the processing sector is.

### 10.4.3 Effective Policy Instruments for Processing

Three policy instruments can directly address processing concentration:

**1. IRA Section 45X Advanced Manufacturing Production Credit**: provides $35/kWh credit for battery cells manufactured in the US, which creates upstream demand for US-processed materials. Indirectly incentivises domestic/allied processing investment. Current coverage: battery cells, modules, active material. Gap: does not directly credit mineral processing (lithium hydroxide conversion, graphite anode purification).

**2. DoD Title III Defense Production Act (DPA) funding**: allows DoD to fund industrial capacity investments deemed essential for national defence. Has been used for rare earth separation (MP Materials Section 232 processing agreement) and cobalt (classified). Could be extended to graphite anode processing (currently absent) and lithium hydroxide conversion (partially covered by Albemarle/Livent expansions).

**3. Allied processing investment coordination**: Five Eyes and Quad frameworks could coordinate allied processing investment across specialised facilities — Japan (REE separation expertise via Sumitomo, Toyota Tsusho), Australia (lithium hydroxide, nickel HPAL), Canada (uranium enrichment, cobalt refining) — to create a collective non-Chinese processing supply chain. No formal mechanism for this coordination currently exists at the multilateral level; bilateral agreements (US-Japan Critical Minerals Agreement 2023, US-Australia CGT 2023) represent partial steps.

---

## 10.5 Alliance-Based Supply Chain Restructuring

### 10.5.1 Counterfactual Supply Shares Under Full Allied Sourcing

The causal model's effective\_control\_at() function can be run under hypothetical supply structure assumptions. The following table shows effective control if the US sourced exclusively from Free Trade Agreement (FTA) partners and Five Eyes allies:

| Mineral | Current effective control (China) | Allied supply scenario effective control | Residual vulnerability |
|---------|----------------------------------|----------------------------------------|----------------------|
| graphite | 95% (processing) | ~35% (Poland anode, Mozambique ore, US synthetic) | Moderate — processing gap persists |
| rare earths | 85% (separation) | ~40% (MP Materials Phase 2, Japan separation) | Moderate — separation capacity constrained |
| cobalt | 78% (processing) | ~55% (Zambia mining, Finland refining) | High — no major allied refiner |
| lithium | 65% (processing) | ~35% (Albemarle/Livent US processing) | Low-moderate — IRA investments underway |
| nickel | 40% China Class I + 50% Indonesia HPAL (Chinese-invested) = ~90% China-aligned | ~45% (Canada, Australia sulphide) | Moderate — sulphide supply insufficient for battery scale |
| uranium | 20% SWU (Russia) | ~5% (Centrus, Urenco US) | Low — ADVANCE Act path credible |

The allied supply scenarios show that full diversion from Chinese sourcing is achievable in principle for lithium and uranium, partially achievable for graphite and rare earths, and remains difficult for cobalt and nickel where no major allied processing capacity exists.

### 10.5.2 The "Australia Pivot" for Battery Materials

Australia holds significant advantages for three battery minerals: lithium (55% global mine share), nickel (HPAL-capable), and cobalt (potential through Nickel West cobalt byproduct). A strategic bilateral investment program between the US and Australia — extending the 2023 Critical Minerals and Clean Energy Transformation Compact — could address processing gaps for all three simultaneously:

- Lithium hydroxide conversion capacity at Kwinana (Albemarle) and Kemerton: already funded; needs scale-up
- Nickel HPAL at Ravensthorpe: BHP/Nickel West has existing HPAL; IRA-incentivised offtake could anchor US supply
- Cobalt at Nickel West (cobalt is a byproduct): small volumes but non-Chinese origin

This "Australia pivot" does not eliminate Chinese processing dependency (Australian capacity cannot match Chinese scale near-term), but reduces it from 70–95% to approximately 50–60%, moving below the threshold where a Chinese restriction creates a complete supply halt.

---

## 10.6 Limitations of the Causal Model as a Policy Tool

The causal engine framework provides principled, structurally grounded policy recommendations that derive from independently identified model parameters. However, several limitations should inform how these recommendations are used.

### 10.6.1 Annual Temporal Resolution

The ODE model operates on annual time steps. Real supply shocks propagate on shorter timescales: the March 2022 nickel squeeze was a within-week event; the cobalt 2018 peak was within-month. Policy responses (stockpile releases, emergency producer contacts) also operate on daily-to-weekly timescales. The model provides annual-resolution bounds, not operational-resolution signals. Real-time supply security management requires supplementary monitoring systems beyond the causal engine's scope.

### 10.6.2 Static Parameter Assumption

The calibrated parameters (α_P, η_D, τ_K) are held constant within each episode and across the forward projection window. This assumption breaks down when:
- Technology shifts change effective demand elasticity (LFP transition in cobalt)
- New processing technologies compress τ_K (HPAL in nickel)
- Market structure changes alter α_P (financialisation of lithium markets in 2022)

The L3 U_t residuals absorb these structural changes in the historical episodes, but forward projections cannot anticipate them. Forward scenarios should be re-calibrated as new episodes provide updated parameter estimates.

### 10.6.3 Single-Commodity Scope

The model prices each commodity independently. In reality, supply chain integration creates cross-commodity dependencies: a graphite restriction reduces EV production, reducing lithium demand, affecting lithium prices. The compound scenario analysis in Chapter 9 approximated these interactions linearly; a multi-commodity general equilibrium model would be required to capture the non-linear interactions precisely.

### 10.6.4 Political Risk Scope

The model takes the restriction parameters (magnitude, duration) as exogenous policy inputs. It does not model the political decision function that produces restrictions — the probability that China imposes a graphite ban in 2026, or that Indonesia tightens nickel ore export rules, is not estimated by the causal engine. Integrating geopolitical risk models (such as GPRI scores or escalation dynamics models) with the causal engine's structural price model would extend the framework to full risk-integrated supply security assessment.

---

## 10.7 Summary of Policy Findings

Five structural findings cross all case study minerals and carry direct policy implications:

**Finding 1**: Processing concentration is the binding vulnerability for five of six minerals. Mine-level supply diversification is necessary but insufficient. Policy investment must explicitly target processing capacity in allied countries.

**Finding 2**: The τ_K parameter determines minimum strategic reserve duration. Minerals with τ_K > 5yr (uranium 14.9yr, graphite 7.8yr, nickel 7.5yr, cobalt 5.75yr) require multi-year strategic reserves; minerals with τ_K < 2yr (rare earths 0.5yr China ramp, lithium 1.3yr) can rely on shorter-term buffer stocks.

**Finding 3**: The L3 normalisation lag exceeds the L2 counterfactual lag in every episode with CEPII data. This means that simply ending a restriction does not end its price effects — the carry-forward of inventory depletion and capacity destruction creates a price scar of 1–5 years beyond restriction end. Stockpile releases should therefore continue for τ_K/2 years after the restriction ends, not immediately cease when the restriction is lifted.

**Finding 4**: Demand elasticity (η_D) is the primary short-run defence against supply shocks. Minerals with near-zero η_D (lithium −0.062, uranium −0.001) have no self-correcting demand-side mechanism; price spikes are fully transmitted. Policies that increase demand-side flexibility — standardised battery modules enabling rapid cathode chemistry substitution, stockpile-triggered demand switching — directly reduce effective η_D and reduce the shock transmission magnitude.

**Finding 5**: Technology adaptation can compress τ_K below the historical mine development benchmark. Indonesia's HPAL response demonstrates that brownfield processing investment can deliver supply at 3× the speed of greenfield mine development. This has both a threat implication (US allies could replicate HPAL-style responses faster than τ_K implies) and a policy implication (the US should invest in shovel-ready processing capacity — engineered and permitted but not yet built — that can be activated within 2–3 years of a supply shock, rather than relying on normal market timelines).
