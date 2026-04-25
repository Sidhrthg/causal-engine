# Chapter 3: Methods

## 3.1 Framework Overview

The causal engine combines three components that operate in sequence: a structurally-identified ordinary differential equation (ODE) model of commodity market dynamics, Pearl's causal hierarchy for associational, interventional, and counterfactual queries, and a knowledge-graph-backed event extraction pipeline that converts free-text policy events into model-compatible shock signals.

The core modelling cycle is: (1) structured events from text are mapped to named shock signals; (2) those signals enter the ODE as external forcing; (3) the ODE integrates forward under the Pearl L1/L2/L3 operator appropriate to the query type; (4) outputs are price trajectories, supply decompositions, and counterfactual comparisons. The four structural parameters {α_P, η_D, τ_K, g} are calibrated per episode from CEPII BACI bilateral trade data by differential evolution (Section 3.4). Five additional parameters are held at theory-consistent constants and subjected to sensitivity analysis (Section 4.6).

---

## 3.2 The ODE Commodity Model

### 3.2.1 State Space

The model tracks three continuous state variables at annual frequency:

| Symbol | Description | Units |
|--------|-------------|-------|
| K_t    | Effective production capacity | kt/yr |
| I_t    | Aggregate inventory | kt |
| P_t    | Market-clearing price | USD/tonne |

Initial conditions {K_0, I_0, P_0} are commodity-specific and drawn from USGS Mineral Commodity Summaries for the base year of each scenario. Scenarios are integrated forward using an annual Euler-Maruyama scheme with time step dt = 1 year.

### 3.2.2 The Eight-Equation Step

Each annual step computes the following quantities in order. Equations below use the notation from `src/minerals/model.py`; parameters are defined in Section 3.4 and Table 3.2.

**Step 1 — Utilisation and raw production.**
Capacity utilisation responds log-linearly to the price signal:

```
u_t = clip(u0 + β_u · log(P_t / P_ref), u_min, u_max)
Q_t = K_t · u_t
```

where `u0` is baseline utilisation, `β_u` controls the utilisation price elasticity, and `clip` enforces physical bounds [u_min, u_max] ⊆ [0, 1]. This ensures Q_t ≤ K_t always.

**Step 2 — Trend demand.**
Baseline demand grows at a constant rate g fitted by differential evolution (Section 3.4.4):

```
g_t = g^{t_index}
```

**Step 3 — Demand with price elasticity and policy.**
World demand responds to price, demand-side policies, and shock signals:

```
D_t = D_0 · g_t · (P_t / P_ref)^{η_D}
        · (1 − θ_sub) · (1 − θ_eff)
        · (1 + s_demand_surge) · m_destruction
```

where `η_D` is the demand price elasticity (negative), `θ_sub` and `θ_eff` are policy-level substitution and efficiency improvements, and `s_demand_surge`, `m_destruction` are shock signals from the event extractor.

**Step 4 — Effective supply from the dominant producer.**
An export restriction shock reduces the dominant producer's realized output:

```
supply_mult = (1 − s_export_restriction) · m_policy · m_capex
Q_eff_t = Q_t · supply_mult
```

**Step 4a — Substitution supply (Pearl L2 node: SubstitutionSupply).**
When the dominant exporter restricts exports, non-dominant producers respond to the resulting price premium:

```
price_premium = max(0, P_t / P_ref − 1)
Q_sub_t = s_export_restriction · Q_t · min(cap_sub, ε_sub · price_premium)
```

The structural causal interpretation: `ε_sub` mediates the path ExportPolicy → Price → SubstitutionSupply. A do-operator that severs the Price→SubstitutionSupply edge (do(ε_sub = 0)) eliminates Q_sub entirely, isolating the direct supply effect of the restriction. This node is a Pearl L2 intervention point: its structural equation can be surgically modified to evaluate counterfactual supply policies.

**Step 4b — Fringe supply (Pearl L2 node: FringeSupply).**
High-cost producers enter the market when price exceeds their cost threshold:

```
fringe_K = fringe_capacity_share · K_0
Q_fringe_t = min(fringe_K, fringe_K · max(0, P_t / P_ref − entry) / entry)
```

Total effective supply is `Q_total_t = Q_eff_t + Q_sub_t + Q_fringe_t`.

**Step 5 — Inventory.**

```
I_{t+1} = max(0, I_t + dt · (Q_total_t − D_t) + s_stockpile)
```

where `s_stockpile` is a scenario-specific one-time stockpile release (zero in baseline runs).

**Step 6 — Tightness and inventory cover.**

```
tight_t = (D_t − Q_total_t) / D_t
cover_t = I_{t+1} / D_t
```

**Step 7 — Price update (Euler-Maruyama in log space).**

```
log(P_{t+1}) = log(P_t)
              + dt · α_P · [tight_t − λ_cover · (cover_t − cover*)]
              + σ_P · √dt · ε_t
```

where ε_t ~ N(0, 1), `α_P` is the price adjustment speed, `λ_cover` weights the inventory-cover stabilization term, `cover*` is the target cover ratio, and `σ_P` is the price volatility parameter. In all production runs σ_P = 0 (deterministic forecasts); the noise term exists for Monte Carlo sensitivity studies.

**Step 8 — Capacity dynamics.**
Capacity mean-reverts toward a long-run target that responds to price:

```
K*_t = K_0 · (P_t / P_ref)^{η_K} · (1 + θ_subsidy) · (1 − s_capex)
K_{t+1} = K_t + dt · [(K*_t − K_t) / τ_K − r_retire · K_t]
```

where `τ_K` is the capacity adjustment half-life, `η_K` is the long-run capacity price elasticity, and `r_retire` is the annual retirement rate.

---

## 3.3 Pearl's Causal Hierarchy

The model implements Pearl's three-layer causal hierarchy (Pearl, 2009).

**Layer 1 — Association (L1): Seeing.**
Standard simulation: integrate the ODE forward from initial conditions under observed shock signals. Output is a price trajectory P_1, …, P_T and supply decomposition {Q_eff, Q_sub, Q_fringe}. This answers the query "given what we know about market conditions, what do we expect prices to do?"

**Layer 2 — Intervention (L1 → do-calculus): Doing.**
An intervention do(X = x) is implemented via graph surgery on the structural causal model. In the code this corresponds to directly setting a shock signal or parameter to a specified value, bypassing the causal mechanism that would ordinarily determine it. For example, do(export_restriction = 0.30) forces Q_eff to 70% of capacity regardless of the policy process that generated the restriction. The SubstitutionSupply and FringeSupply nodes (Steps 4a–4b) are the canonical L2 intervention points: their structural equations can be independently manipulated to answer questions like "what if non-OPEC producers had not responded to the price premium?"

**Layer 3 — Counterfactual: Imagining.**
L3 inference uses the Abduction-Action-Prediction protocol (Pearl, 2009, Ch. 7):

1. **Abduction**: given observed price data P̃_1, …, P̃_T and the model's predicted trajectory P^hat_1, …, P^hat_T, estimate the exogenous noise terms (residuals) U_t = P̃_t / P^hat_t for each observed year. These multiplicative residuals absorb all unit-specific variation not explained by the structural equations — idiosyncratic demand shocks, measurement error, and any factors not modelled.

2. **Action**: intervene on the structural graph. For example, set do(export_restriction = 0) to construct the no-restriction counterfactual.

3. **Prediction**: re-run the ODE under the counterfactual action while replaying the fitted residuals U_t. This gives the counterfactual trajectory P^{CF}_t, which is world-specific: it tells us what prices *would have been* in this particular historical episode had the policy differed, not what they would be in an average world.

The L3 implementation is in `src/minerals/causal_engine.py` (`CausalInferenceEngine.counterfactual()` method). Residuals are stored per-year and applied multiplicatively to the price state at each Euler step.

---

## 3.4 Parameter Identification

Each commodity-episode pair has four structural parameters {α_P, η_D, τ_K, g} calibrated to that episode by differential evolution (SciPy's `differential_evolution`, maximizing DA + Spearman ρ over the episode's CEPII validation window). The calibration is episode-specific: parameters are not shared across commodities and do not assume any functional form for the objective surface. Global optimization is used because the ODE-based objective function is non-convex in the joint parameter space — the directional accuracy landscape contains multiple local maxima, and gradient-based methods do not reliably find the global optimum.

The three causally-identified estimators described below (Sections 3.4.1–3.4.3) serve two roles. First, they motivate the sign constraints used in the DE search bounds: η_D ∈ [−1.5, −0.01], α_P ∈ [0.01, 4.0], τ_K ∈ [1, 20]. Second, they provide a consistency check: DE solutions that lie outside the region implied by the structural regressions warrant additional scrutiny. However, these estimators are not used to produce the evaluation parameters directly — a critical distinction discussed further in Section 3.4.6.

The identification strategies for individual parameters follow the recommendations in `src/minerals/causal_inference.py::CommoditySupplyChainDAG.get_parameter_identifications()`.

### 3.4.1 Demand Elasticity η_D — Two-Stage Least Squares

**Identification problem.** OLS regression of log-demand on log-price conflates demand shocks with supply shocks: in a growing market with a supply-constrained dominant producer, prices and quantities can move together even when demand is elastic. This produces an upward (positive) bias in OLS estimates of η_D.

**Instrument.** The two-year lag of the dominant exporter's export quantity changes, Δlog(dom_supply_{t-2}), serves as the excluded instrument. The identifying assumption is that capacity investment decisions made two years prior (permitting, capital expenditure commitments) are driven by geology and capital cycles rather than current world demand fluctuations, satisfying the exclusion restriction.

**Procedure.** Both stages are estimated in first differences to remove commodity-specific fixed effects:

```
First stage:   Δlog(P_t) = a + b · Δlog(dom_supply_{t-2}) + ε
Second stage:  Δlog(D_t) = a + η_D · Δloĝ(P_t) + ε
```

A first-stage F-statistic below 5 indicates a weak instrument; in this case the estimator falls back to OLS in log-log levels with a year trend, which is biased toward zero and reported with a flag. Confidence intervals are computed by a 500-draw circular block bootstrap with block size T^(1/3) ≈ 3 years to preserve the serial autocorrelation structure.

### 3.4.2 Price Adjustment Speed α_P — OLS on HP-Filtered Gap

The structural price equation (Step 7) implies that log-price changes should be proportional to the market tightness signal. However, the raw supply-shortfall proxy (lagged demand minus lagged supply) is endogenous: it conflates supply and demand shocks by construction.

The cleaner measure is the HP-filtered price cycle component:

```
price_gap_t = log(P_t) − log(P̃_t)
```

where P̃_t is the Hodrick-Prescott trend (smoothing parameter λ = 100, calibrated for annual data). The structural regression is then:

```
Δlog(P_t) = α_P · price_gap_{t-1} + noise
```

estimated by OLS. α_P must be positive (prices mean-revert upward when below trend); if the OLS point estimate is non-positive — which can occur when demand growth dominates the price signal — the estimate is replaced by the literature prior of 0.30.

### 3.4.3 Capacity Adjustment Half-Life τ_K — AR(1) on Log Supply

The capacity equation (Step 8) implies that log-capacity follows an AR(1) process in the neighbourhood of the steady state:

```
log(K_t) = c + ρ · log(K_{t-1}) + ε_t,   τ_K = 1 / (1 − ρ)
```

Since directly observable capacity series are unavailable for most commodities, the dominant exporter's annual export volume log(dom_supply_t) is used as a proxy — it tracks the production-side capacity decisions of the market's largest actor. The AR(1) coefficient ρ̂ is estimated by OLS; τ_K is recovered via the delta method. The estimate is constrained to ρ ∈ [0, 0.95] to rule out explosive or degenerate processes (ρ = 0.95 corresponds to τ_K = 20 years).

### 3.4.4 Demand Growth g — Per-Episode Calibration

The compound annual demand growth rate g is calibrated per episode jointly with {α_P, η_D, τ_K} by the same differential evolution procedure, using the DA + Spearman ρ objective. It is not shared across episodes: a mineral's demand growth rate during an EV adoption cycle (e.g., graphite 2022) is meaningfully different from its rate during a pre-EV industrial cycle (graphite 2008), and constraining them to a common value would impose a structural break assumption that cannot be verified on the available data.

### 3.4.5 Fixed Parameters

Five parameters are held constant across all commodities and scenarios. They are fixed at theory-consistent values rather than estimated because (a) they represent structural properties that cannot be identified separately from the calibrated parameters on annual trade data, and (b) sensitivity analysis (Section 4.6) shows the model's directional accuracy is robust to reasonable variation around these values.

| Symbol | Value | Interpretation |
|--------|-------|---------------|
| u0 | 0.92 | Baseline utilisation (92% of nameplate capacity) |
| β_u | 0.10 | Utilisation price elasticity |
| cover* | 0.20 | Target inventory cover ratio (≈2.4 months) |
| λ_cover | 0.60 | Weight of the inventory-stabilisation term in the price equation |
| σ_P | 0.00 | Price volatility (zero in deterministic forecasts) |

The sensitivity grid over {u0, cover*, λ_cover} is reported in Section 4.6.

### 3.4.6 Instrument Validity Diagnostics

The 2SLS estimator for η_D relies on the two-year supply lag as an excluded instrument. A first-stage F-statistic below 10 indicates instrument weakness (Staiger and Stock, 1997); below 5 is considered severely weak. Table 3.1 reports first-stage F-statistics for the three commodities with available CEPII price series.

**Table 3.1 — First-stage F-statistics for supply-lag instrument (η_D identification)**

| Commodity | F-statistic | Verdict | Fallback procedure |
|-----------|-------------|---------|---------|
| Graphite | 0.34 | Severely weak | OLS log-log with year trend (biased toward zero) |
| Lithium | 0.76 | Severely weak | OLS log-log → positive sign → clamped to −0.15 |
| Soybeans | 1.30 | Weak | OLS log-log with year trend (biased toward zero) |

All three instruments are severely weak. When F < 5, `parameter_fitting.py` falls back to OLS in log-log levels with a year trend, which is biased toward zero and explicitly flagged as such. For lithium, the OLS estimate was positive (sign reversal, consistent with a supply-constrained growing market), so the code clamps to −0.15 as a sign-floor prior. The η_D values used in the directional accuracy evaluation come from per-episode differential evolution calibration (`predictability.py`), not from this pipeline — a point that must be understood to interpret the identification section correctly.

Similarly, the HP-gap OLS estimator for α_P returns non-positive point estimates for all three commodities, defaulting to the literature prior of 0.30. The AR(1) estimator for τ_K produces plausible values (ρ ∈ [0.6, 0.85]) but these serve as starting-point priors rather than evaluation parameters.

The implication is transparent: the structurally-motivated identification strategies clarify *which direction* parameters should lie in and *why*, but the episode-specific values used in evaluation come from differential evolution calibration, which is empirically identified against the CEPII price series directly. This is the honest account of how the model's evaluated parameters are produced.

---

## 3.5 Knowledge Graph and Event Extraction

### 3.5.1 Seed Knowledge Graph

The knowledge graph (KG) represents causal and semantic relationships among commodity-market entities. It is constructed as a directed, typed, property-annotated multigraph using the schema defined in `src/minerals/knowledge_graph.py`.

**Entity types**: commodity, country, policy, company, facility, market, technology, event, economic indicator, industry, region, trade route, risk factor.

**Relationship types**: CAUSES, PRODUCES, EXPORTS_TO, IMPORTS_FROM, SUPPLIES, PROCESSES, DEPENDS_ON, SUBSTITUTES, COMPETES_WITH, REGULATES, INFLUENCES, CORRELATED_WITH, RISK_FOR, and IS_A (hierarchical taxonomy).

The **seed KG** contains 83 entities and 211 typed relationships, handcrafted from expert knowledge of critical minerals supply chains. Each relationship carries a confidence weight (0–1), a temporal scope, and a provenance tag.

**Dynamic CEPII enrichment.** PRODUCES relationships are further enriched with per-year exporter shares computed from CEPII BACI bilateral trade data (1995–2022). The `Relationship.share_at(year)` method linearly interpolates between annual data points, so the graph returns the correct share for any query year without discretising to fixed breakpoints. This eliminates the artefact of a static KG misrepresenting, e.g., China's graphite export share in 2008 (56%) versus 2022 (32%).

**PROCESSES relationships (seed values).** A separate PROCESSES relationship layer captures midstream beneficiation and processing concentration, which often differs substantially from raw production. For example, China's share of battery-grade anode processing rose from 65% in 2005 to 95% in 2022, while its raw flake export share declined. These PROCESSES relationships are hardcoded seed entries in `build_critical_minerals_kg()`, with per-year `yearly_share` dictionaries and provenance tags citing USGS Mineral Commodity Summaries, the Cobalt Institute, and IEA Critical Minerals 2021. They are not dynamically extracted from a data pipeline; they represent curated expert values with explicit source attribution. The `effective_control_at(country, commodity, year)` method returns the binding constraint — `max(produces_share, processes_share)` — and flags whether supply control is exercised at the mining or processing stage.

**Document-corpus enrichment.** `KGExtractor.enrich()` was applied to the full HippoRAG document corpus (1,661 passages from USGS MCS, IEA, and trade databases) using 8 episode-targeted queries. The enrichment merged **441 triples** extracted by GPT-4o-mini into the seed KG, adding **+389 entities** and **+377 relationships** for a final enriched KG of 472 entities and 588 relationships. By relationship type, the enriched KG contains: 118 CAUSES, 118 PRODUCES, 55 CONSUMES, 46 EXPORTS\_TO, 26 REGULATES, 20 DISRUPTS, 12 DEPENDS\_ON, and 11 PROCESSES edges, among others. The enriched KG is serialised to `data/canonical/enriched_kg.json` and loaded at inference time for knowledge graph visualisation.

**Separation of the KG and the quantitative pipeline.** The KG and the ODE simulation model (`src/minerals/simulate.py`) operate on parallel tracks. The ODE model is parameterised entirely from CEPII bilateral trade price series through differential evolution calibration; it does not query the KG at runtime. The KG's role is structural: `effective_control_at()` provides binding-stage identification (processing vs. mining) that informs how restriction magnitudes are interpreted in policy scenarios, and the enriched KG subgraph provides the causal pathway visualisation displayed in the API. The `KnowledgeGraph.to_causal_dag()` method, which extracts the CAUSES sub-graph as a callable DAG, is available for future integration but is not currently wired into the simulation pipeline.

### 3.5.2 Event Shock Mapper

Free-text policy events are converted to model-compatible shock signals by the `EventShockMapper` in `src/minerals/event_shock_mapper.py`. The pipeline has two components.

**Component 1 — Named entity recognition (rule-based).**
Country names and commodity aliases are matched using word-boundary regular expressions (`\b` prefix matching with explicit blocklist for common false-positive suffixes, e.g., "bank" from "ban"). All canonical aliases for each commodity are registered separately.

**Component 2 — Two-pass shock extraction.**
- *Pass 1 (confidence 0.60)*: requires a country match + a keyword match + a commodity match in the same sentence or ±2-sentence window.
- *Pass 2 (confidence 0.45)*: keyword + commodity only, without a country anchor. This recovers commodity-level events that do not name a specific country but carry strong directional language (e.g., "lithium glut deepens").

Each keyword maps to a shock type and a default direction via the `_KEYWORD_SHOCK` dictionary (34 entries covering export controls, bans, sanctions, price surges, demand destruction, capacity shocks, and geopolitical events). The mapper returns a list of `ShockMapping` objects with commodity, shock type, signed magnitude, duration, confidence, and the extracted evidence text.

**Evaluation.** The mapper is evaluated against a 20-example gold standard (plus 3 negative controls) defined in `src/minerals/extractor_eval.py`. Results are reported in Section 4.5 of the Results chapter. Summary: commodity F1 = 0.911, shock-type F1 = 0.878, direction accuracy = 0.938.

---

## 3.6 Transshipment Analysis

Export statistics from a dominant producer country frequently understate the effective reach of that country's supply restrictions because a significant fraction of exports is re-exported through intermediary countries before reaching final destinations. The `TransshipmentAnalyzer` in `src/minerals/transshipment.py` addresses this in four steps:

1. **Multi-hop flow tracing.** Given a CEPII bilateral trade matrix for year t, the analyzer traces flows from the origin (e.g., China → Singapore → Brazil → USA) and propagates quantity estimates at each hop using value matching and bottleneck constraints.

2. **Rerouting detection.** After a restriction event year, the analyzer tests whether non-producer hub countries show statistically significant (two-sample t-test, α = 0.10) increases in inbound flows from the restricting country and simultaneous increases in outbound flows to final destinations.

3. **Mirror statistics discrepancy.** The bilateral asymmetry between exporter-reported and importer-reported quantities (Y > X implies hidden flows or mislabelling) is computed as a discrepancy percentage.

4. **Circumvention rate estimation.** The effective rate at which a nominal export restriction is circumvented through third countries is estimated as the fraction of the original restricted volume that appears as increased hub-country re-exports within two years.

Circumvention-corrected supply series are available for use in parameter fitting to remove the resulting bias in η_D and τ_K estimates. In the current validation study, uncorrected CEPII data are used for all parameters; corrected series are used only for the transshipment scenario analysis reported in the API (`/api/transshipment` endpoint).

---

## 3.7 Validation Design

### 3.7.1 Episodes and Evaluation Metric

Eleven commodity-year episodes are evaluated: eight in-sample (each episode is also the source of its own DE-calibrated parameters) and seven cross-episode out-of-sample (OOS) pairs. The OOS design is a **cross-episode hold-out**, not a temporal split: parameters calibrated to one episode are applied to a different episode's shock sequence and validated against that episode's CEPII price series. This tests whether structural parameters generalize across market conditions, not merely across years within the same market regime. Episodes span the years 2000–2023 and cover graphite, lithium, cobalt, nickel, and soybeans.

The primary evaluation metric is **directional accuracy (DA)**: the proportion of annual price-change predictions for which the model's predicted direction (up/down) matches the realized direction:

```
DA = (1/T) Σ_t 1[sign(ΔP̂_t) = sign(ΔP_t)]
```

DA is chosen over RMSE because the primary policy question is "will this shock push prices up or down?"—correct directional calls are what a procurement analyst, policymaker, or investor needs to act on. RMSE is sensitive to the magnitude of price level assumptions and to the arbitrary choice of units; DA is unit-free and invariant to price normalization.

Spearman correlation ρ between predicted and realized price trajectories is reported as a secondary metric.

### 3.7.2 Circularity Control

Cobalt and nickel OOS pairs are flagged for partial circularity and reported separately. Both commodities have parameters calibrated against LME (London Metal Exchange) spot price series and then cross-validated against the same LME series. When donor and recipient episodes both use LME as the price target, the model cannot fail in a way that would reveal model misspecification relative to an independent data source — the calibration objective already optimized against the validation target. Cobalt and nickel results are included in Table 4.2 with an explicit circularity flag, and two mean DA values are reported: a clean mean using only graphite and soybeans OOS pairs (DA = 0.561), and a full mean including the partially circular pairs (DA = 0.740). All comparisons between the causal engine and baselines use the clean mean.

The remaining commodities (graphite, lithium, soybeans) use CEPII BACI bilateral trade unit values — computed as total trade value divided by total trade quantity — as the realized price series. This series is independent of the LME data used in calibration and is therefore a genuine out-of-sample validation target.

### 3.7.3 Baselines

Three baselines are evaluated:

- **Momentum**: predicts next year's price moves in the same direction as this year's price move — a trend-continuation rule. DA > 0.50 if price trends persist.
- **AR(1)**: price is forecast from a first-order autoregressive model estimated over the in-sample window. DA > 0.50 requires that past autocorrelation structure is informative about future direction.
- **Concentration Heuristic (CH)**: a signal-based baseline that predicts direction proportional to the net signed shock signal weighted by the dominant supplier's market share (e.g., China's 90% graphite share). This baseline uses the same shock inputs as the causal engine but replaces the full ODE simulation with a linear weighting rule. It tests whether the ODE machinery adds value beyond knowing that "China has 90% of graphite, so any restriction raises prices."

The causal engine is compared against all three baselines. Any model that correctly encodes the causal mechanism of a shock should outperform AR(1) and Momentum, which have no access to shock signals. Outperforming CH requires that the nonlinear ODE dynamics add directional information beyond the naive signal-weighting rule.

### 3.7.4 RAG Retrieval Parameters and Their Scope

The knowledge graph retrieval pipeline uses three hyperparameters: K = 5 (top-K retrieved chunks), damping = 0.85 (PageRank damping for entity reranking), and chunk size = 500 tokens. These parameters govern **retrieval quality only** — they affect which evidence passages are returned when the system is asked to explain a shock or to enrich the KG, but they do not enter the ODE forecast pipeline. The price trajectory produced by `engine.run()` is determined entirely by the structural parameters {η_D, α_P, τ_K, g, u0, β_u, τ_K, cover*, λ_cover} and the shock signals; the RAG parameters are upstream of shock extraction quality.

The appropriate sensitivity analysis for the RAG component is the extractor evaluation (Section 4.5, Table 4.5): commodity F1 = 0.911, direction accuracy = 0.938. The appropriate sensitivity analysis for the ODE parameters is the grid study in Section 4.6. These are two independent sensitivity analyses for two distinct sub-systems, and they should not be conflated.

---

## 3.8 Data Sources

| Dataset | Coverage | Used for |
|---------|----------|---------|
| CEPII BACI | Global bilateral trade flows, 2000–2022; HS6 product codes | Parameter fitting (η_D, τ_K), OOS validation prices |
| USGS Mineral Commodity Summaries | Annual world mine production, reserves, 2000–2024 | Initial conditions K_0, D_0; producer set for transshipment |
| World Bank Commodity Price Data | Monthly spot prices, 1960–2024 | Scenario reference prices P_ref; L3 abduction targets |
| IEA Critical Minerals Market Review | Demand forecasts by technology scenario | Demand growth baseline g priors |
| LME historical prices | Daily cobalt/nickel spot, 2000–2023 | Calibration target for cobalt/nickel (circularity-flagged) |

All data processing is performed in `src/minerals/parameter_fitting.py` and `src/minerals/transshipment.py`. Country name normalization (resolving ISO3 codes, historical name variants, and CEPII reporter codes) is handled by `src/minerals/country_codes.py`.

---

## Summary

The methods chapter has described:

- An 8-step Euler-Maruyama ODE with state {K, I, P} that models commodity market dynamics at annual frequency
- Two explicit Pearl L2 intervention nodes (SubstitutionSupply, FringeSupply) whose structural equations support do-calculus operations
- An L3 counterfactual protocol using Abduction-Action-Prediction with multiplicative residuals
- Per-episode differential evolution calibration of all four structural parameters {α_P, η_D, τ_K, g} against CEPII DA + Spearman ρ, with sign constraints motivated by causally-identified structural regressions (2SLS for η_D, HP-gap OLS for α_P, AR(1) for τ_K); instrument validity diagnostics (F < 2 for all three commodities) confirm the estimators inform priors but not evaluation parameters directly
- A dynamic knowledge graph enriched from 83-entity seed to 472 entities / 588 relationships via CEPII PRODUCES shares, USGS PROCESSES shares, and document-corpus triple extraction (441 triples from 8 episode queries); paired with a rule-based two-pass event shock mapper (commodity F1=0.911, type F1=0.878, direction=0.938)
- A transshipment analyzer that corrects supply series for re-export circumvention
- A validation design with explicit circularity control, CEPII-independent price series, and a directional accuracy metric appropriate for the policy question

Chapter 4 reports the results of this validation design.
