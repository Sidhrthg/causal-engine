# Chapter 8: Nickel — Technology Re-Routing and the Limits of Export Restriction

## 8.1 Introduction

The Indonesia nickel ore export ban of 2020 is the only episode in this study where a supply restriction was ultimately defeated by a technology adaptation response. Indonesia banned unprocessed nickel ore exports in January 2020, expecting to force downstream processing investment within the country and capture greater value-added revenue. The ban succeeded in that immediate objective: Chinese battery material companies, facing the loss of their primary nickel ore supply, invested rapidly in High-Pressure Acid Leach (HPAL) processing facilities inside Indonesia. By 2022–2023, these facilities had flooded the global nickel market with high-purity Class 1 nickel and mixed hydroxide precipitate (MHP), collapsing the price structure that the export ban had initially supported.

This outcome demonstrates a causal mechanism that the ODE's static circumvention\_rate parameter cannot fully represent ex-ante: technology re-routing, where the restriction triggers a capital investment response that ultimately expands global supply beyond pre-restriction levels. The causal knowledge graph captures this pathway — Indonesia → HPAL investment → expanded processed supply — but the timing and scale of HPAL capacity ramp-up within a four-year window was faster than any prior mine development cycle would have predicted.

For the causal model, this creates an important structural tension. The calibrated τ_K = 7.51yr reflects the historical nickel mine development timeline (greenfield laterite projects in the Philippines, Papua New Guinea, or Australia). The HPAL response operated on a shorter timeline — approximately 2–3 years from investment commitment to first production — because it built processing facilities on existing ore bodies rather than developing new mines. The L3 analysis separates the ban-driven price effects from the HPAL-technology-driven effects: the former is captured in the structural parameters, the latter appears in the U_t residuals as a supply-expanding shock that the ban-parameter framework cannot accommodate.

This chapter presents the full causal analysis of the Indonesia 2020 nickel ban, the HPAL technology response, and the implications for forward supply projections. Section 8.2 establishes market context. Section 8.3 presents the causal knowledge graph. Section 8.4 covers the 2020 episode. Section 8.5 presents the L3 duration analysis. Section 8.6 discusses the HPAL technology disruption in detail. Section 8.7 presents forward projections. Section 8.8 summarises.

---

## 8.2 Market Context and US Strategic Exposure

### 8.2.1 Supply Structure

Nickel supply is structured around two distinct ore types with different downstream uses. Sulphide ores (Canada, Russia, Australia) produce high-grade Class 1 nickel directly suitable for battery applications. Laterite ores (Indonesia, Philippines, New Caledonia) require more intensive processing (RKEF smelting for stainless steel or HPAL for battery-grade nickel). Indonesia holds approximately 26% of global nickel reserves and dominates laterite ore supply.

As of 2022, Indonesia accounts for approximately 37% of global refined nickel supply (following the HPAL expansion); the Philippines contributes approximately 13% as ore exporter. The dominant supplier concentration is less extreme than graphite (China 95%) or cobalt (DRC 72%), but Indonesia's ore ban demonstrated that ore-level concentration can be as strategically significant as refined metal concentration — the supply chain cannot easily source laterite ore substitutes at scale.

| Year | Indonesia PRODUCES share | China PROCESSES share (stainless/battery) | Effective control | Binding stage |
|------|-------------------------|------------------------------------------|-------------------|---------------|
| 2019 | ~33% (ore) | ~55% (RKEF/stainless) | **55%** | Processing (China) |
| 2021 | ~37% (ore) | ~60% (HPAL ramp) | **60%** | Processing (China) |
| 2023 | ~40% (ore+proc) | ~65% (HPAL full) | **65%** | Processing (China/Indonesia) |

The effective control calculation reveals a nuance: post-HPAL, Indonesia itself has become a significant processing centre, but Chinese-owned HPAL facilities in Indonesia (owned by CNGR, Zhejiang Huayou, GEM) mean that Chinese corporate interests control the processing even when the processing occurs in Indonesia. The geopolitical interpretation therefore depends on whether control is measured by physical location (Indonesia) or corporate nationality (China). For the US supply chain, the operative question is whether supply is available on Western market terms — and HPAL output from Chinese-owned Indonesian facilities is not.

### 8.2.2 US Demand and Strategic Exposure

US net import reliance for nickel is approximately 40% (USGS MCS 2024), the lowest among the supply-restricted minerals (graphite: 100%, cobalt: 76%). Domestic nickel production from the Eagle Mine (Michigan) and Vale's Copper Cliff operations (through recycled secondary supply) partially offsets import needs, and Canada — a close ally — is a significant sulphide nickel supplier.

However, the battery-grade nickel market is more concentrated than the stainless steel nickel market: HPAL-grade MHP and Class 1 nickel for NMC cathodes is produced at significant scale only in Indonesia (Chinese-owned HPAL) and limited sulphide operations. US EV manufacturers' access to battery-grade nickel on non-Chinese terms is constrained.

---

## 8.3 Causal Knowledge Graph

The knowledge graph for nickel (Figure 8.1) is constructed by querying the HippoRAG-indexed corpus with the Claude API-generated query: *"Indonesia nickel ore export ban HPAL technology investment battery grade nickel supply price volatility."*

**Figure 8.1: Nickel 2020 Knowledge Graph Snapshot**

*(See `outputs/kg_scenarios/validation/nickel_2020.png`)*

The 2020 snapshot shows:
- **Shock origin**: Indonesia (ore ban node, 33% PRODUCES share), with export restriction edge
- **Technology node**: HPAL\_investment as a secondary focal node — a forward-looking causal entity representing the investment response the ban triggered
- **China node**: PROCESSES share 60% via RKEF and emerging HPAL
- **Blue node** (nickel): focal commodity
- **Heat-coloured downstream nodes**: stainless steel, EV batteries, battery supply chain (orange/red)
- **HPAL technology edge**: Indonesia → HPAL\_investment → expanded\_processed\_supply (this edge has low impact in 2020 but becomes the dominant causal path by 2023)
- **Circumvention pathway**: Philippine ore exporters partially compensating for Indonesian ban

The structural difference from the 2023 topology (`outputs/kg_scenarios/validation/nickel_2023.png`) is striking: by 2023, the HPAL\_investment node has become the dominant source of supply (high-degree, maximum edge width) and the restriction node has diminished in impact weight — reflecting the causal re-routing the model documents.

---

## 8.4 Episode: The 2020 Indonesia Ore Export Ban

### 8.4.1 Historical Context

Indonesia implemented its nickel ore export ban effective 1 January 2020, two years ahead of the originally announced 2022 date, as a deliberate strategic acceleration to force immediate Chinese investment decisions. The ban applied to all unprocessed nickel ore with less than 1.7% nickel content — effectively all laterite ore from Indonesian mines.

The initial market response was a modest price rise: the LME nickel price reached approximately $14,000/tonne in early 2020 before COVID-related demand destruction collapsed it to approximately $11,000 by April 2020. The structural impact only manifested fully in 2021–2022, when post-COVID EV demand recovery met the supply constraint created by two years of Indonesian ore ban. The LME nickel price reached $48,078/tonne on 8 March 2022 in a short-squeeze event (the "nickel squeeze" of March 2022, driven by Tsingshan's short position covering), before trading was suspended by the LME.

The HPAL response timeline: Chinese battery material companies (CNGR, Huayou Cobalt, GEM) signed investment commitments for Indonesian HPAL facilities in 2020–2021, with first production beginning in 2022 and full capacity reached by 2023. By end-2023, Indonesian HPAL had added approximately 100,000 tonnes/year of battery-grade nickel and MHP, transforming the market from deficit to significant surplus. The nickel price fell from the 2022 spike to approximately $16,000/tonne by early 2024.

### 8.4.2 Calibrated Parameters

| Parameter | Value | Identification |
|-----------|-------|----------------|
| α_P | 1.621 | Price amplification; amplification regime (>1.5) |
| η_D | −0.495 | Moderate demand elasticity (stainless + battery bifurcation) |
| τ_K | 7.514yr | Historical nickel mine development (greenfield laterite) |
| σ_P | 0.4070 | Moderate residual noise (HPAL response in residuals) |
| g | 1.1679/yr | EV + stainless background demand growth |

The τ_K = 7.51yr is calibrated to greenfield mine development, not HPAL facility construction. This is the critical structural tension: the HPAL response operated on a 2–3 year construction timeline, faster by a factor of approximately 3 than the τ_K implies. The U_t residuals consequently absorb the HPAL supply expansion as a large unexpected supply shock — reducing prices below the ODE's structural prediction from 2023 onward. This is the correct behaviour of the L3 machinery: when a technology response occurs faster than the structural model's capacity parameter, the L3 abduction captures it as a residual rather than misattributing it to the structural ban parameter.

### 8.4.3 Model Performance

From `outputs/predictability_run.txt`, formal validation against the World Bank Pink Sheet nickel price series (independent of USGS/CEPII calibration sources):

**nickel_2006_stainless_boom_and_gfc (2005–2009)**

**DA = 1.000 | Spearman ρ = 1.000 | RMSE = 2.64 | MagR = 2.42 | Grade: A**

| Year | Model | World Bank | Δ Model | Δ WB | Agree |
|------|-------|-----------|---------|------|-------|
| 2005 | 1.000 | 1.000 | — | — | |
| 2006 | 1.125 | 1.645 | +0.125 | +0.645 | ✓ |
| 2007 | 3.221 | 2.525 | +2.096 | +0.880 | ✓ |
| 2008 | 1.020 | 1.432 | −2.201 | −1.093 | ✓ |
| 2009 | 0.003 | 0.994 | −1.017 | −0.438 | ✓ |

All four directional transitions correct (DA = 1.000). RMSE = 2.64 is elevated, driven by the 2009 model prediction near zero (0.003×) while the World Bank annual average stays near 1.0×: the ODE's macro\_demand\_shock collapses inventory to near-zero but annual price data smooths the within-year GFC trough. This is a known artefact of annual resolution — the structural mechanism is correct (demand shock → price collapse) but the timing and depth are compressed in the annual data.

**nickel_2022_hpal_oversupply_crash (2020–2024)**

**DA = 1.000 | Spearman ρ = 0.900 | RMSE = 0.27 | MagR = 1.30 | Grade: A**

| Year | Model | World Bank | Δ Model | Δ WB | Agree |
|------|-------|-----------|---------|------|-------|
| 2020 | 1.000 | 1.000 | — | — | |
| 2021 | 1.493 | 1.339 | +0.493 | +0.339 | ✓ |
| 2022 | 2.181 | 1.874 | +0.688 | +0.534 | ✓ |
| 2023 | 1.824 | 1.561 | −0.357 | −0.313 | ✓ |
| 2024 | 0.707 | 1.220 | −1.117 | −0.341 | ✓ |

All four directional transitions correct. Spearman ρ = 0.900 (vs 1.000 for the 2006 episode) reflects the 2024 magnitude divergence: the model predicts a deeper 2024 price fall (0.71×) than the World Bank annual average (1.22×). This gap mirrors the cobalt_2022 situation: the ODE's fringe supply mechanism and demand reduction shocks generate a structural oversupply that the actual annual average partially smooths, and Chinese HPAL producers limited additional exports to support market prices — a strategic output management not represented in the structural model.

The extended trajectory below spans the full Indonesian ban episode from 2019 (duration analysis base year = 2019):

| Year | Model (nsr-pure) | CEPII (factual) | Δ Model | Δ CEPII | Agree |
|------|-----------------|-----------------|---------|---------|-------|
| 2019 | 1.000 | 1.000 | — | — | |
| 2020 | 1.002 | 1.002 | +0.002 | +0.002 | ✓ |
| 2021 | 1.348 | 1.661 | +0.346 | +0.659 | ✓ |
| 2022 | 1.796 | 1.799 | +0.448 | +0.138 | ✓ |
| 2023 | 2.188 | 2.475 | +0.392 | +0.676 | ✓ |
| 2024 | 0.551 | 0.462 | −1.637 | −2.013 | ✓ |
| 2025 | 1.707 | 1.518 | +1.156 | +1.056 | ✓ |
| 2026 | 2.768 | 2.635 | +1.061 | +1.117 | ✓ |
| 2027 | 3.875 | 3.788 | +1.107 | +1.153 | ✓ |

The model achieves perfect directional accuracy across all eight transitions in this extended table. The model correctly predicts the 2024 price collapse, driven in the structural model by fringe supply entry and demand reduction shocks representing HPAL capacity flooding the market. The slight underestimate of 2021–2023 price levels (model 1.35–2.19 vs CEPII 1.66–2.48) reflects the speculative premium from the March 2022 LME nickel squeeze — a financial market microstructure event not present in the structural ODE.

---

## 8.5 L3 Duration Analysis

### 8.5.1 Abduction of Residual Trajectory

| Year | U_t (abducted) | Status | Interpretation |
|------|---------------|--------|---------------|
| 2019 | +0.000 | Baseline | |
| 2020 | −0.011 | ⚠ shock-active | COVID demand shock absorbs ore ban |
| 2021 | −0.225 | ⚠ shock-active | HPAL investment begun; model slightly overshoots |
| 2022 | +0.023 | Endogeneity-corrected | LME nickel squeeze (financial) |
| 2023 | −0.474 | Post-HPAL ramp | HPAL supply exceeds model prediction |
| 2024 | +0.905 | Post-collapse | Inventory rebuild; model undershoots recovery |

Endogeneity-corrected U_t (shock-year substitution):
- 2020: U_raw = −0.011 → U_corr = −0.119
- 2021: U_raw = −0.225 → U_corr = −0.237
- 2022: U_raw = +0.023 → U_corr = −0.356

The endogeneity correction is moderate for nickel (2020–2022 correction changes U values by 0.1–0.4), reflecting the fact that the nickel ban was a clearly dateable structural event whose effect on prices can be partially separated from the concurrent demand and financial dynamics.

### 8.5.2 Counterfactual Duration Table

Applying Pearl's L3 do-calculus — do(export_restriction ends year T):

| | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 2027 | Norm yr | (corr) |
|---|------|------|------|------|------|------|------|------|------|---------|--------|
| nsr-l3 | 1.000 | 1.959 | 1.431 | 4.423 | 1.976 | 7.330 | 0.306 | 1.429 | 4.680 | (ref) | |
| nsr-pure | 1.000 | 1.002 | 1.348 | 1.796 | 2.188 | 0.551 | 1.707 | 2.768 | 3.875 | | |
| factual | 1.000 | 1.002 | 1.661 | 1.799 | 2.475 | 0.462 | 1.518 | 2.635 | 3.788 | | |
| T=2020 | 1.000 | 1.959 | 1.905 | 4.166 | 1.966 | 7.373 | 0.302 | 1.411 | 4.640 | 2023 (+3yr) | never |
| T=2021 | 1.000 | 1.959 | 1.905 | 4.767 | 1.823 | 7.796 | 0.277 | 1.310 | 4.479 | 2025 (+4yr) | never |
| T=2022 | 1.000 | 1.959 | 1.905 | 4.767 | 2.400 | 4.718 | 0.394 | 1.764 | 5.022 | 2025 (+3yr) | never |
| T=2023 | 1.000 | 1.959 | 1.905 | 4.767 | 2.400 | 4.718 | 0.394 | 1.764 | 5.022 | 2025 (+2yr) | never |

From the cross-mineral duration ranking: factual end T=2022, normalisation at 2025, lag = +3yr.

### 8.5.3 Structural Interpretation: The HPAL Separation

The divergence between the full U_t column and the endogeneity-corrected column is the most analytically significant result in the nickel L3 analysis. Under full U_t, the T=2020 scenario normalises in 2023 (+3yr); under endogeneity-corrected U_t, it "never" normalises. This is because:

Under full U_t conditioning: the reference trajectory (nsr-l3) inherits the large U_t values from the actual HPAL supply flood (2023–2024 negative U_t), which drives the nsr-l3 trajectory below the factual, making convergence achievable.

Under endogeneity-corrected U_t: the correction removes HPAL-driven supply effects from the reference, leaving a reference that stays elevated and does not come down. Factual prices, which fell because HPAL arrived, cannot converge to a reference that assumes HPAL had not arrived.

This methodological result demonstrates that Pearl's L3 endogeneity correction is structurally important for the nickel case: whether or not HPAL supply expansion is treated as endogenous to the export restriction (it was triggered by the ban) determines whether the policy counterfactual shows normalisation or permanent divergence. This is not a model artefact — it is a genuine causal ambiguity about whether HPAL was the ban's consequence (endogenous) or an independent technology development (exogenous to the restriction parameter).

The thesis takes the conservative position: the ban is the sufficient cause of the HPAL investment decision (absent the ban, Chinese HPAL in Indonesia was commercially unviable due to ore access economics). Under this interpretation, the endogeneity-corrected U_t is the appropriate reference — the HPAL technology response is part of the ban's causal pathway, not an independent external event.

---

## 8.6 The HPAL Technology Disruption

The nickel case provides the only historical instance in this study where a supply restriction was ultimately reversed by a technology investment response with faster dynamics than the structural τ_K parameter implies. This finding has general implications for causal models of supply chain disruptions.

The key structural features that made HPAL re-routing possible were:
1. **Brownfield deployment**: HPAL facilities were built adjacent to existing ore bodies, not requiring new mine development. Construction time was 2–3 years vs τ_K = 7.5yr for greenfield mines.
2. **Capital availability**: Chinese state-backed industrial capital (CNGR, GEM, Huayou) could finance large HPAL investments without normal commercial return hurdles, compressing the investment decision timeline.
3. **Regulatory facilitation**: Indonesia's government actively facilitated Chinese HPAL investment as the intended policy outcome of the ban, removing permitting obstacles that would otherwise slow deployment.
4. **Scale of investment**: Approximately $5–8 billion of HPAL investment was committed within 18 months of the ban taking effect — an investment response rate with no historical precedent in critical minerals.

The causal model's τ_K parameter represents the geological and financial constraints on supply response for normal market actors operating without strategic facilitation. When a dominant-country government actively facilitates foreign investment to build processing capacity within its borders, the effective supply response time can be shortened by a factor of three or more.

For forward supply analysis, this means that the τ_K-based lag estimates for supply response should be treated as upper bounds when the restriction originates from a country that has both the ability and incentive to attract compensating investment. Indonesia's HPAL experience suggests that a Chinese export restriction on a processed material could similarly attract investment in compensating processing capacity — but only if the target countries (US, EU, Japan) have the regulatory agility and capital availability to replicate the Indonesian model, which has not yet been demonstrated.

---

## 8.7 Forward Projections

### 8.7.1 Scenario Design

Forward scenarios use the calibrated regime: τ_K = 7.514yr, α_P = 1.621, η_D = −0.495, g = 1.1679/yr. The hypothetical restriction posits a new Indonesian ore ban extension or tightening, combined with a restriction on HPAL output exports to Western buyers.

### 8.7.2 Price Trajectory Table

Price index (P / P_2024):

| Scenario | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 | 2030 | 2031 | Peak | Peak yr | Norm yr | Lag |
|----------|------|------|------|------|------|------|------|------|------|---------|---------|-----|
| baseline | 1.000 | 1.000 | 1.493 | 1.496 | 2.012 | 1.721 | 2.731 | 2.606 | 2.731 | 2030 | 2025 | — |
| mild_ban | 1.000 | 1.000 | 2.326 | 0.702 | 1.816 | 2.124 | 2.184 | 2.776 | 2.776 | 2031 | never | — |
| full_ban | 1.000 | 1.000 | 2.326 | 0.702 | 2.354 | 1.178 | 2.896 | 2.548 | 2.896 | 2030 | 2031 | +4yr |
| severe_ban | 1.000 | 1.000 | 3.070 | 0.793 | 3.030 | 2.100 | 1.753 | 3.053 | 3.070 | 2026 | 2032 | +4yr |

### 8.7.3 Key Findings

**Finding 1: FULL_BAN peaks at 2.90× in 2030 — moderate relative to cobalt (4.5×) and uranium (4.7×).** This reflects two offsetting factors: nickel's relatively lower US import reliance (40%) and higher demand elasticity (η_D = −0.495) limit the spike, but τ_K = 7.51yr ensures a long price scar.

**Finding 2: The HPAL precedent matters for the forward scenario.** If a future restriction on Indonesian nickel again triggered an HPAL-style technology response, the actual price peak would be lower than the ODE predicts — the model's τ_K = 7.51yr would overestimate recovery lag if brownfield HPAL deployment again compressed response time to 2–3 years. Forward projections should note this upper-bound qualification.

**Finding 3: Oscillatory dynamics under mild ban.** The MILD_BAN scenario shows a price collapse at 2027 (0.702) followed by recovery, reflecting the ODE's inventory-rebound dynamics when supply re-enters after a short restriction. This oscillation is characteristic of moderate-τ_K minerals (τ_K ~7yr) under short restrictions: the inventory rebuild overshoots, creating a temporary surplus before demand growth re-tightens the market.

---

## 8.8 Summary

Nickel presents the unique case of a supply restriction defeated by technology re-routing. Indonesia's 2020 ore export ban triggered a Chinese HPAL investment response that ultimately expanded global battery-grade nickel supply beyond pre-ban levels — a causal pathway that the ODE's static τ_K captures only in residual form.

The L3 analysis separates ban-driven price effects (structural parameters) from HPAL-technology-driven effects (U_t residuals), demonstrating the endogeneity of the technology response to the restriction decision. The policy implication is two-fold: first, restrictions on ore (unprocessed material) are more easily circumvented than restrictions on processed materials, because ore-level bans incentivise processing investment rather than just reducing supply; second, the circumvention response requires state-backed capital at scale and regulatory facilitation — conditions that may not hold for Western countries seeking to replicate the Indonesian HPAL model.

The forward projections place nickel at moderate structural risk (FULL_BAN peaks 2.9× vs cobalt 4.5× or uranium 4.7×), with a four-year post-restriction lag. The primary uncertainty is whether a future restriction would again trigger HPAL-style technology response — if so, forward projections overestimate the price peak and lag.
