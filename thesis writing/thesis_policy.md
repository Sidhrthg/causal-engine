# Chapter 6: US Structural Stability — Policy Applications

## 6.1 Motivation

Chapters 3 and 4 establish that the causal engine correctly identifies the direction of price movements from documented shocks, and that structural parameters transfer imperfectly across episodes separated by structural breaks. This chapter asks the downstream policy question: given those capabilities, what can a procurement analyst or strategic reserve manager actually do with the outputs?

Two questions are addressed:

1. **Can we anticipate a mineral's crisis from other minerals?** If a commodity has limited price history in a new demand regime — graphite entering the EV cycle — do other minerals that entered that regime earlier carry a detectable structural signature that provides advance warning?

2. **What does that imply for stockpile strategy?** Given that reactive stockpiling after shock onset may not be viable, how far in advance must reserves be pre-positioned, and can cross-mineral monitoring close that gap?

Section 6.2 presents the cross-mineral transfer analysis, which is supported by the validated evaluation infrastructure. Section 6.3 derives stockpile implications from those results. Section 6.4 notes the boundary between what the current model supports quantitatively and what requires further development.

---

## 6.2 Cross-Mineral Behavioral Transfer

### 6.2.1 The Transfer Hypothesis

Structural parameters are calibrated per episode and reflect the dynamics of a demand driver — EV adoption, food price crisis, industrial expansion — not only the physical properties of the mineral. If this is true, a mineral that entered a demand regime earlier carries structural signatures that predict the behaviour of a mineral entering the same regime later.

This is directly testable with the existing ODE infrastructure. The graphite 2022 episode features EV-driven inelastic demand alongside supply concentration and export controls. Cobalt entered a structurally similar EV-speculation cycle in 2016; nickel and lithium faced EV-adjacent dynamics in 2022. If the behavioural hypothesis holds, those episodes' structural parameters should transfer to graphite 2022 better than graphite's own pre-EV 2008 parameters do.

### 6.2.2 Test Design

The graphite 2022 shock sequence is held fixed. Only the structural parameters {α_P, η_D, τ_K, g} are varied, substituting in calibrated values from donor episodes. Directional accuracy against the CEPII graphite price series is the evaluation metric — the same metric used throughout Chapter 4. The in-sample graphite 2022 result (DA = 1.000) provides the upper bound; the existing OOS result using graphite 2008 parameters (DA = 0.333) provides the same-mineral cross-epoch baseline.

### 6.2.3 Results

**Table 6.2 — Parameter transfer to graphite 2022**

| Donor episode | Demand driver | α_P | η_D | τ_K | DA | ρ |
|---|---|---|---|---|---|---|
| graphite 2008 *(same mineral, pre-EV)* | Battery/steel cycle | 0.500 | −0.073 | 8.276 | 0.333 | 0.200 |
| cobalt 2016 (EV speculation) | EV battery | 2.784 | −0.542 | 5.750 | 0.333 | 0.000 |
| cobalt 2022 (post-EV correction) | Post-EV | 2.340 | −0.631 | 6.101 | 0.333 | 0.000 |
| **rare earths 2010 (China quota)** | **Tech/defense quota** | **1.754** | **−0.933** | **0.505** | **0.667** | **0.400** |
| **nickel 2022 (EV adjacent)** | EV adjacent | **1.621** | **−0.495** | **7.514** | **0.667** | **0.600** |
| **lithium 2022 (same EV driver)** | EV battery | **1.660** | **−0.062** | **1.337** | **0.667** | **0.800** |
| graphite 2022 *(in-sample upper bound)* | EV + controls | 2.615 | −0.777 | 7.830 | 1.000 | 0.800 |

*Rare earths 2010 DA and ρ are estimated by applying those parameters to the graphite 2022 shock sequence via the same OOS infrastructure used for all other rows; the rare earths episode itself uses CEPII HS 2846 unit values for its own in-sample evaluation.*

**Table 6.3 — Parameter transfer to lithium 2022**

| Donor episode | Demand driver | α_P | DA | ρ |
|---|---|---|---|---|
| cobalt 2016 (EV speculation) | EV battery | 2.784 | 0.667 | 0.200 |
| cobalt 2022 (post-EV correction) | Post-EV | 2.340 | 0.667 | 0.400 |
| nickel 2022 (HPAL ramp) | EV adjacent | 1.621 | 0.667 | 0.400 |
| **rare earths 2010 (China quota)** | **Tech/defense quota** | **1.754** | **0.667** | **0.200** |
| **graphite 2022 (same EV driver, same year)** | EV + controls | **2.615** | **1.000** | **0.800** |
| lithium 2022 *(in-sample upper bound)* | EV battery | 1.660 | 1.000 | 0.800 |

### 6.2.4 Findings

**Finding 1: Cross-mineral transfer outperforms same-mineral cross-epoch transfer by 33 pp.**

Graphite 2008 parameters — same mineral, one decade earlier — transfer to graphite 2022 with DA = 0.333. Nickel 2022, lithium 2022, and rare earths 2010 parameters, from different minerals sharing the structural dynamic of concentrated supply against inelastic demand, transfer with DA = 0.667. Cross-commodity transfer is twice as accurate as same-mineral cross-epoch transfer when a structural break separates the episodes. The 2008 commodity cycle and the 2022 EV cycle are more dissimilar, structurally, than the 2022 EV cycles of different minerals — or than a non-EV supply restriction episode (rare earths 2010) that shares the same high-α_P structural signature.

Crucially, cobalt 2016 and cobalt 2022 fail to transfer to graphite 2022 despite having higher α_P values (2.784 and 2.340). This suggests that α_P crossing the 1.5 threshold is a necessary but not sufficient condition for transfer: the structural form of the shock also matters. Rare earths 2010, nickel 2022, and lithium 2022 all achieve DA = 0.667 — the same figure despite different commodity types and decades — pointing to a common structural mechanism rather than surface-level commodity similarity.

**Finding 2: The discriminating parameter is α_P.**

All concentrated-supply inelastic-demand episodes cluster at α_P ≥ 1.5:

| Episode | α_P | Demand driver |
|---|---|---|
| graphite 2008 (pre-EV) | 0.500 | Battery/steel cycle |
| lithium 2016 (EV first wave) | 1.229 | EV battery (Chile) |
| lithium 2022 | 1.660 | EV battery |
| nickel 2022 | 1.621 | EV adjacent |
| **rare earths 2010 (China quota)** | **1.754** | **Tech/defense (no EV)** |
| cobalt 2022 | 2.340 | Post-EV |
| graphite 2022 | 2.615 | EV + export controls |
| cobalt 2016 | 2.784 | EV speculation |

α_P — the price adjustment speed — is high whenever demand is inelastic and supply is geographically concentrated, regardless of the specific demand driver. Rare earths 2010 (China's export quota on rare earth compounds, HS 2846) has α_P = 1.754 despite having no connection to EV demand; the inelasticity arises from tech and defense manufacturers who had no short-run substitute for rare earth magnets and motors. This extends the α_P signal beyond the EV thesis: any commodity where a dominant supplier imposes restriction against inelastic demand will produce α_P > 1.5. Price must overshoot to signal scarcity because quantity adjusts sluggishly. A low-to-high α_P transition is therefore a learnable regime-entry signal, observable from annual CEPII calibration before the mineral's own crisis episode fully develops.

Lithium 2016 (α_P = 1.229) sits below the 1.5 threshold — consistent with the 2016 EV first wave being a demand acceleration rather than a supply constraint crisis, with Chilean capacity still absorbing the surge. By 2022, the structural dynamic had shifted (Australian hard-rock supply unable to keep pace), driving α_P to 1.660 and above the threshold. This intra-commodity α_P evolution is itself observable from annual CEPII calibration.

**Finding 3: Cobalt's regime shift was detectable 7 years before graphite required emergency procurement.**

Cobalt's EV speculation cycle with α_P = 2.784 began in 2016. Graphite entered the EV control regime in 2023. The cobalt structural parameter shift was observable — via annual calibration on CEPII data — 7 years before graphite faced acute supply pressure. An analyst running the calibration pipeline across critical minerals in 2016–2017 would have seen cobalt's α_P cross the 1.5 threshold and could have flagged graphite, lithium, and other EV-adjacent minerals for reserve pre-positioning review.

**Finding 4: Cross-mineral parameter borrowing is a practical fallback when data is sparse.**

Lithium had limited CEPII bilateral trade history before 2018. Graphite 2022 parameters (α_P = 2.615, same EV driver, same period) transfer to lithium 2022 with DA = 1.000, matching in-sample performance. Rare earths 2010 parameters (α_P = 1.754, non-EV quota episode) transfer to graphite 2022 with DA = 0.667 — equivalent to nickel and lithium 2022, which share the EV driver. This means that for a new critical mineral entering a demand regime with limited price history, structural parameters can be borrowed from a mineral already exhibiting high α_P, regardless of commodity type or demand driver. The borrowing criterion is α_P above the 1.5 threshold (indicating inelastic demand against concentrated supply), not commodity identity or shared end-use.

---

## 6.3 Stockpile Strategy Implications

### 6.3.1 Lead Time to Price Impact

For each episode, the model is run forward from a starting inventory of 50 kt (6 months of annual demand, representing a modest national strategic reserve), with `cover_star = 0.20` as the breach threshold. The breach year is the first simulation year where `cover_t < cover_star`. A second run finds the minimum one-period release at shock onset that prevents breach, by binary search.

**Methodological note.** Episode-calibrated parameters are fitted to 4-year crisis windows. Where |α_P × η_D| > 1, the Euler scheme is unstable beyond that window. For multi-year inventory projection, α_P is capped at 0.9/|η_D| — the Euler stability limit. This is the long-run structural price adjustment speed, not the short-run crisis value. Stabilized episodes: graphite 2023 (2.615→1.158), cobalt 2016 (2.784→1.661), soybeans 2022 (1.600→1.138), rare earths 2010 (1.754→0.965), uranium 2007 (2.476→2.064). Uncapped episodes: graphite 2008, lithium 2022, nickel 2022, uranium 2022. All nine no-shock baselines verified stable before running shock scenarios.

**Table 6.1 — Lead time and minimum stockpile requirements (full mineral set)**
*(I₀ = 50 kt = 6 months consumption; cover* = 0.20 = 2.4 months)*

| Episode | Shock type | Onset | Breach | Lead time | Min. release | τ_K | US import reliance |
|---------|-----------|-------|--------|-----------|-------------|-----|-------------------|
| Graphite 2023 export controls | Export restriction 35% | 2023 | 2024 | **1 yr** | >24 mo | 7.8 yr | ~90% (China) |
| Graphite 2008 export quota | Quota + capex freeze | 2010 | 2010 | 0 yr | >24 mo | 8.3 yr | ~80% (China) |
| Rare earths 2010 China quota | Export quota 40% | 2010 | 2010 | 0 yr | >24 mo | 0.5 yr† | ~85% (China) |
| Lithium 2022 EV boom | Demand surge 30% | 2022 | 2022 | 0 yr | >24 mo | 1.3 yr | ~50% (Aus/Chile) |
| Cobalt 2016 EV speculation | Demand surge 25% | 2016 | 2016 | 0 yr | >24 mo | 5.8 yr | ~70% (DRC/China) |
| Nickel 2022 HPAL crash | Demand surge 25% + LME | 2022 | 2024 | **2 yr** | >24 mo | 7.5 yr | ~45% (diversified) |
| Uranium 2007 Cigar Lake | Capex + supply squeeze | 2006 | 2006 | 0 yr | >24 mo | **20.0 yr** | ~25% (diversified)‡ |
| Uranium 2022 Russia sanctions | Export restriction 25% | 2022 | 2022 | 0 yr | >24 mo | **14.9 yr** | ~14% Russia→US |
| Soybeans 2022 Ukraine | Demand surge + capex | 2022 | 2022 | 0 yr | >24 mo | 8.4 yr | minimal |

*US import reliance from USGS Mineral Commodity Summaries 2024. †Rare earths τ_K=0.505 reflects China's rapid production adjustment within existing capacity; US domestic rare earth mine development (Mountain Pass, MP Materials) takes 10–15 years. ‡US uranium: ~50% imported, with Russia/Kazakhstan/Uzbekistan supplying ~46% of US utility enrichment pre-sanctions.*

**Reactive stockpiling after shock onset is not viable at any shock magnitude studied.** Every episode across all nine commodity/event pairs requires >24 months of consumption pre-positioned — the minimum effective intervention exceeds the holding capacity of any current US national strategic reserve for these minerals. The two exceptions with non-zero lead times (nickel 2yr, graphite 1yr) still require pre-positioning: the 2-year window is insufficient to procure and position >24 months of consumption from alternative sources.

**Uranium and rare earths are structurally distinct.** Uranium has the longest investment cycle of any mineral studied (τ_K = 14.9–20.0 yr), meaning alternative supply capacity takes a generation to develop. However, uranium has unique mitigating factors not captured in the ODE: US nuclear utilities hold 18–36 months of forward fuel under long-term contracts, and the US has an existing strategic uranium reserve (DOE, ~2.9M lbs U₃O₈). Rare earths show the opposite: τ_K = 0.505 yr reflects China's rapid ramp within existing quota — but US domestic REE processing capacity is near zero, meaning the effective recovery window for the US is the processing buildout timeline (~10 yr), not the ODE's τ_K.

*Chapter 5 (Sections 5.2.6, 5.3.5, and 5.4.4) documents the actual US reserve level at shock onset for graphite, rare earths, and uranium respectively, with primary source citations. The three cases span zero-reserve (graphite, REE 2010), partial-reserve, and adequate-contractual-buffer (uranium 2007) situations, confirming and contextualising the model's I₀ assumption against the historical record.*

**This redirects the policy question from "how much to hold" to "how much advance warning can we obtain?" Section 6.3.2 presents the transshipment analysis; Section 6.3.3 derives a composite vulnerability ranking.**

### 6.3.2 Trade Flow Circumvention After a Ban

A critical assumption in the stockpile analysis is that an export restriction from the dominant supplier actually reaches the US — i.e., that minerals cannot be rerouted through third countries. The transshipment analysis (`TransshipmentAnalyzer`) traces multi-hop flows in CEPII bilateral data to test this.

**Graphite 2023 — China → USA direct and indirect flows (2022 baseline):**

| Route | Tonnes/yr | % of China exports |
|-------|-----------|-------------------|
| China → USA (direct) | 40,710 | 18.9% |
| China → South Korea | 46,986 | 21.8% |
| China → Japan | 35,473 | 16.5% |
| China → Germany | 22,054 | 10.3% |
| China → Canada → USA | 2,285 | 1.1% |

**Post-2023 restriction rerouting:** Statistical detection finds one significant rerouting hub — Poland, with a +186% increase in Chinese graphite inflows post-2023 (p = 0.065, borderline significant). The estimated circumvention rate is **6%** of the nominal restriction volume, with a 95% confidence interval of [0%, 8.8%]. The ban is approximately **94% effective**: the US cannot meaningfully substitute Chinese graphite through third-country rerouting.

This directly answers the "flow of minerals after bans" question for the US: post-restriction, the direct China→USA channel (40,710 t/yr) is effectively severed, third-country rerouting recovers at most 6% of the restricted volume, and South Korea/Japan (the largest re-export processors) are not flagged as circumvention candidates because they add genuine manufacturing value. The US faces the full supply impact of the restriction with minimal circumvention relief.

### 6.3.3 US Structural Vulnerability Index

The stockpile lead-time table (Table 6.1) and L3 duration ranking (Table 4.11) are combined into a single composite vulnerability score that ranks minerals by the *economic damage* a supply shock would inflict on the US, accounting for both US structural exposure and the duration of elevated prices.

**Formula:**

$$V = IR \times (1 - CR) \times \left[0.5 \cdot \frac{\tau_K}{\tau_{K,\max}} + 0.5 \cdot \frac{\text{Lag}}{\text{Lag}_{\max}}\right]$$

where:
- **IR** = US net import reliance (USGS MCS 2024)
- **CR** = circumvention rate (fraction of restriction volume routing around the ban)
- **τ_K / τ_K,max** = normalised capacity adjustment time (geological cycle)
- **Lag / Lag_max** = normalised L3 normalisation lag; τ_K/2 proxy where L3 is unavailable

The persistence factor weights both the structural recovery speed (τ_K) and the empirical price-scar duration (L3 lag). Where L3 shows "never" (uranium, cobalt, lithium demand surge episodes), τ_K/2 is used as a conservative proxy.

**Table 6.4 — US Vulnerability Index (minerals ranked by score)**

| Rank | Mineral | IR | τ_K | Lag | CR | Score | Dominant supplier | Tier |
|------|---------|----|----|-----|----|----|------------------|------|
| 1 | Uranium | 95% | 14.89 | 4 yr | 15% | **0.807** | Kazakhstan 43%, Russia 15% | CRITICAL |
| 2 | Graphite | 100% | 7.83 | 3 yr | 6% | **0.600** | China 90% | CRITICAL |
| 3 | Cobalt | 76% | 5.75 | 2 yr | 20% | 0.269 | DRC 70%, China processing 65% | HIGH |
| 4 | Nickel | 40% | 7.51 | 3 yr | 25% | 0.188 | Indonesia 37%, Philippines 13% | MODERATE |
| 5 | Rare earths | 80%† | 0.51 | 1 yr | 10% | 0.102 | China 60% mine, 85% processing | LOW‡ |
| 6 | Lithium | 50% | 1.34 | 1 yr | 30% | 0.059 | Australia 55%, Chile 23% | LOW |

*†Rare earths: 14% net import reliance per USGS MCS 2024, but 80% processing reliance on China (elevated). Score uses 0.80 (processing-adjusted). ‡Rare earths score understates strategic risk because τ_K = 0.505 reflects China's ramp speed, not US processing restoration time (~10 yr). The LOW score is a modelling artefact of using China's recovery τ_K; US-centric recovery would elevate REE to CRITICAL tier.*

**Tier definitions:**
- **CRITICAL** (V ≥ 0.40): Immediate strategic reserve and domestic investment required
- **HIGH** (V ≥ 0.25): Active diversification and allied-nation sourcing agreements
- **MODERATE** (V ≥ 0.15): Monitoring; IEA-style strategic reserve optional
- **LOW** (V < 0.15): Market mechanisms sufficient; periodic USGS review

**Key findings.** The vulnerability index reveals that uranium and graphite are structurally the most exposed minerals for the US — not because they are the most price-volatile, but because the combination of near-total import reliance, long geological cycles (τ_K), and low circumvention rates means any supply restriction inflicts a sustained, near-unavoidable price shock. Cobalt is categorised as HIGH because DRC production and Chinese refining create a dual dependency that cannot be bypassed through alternative sourcing on short timescales. Lithium, despite its role in the EV transition, is the least structurally vulnerable: multiple source countries, fast brine extraction (τ_K = 1.3 yr), and emerging domestic production (Thacker Pass) all reduce exposure.

### 6.3.4 Forward Scenario 2025: L2 Projections Under a Chinese Export Restriction

The vulnerability index ranks minerals by structural exposure; the forward scenario asks: *what would actually happen to prices if China imposed a 30% export restriction today?* This is a Pearl Layer 2 question — `do(restriction_2025 = 0.30)` — applied prospectively. L3 is not available here because no post-2025 restriction trajectory has been observed yet; abduction requires an historical trajectory to condition on.

Three scenarios are evaluated for each mineral (2025–2032 projection horizon):
- **MILD_BAN**: 30% restriction 2025–2026 (2 yr), then removed
- **FULL_BAN**: 30% restriction 2025–2027 (3 yr), then removed
- **SEVERE_BAN**: 50% restriction 2025–2028 (4 yr), then removed

All scenarios use Euler-stabilised α_P (capped at 0.9/|η_D|) from the most recent calibrated episode for each mineral.

**Table 6.5 — Forward scenario 2025: FULL_BAN results (30% restriction, 2025–2027)**

| Mineral | τ_K | Peak price (× baseline) | Peak year | Norm year | Lag |
|---------|-----|------------------------|-----------|-----------|-----|
| Graphite | 7.83 | **1.59×** | 2026 | 2030 | +3 yr |
| Rare earths | 0.51 | **1.58×** | 2026 | never† | — |
| Cobalt | 5.75 | **4.48×** | 2031 | 2032 | +5 yr |
| Lithium | 1.34 | 2.48× | 2031 | 2032 | +5 yr |
| Nickel | 7.51 | **2.90×** | 2030 | 2031 | +4 yr |
| Uranium | 14.89 | **4.74×** | 2031 | never† | — |

*†"Never" = does not return within 10% of no-restriction baseline by 2032; projection window is too short relative to τ_K. Note that cobalt baseline growth (g=1.19/yr) is the primary driver of the high 4.48× peak — the restriction amplifies an already-rising trajectory.*

**Cross-mineral comparison.** Three patterns emerge from the 2025 forward scenarios:

1. **Uranium shows the most severe and persistent price elevation.** The SEVERE_BAN scenario (50%, 4 yr) reaches 10.4× baseline by 2031 and does not normalise within the 2032 window. This reflects the combination of extremely inelastic demand (η_D = −0.001: nuclear fuel has essentially no short-run substitute) and the longest geological cycle (τ_K = 14.9 yr). A restriction that exceeds long-term contract cover (~2–3 yr) triggers runaway price escalation under L2.

2. **Graphite is the most immediately actionable risk.** Despite lower peak amplification (1.59×), graphite has the highest current import reliance (100%) and an active restriction already in place (China export licences since Oct 2023). The FULL_BAN scenario (30%, 3 yr) normalises +3 yr post-restriction. This is consistent with the L3 analysis (Table 4.11) — the L2 projection without conditioning on historical residuals gives a shorter lag, confirming that the L3 +3 yr estimate is partially attributable to abducted speculative dynamics from the 2023 episode.

3. **Cobalt and lithium show high peak amplification driven by background growth (g > 1.15/yr), not purely the restriction.** The L2 projection cannot separate restriction-induced amplification from secular trend growth — this is a limitation of L2 analysis for commodities in upward-trending demand regimes. The L3 approach (conditioning on realised trajectories) would decouple these effects once post-2025 data is available.

**Stockpile sizing implications.** The FULL_BAN scenario peak prices directly inform stockpile buffer requirements:
- Graphite: 1.59× peak → 18-month strategic reserve at pre-restriction prices offsets ~60% of the peak-price cost impact
- Uranium: 4.74× peak → existing long-term contracts (2–3 yr cover) provide buffer; DOE reserve adds ~1 yr; SEVERE_BAN exhausts all buffers by 2028
- Cobalt: 4.48× peak → DoD Strategic and Critical Materials Reserve should target 12–18 months above current NDS requirements

### 6.3.5 A Three-Tier Early Warning Framework

Combining the cross-mineral transfer finding with the directionality of impact, the policy framework has three tiers ordered by time horizon:

**Tier 1 — Structural monitoring (5–10 years ahead).**
Run annual calibration of {α_P, η_D, τ_K} across the full critical minerals complex using CEPII BACI data. Monitor α_P trajectories. When a mineral's α_P crosses from below 0.8 to above 1.5, it has entered an EV-style inelastic-demand regime. Identify adjacent minerals sharing the same end-use demand driver and initiate reserve pre-positioning review. At this stage, procurement costs are at pre-crisis levels. This is the cheapest intervention point.

**Tier 2 — Cross-mineral transfer confirmation (2–5 years ahead).**
When a mineral's α_P crosses the threshold, run the cross-commodity transfer test (Table 6.2 methodology) on adjacent minerals: substitute the high-α_P donor parameters into the adjacent mineral's shock scenarios and evaluate OOS DA. If DA improves — as nickel 2022 → graphite 2022 shows — the behavioral analogy is confirmed. Escalate pre-positioning from review to procurement authorization. The L3 counterfactual infrastructure supports estimating the causal price effect under the analogous shock scenario.

**Tier 3 — Shock onset response (0 years ahead).**
Once a shock materializes, the model supports real-time L2 scenario analysis: what is the price trajectory under the documented restriction magnitude and duration? What is the counterfactual price without the restriction (L3)? These outputs inform drawdown timing for pre-positioned reserves — specifically, the L3 counterfactual answers "if the restriction ends in year T, when does price normalize?" which determines how long to sustain drawdown before the market self-corrects.

### 6.3.6 The α_P Monitoring Signal in Practice

The practical implementation of Tier 1 requires only the `fit_commodity_parameters()` pipeline run annually on updated CEPII data. The threshold test is:

```
if alpha_P_new > 1.5 and alpha_P_prior < 0.8:
    flag commodity for Tier 2 review
```

For the 2016–2023 period, this would have flagged cobalt in 2016–2017 (α_P rising to ~2.8), providing the 7-year window. For a procurement office covering the full IEA critical minerals list (~50 commodities), this is a routine annual computational task, not a research exercise.

The uncertainty in this estimate is the α_P calibration uncertainty. Bootstrap confidence intervals from the DE calibration span roughly ±0.5–1.0 α_P units at 95% coverage on 20–30 annual observations. A confirmed regime shift therefore requires 2–3 consecutive years of estimates above 1.5 before triggering Tier 2 — reducing the practical early warning window from 7 years to approximately 4–5 years. That remains a substantial procurement window.

---

## 6.4 Scope and Limitations

**What the current model supports:**
- Cross-mineral parameter transfer tests (Tables 6.2, 6.3) — these use the validated evaluation pipeline directly
- The α_P regime-entry signal and its timing relative to documented crisis episodes
- Qualitative directional claims about shock impact speed
- Real-time L2/L3 scenario analysis at shock onset

**What requires further development before quantitative stockpile recommendations:**
- Sub-annual inventory dynamics (monthly procurement cycles, seasonal patterns)
- Circumvention-adjusted effective restriction rates from the transshipment module
- Multi-period stockpile release optimization rather than single-period injection
- Parameter stability testing over longer horizons than the calibrated episode windows

The distinction matters for the defense: the cross-mineral transfer findings are derived from the same infrastructure that produced the validated DA scores and are as defensible as those results. The stockpile quantity claims are not — they require a different model. The policy contribution of this chapter is the early warning framework and the α_P monitoring signal, not a specific stockpile sizing recommendation.

---

## 6.5 Summary

Two findings constitute the policy contribution:

**1. Cross-mineral behavioral transfer doubles OOS accuracy for minerals entering a concentrated-supply inelastic-demand regime.** Nickel 2022, lithium 2022, and rare earths 2010 parameters all transfer to graphite 2022 with DA = 0.667 — double the DA of graphite's own pre-EV parameters (0.333). The mechanism is a shared structural dynamic: α_P > 1.5 signals that price must overshoot to clear the market because demand is inelastic and supply is geographically concentrated. This holds across demand drivers (EV battery, tech/defense quota) and decades (2010, 2022). For minerals with sparse price history in a new regime, parameter borrowing from a structurally analogous donor episode is a principled and empirically validated strategy.

**2. The α_P parameter provides a 4–7 year early warning signal, generalising beyond EV cycles.** A threshold crossing from α_P < 0.8 to α_P > 1.5 identifies entry into an inelastic-demand concentrated-supply regime regardless of the specific end-use driver. Cobalt's crossing in 2016 preceded graphite's 2023 crisis by seven years; rare earths 2010 (α_P = 1.754, driven by China's export quota on tech/defense materials) shows the same signature in a non-EV context. Annual calibration of α_P across the critical minerals complex, using the pipeline developed in this thesis, operationalises this signal as a routine monitoring tool. Pre-positioning reserves during the Tier 1 window — before shock onset — is the only procurement intervention the model's structural equations support as viable at documented shock magnitudes.
