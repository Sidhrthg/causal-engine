# Chapter 5: Case Studies in Supply Shock Causal Analysis

## 5.1 Overview

This chapter applies the causal inference framework developed in Chapters 3 and 4 to six critical minerals that expose distinct dimensions of US structural vulnerability. Each case follows a common template: market context and US exposure, calibrated structural parameters and their economic interpretation, episode-specific L3 counterfactual results, historical reserve context, and forward implications. Chapter 4 established that the model achieves mean DA = 0.917 across ten validation episodes; this chapter asks what those results mean for specific minerals and specific policy failures.

**Graphite** (Section 5.2) receives the primary treatment: it is the mineral with the highest US import reliance (100%), an active restriction already in place since October 2023, and the largest validated causal price effect in the dataset (+111.5 pp). **Rare earths** (Section 5.3) is the secondary case: the 2010 China export quota is a historically documented episode where a deliberate US policy decision — liquidating the National Defense Stockpile — created the vulnerability that China exploited. Together these two cases anchor the empirical argument. Section 5.4 presents four supporting cases (cobalt, lithium, nickel, uranium) that each illuminate a distinct structural mechanism.

---

## 5.2 Graphite: China's Dominant Position (Primary Case Study)

### 5.2.1 Market Context and US Strategic Exposure

Natural graphite is the material from which lithium-ion battery anodes are manufactured. Anode-grade spherical graphite requires purity above 99.95% carbon, a specification met almost exclusively through Chinese processing. As of 2023, China accounts for approximately 90% of global natural graphite exports by volume (USGS Mineral Commodity Summaries 2024) and controls the dominant share of downstream anode material processing. The United States has produced no domestic natural graphite since approximately 1990 and holds no strategic reserve.

US annual consumption of natural graphite was approximately 90,000 tonnes in 2022, with battery manufacturers (predominantly EV anode suppliers) comprising the fastest-growing demand segment. Unlike cobalt or lithium, for which non-Chinese supply chains exist, anode-grade graphite has no short-run substitute: synthetic graphite can replace it at three to five times the cost and with different electrochemical properties that require cell redesign. The US battery industry has no bridge technology for a supply disruption.

This creates the structural conditions for maximum vulnerability: single-source supply, no domestic alternative, no strategic reserve, inelastic demand from a growing sector, and a dominant supplier with both the motive and the mechanism to restrict exports.

### 5.2.2 Calibrated Parameters and Regime Interpretation

The causal engine is calibrated independently on two graphite episodes separated by fourteen years. The parameters that emerge encode the structural change in the market between episodes:

| Parameter | graphite_2008 | graphite_2022 | Economic interpretation |
|-----------|--------------|--------------|------------------------|
| α_P | 0.500 | **2.615** | Price adjustment speed: 5× faster in EV era |
| η_D | −0.073 | −0.777 | Demand elasticity: 10× more elastic pre-EV |
| τ_K | 8.276 yr | 7.830 yr | Mine development cycle: stable across regimes |
| g | 1.132/yr | 0.973/yr | Demand growth: EV surge followed by LFP correction |

The dominant finding is the α_P regime shift. In the pre-EV era (2008 episode), α_P = 0.500 indicates a market where prices adjust gradually and demand substitution buffers supply shocks. By the EV era (2022 episode), α_P = 2.615 — above the critical threshold of 1.5 identified in Chapter 4 as the signal that a mineral has entered an inelastic-demand, concentrated-supply regime. The product |α_P × η_D| = 2.031 exceeds 1 for the 2022 episode, indicating that the market's own price dynamics amplify shocks rather than attenuate them. τ_K remains constant at approximately 8 years: this reflects the geological and capital constraint on new mine development, which is invariant to demand regimes and confirms that the structural change is on the demand side (inelasticity from EV adoption), not the supply side.

This parameter signature — stable τ_K, rising α_P, rising |η_D| — is the fingerprint of a commodity transitioning from industrial input to strategic constraint. Cobalt showed the same transition in 2016; graphite showed it by 2022.

### 5.2.3 Episode 1: The 2008 Export Quota (Pre-EV Regime)

China progressively tightened graphite export quotas from 2009 through 2012, with the most restrictive measures in 2010–2011. The episode spans a pre-restriction demand surge (2008: +46% demand shock from steel/battery industry growth), the Global Financial Crisis demand collapse (2009: −40% macro demand shock), the export quota implementation (2010–2011: 35% quota reduction), and a simultaneous capex freeze (50% reduction in investment target) as investors pulled back.

The model achieves **DA = 1.000, Spearman ρ = 1.000** on this episode (Table 4.1), with an L3 counterfactual causal effect of **+58.1 pp** — the 2010–2011 quota turned what would have been a post-GFC cyclical recovery into a sustained structural price elevation. The high ρ = 1.000 indicates that not just the direction but the magnitude and timing of the price trajectory are correctly captured. The pre-EV α_P = 0.500 means price adjustment was gradual: the quota pushed prices up, but elasticity and inventory buffering prevented the extreme overshoot seen in the 2022 episode.

**Out-of-sample validation**: graphite_2008 parameters transferred to graphite_2022 (OOS transfer test) achieve DA = 0.600, correctly capturing the directional regime — the 2008 pre-EV parameters still signal price increases under export controls, even though they underpredict the amplification magnitude. This confirms that the directional mechanism (restriction → price up) is stable across regimes; α_P governs the magnitude but not the direction.

### 5.2.4 Episode 2: The 2022 Export Licence (EV Restriction Regime)

China's Ministry of Commerce implemented a graphite export licence system in October 2023, requiring exporters to obtain individual licences for spherical graphite and graphite electrodes. The stated rationale was environmental and national security protection. The effective result was a supply restriction: licence processing times introduced delays, and the licence regime provided a legal mechanism for selective denial without a formal quota.

The model achieves **DA = 1.000, Spearman ρ = 0.800** on this episode (calibration period 2021–2024), with an L3 counterfactual causal effect of **+111.5 pp** — the export licence system caused a price premium of over 100 percentage points relative to the counterfactual world where no licence system existed. The Spearman ρ = 0.800 (vs 1.000 for the 2008 episode) reflects a genuine model limitation: the 2022 price trajectory includes a demand destruction component from LFP battery adoption (which reduces anode graphite content per kWh) that partially offsets the restriction's price effect. The model captures both forces but imperfectly matches their timing.

The α_P = 2.615 in this episode is the largest price adjustment parameter in the dataset (joint first with cobalt). This means the graphite market, once the EV transition made demand inelastic, became one of the most price-reactive commodities studied. Any supply signal — restriction, inventory draw, geopolitical uncertainty — produces amplified price movement.

### 5.2.5 L3 Duration Analysis: How Long Does the Price Scar Persist?

The L3 duration analysis (Section 4.6.5, Table 4.9) addresses the operational policy question: *given that the restriction has been in place, if it ends at year T, how many additional years do prices remain elevated?*

**Table 5.2a — Graphite 2022: L3 normalisation lag by restriction end year T**

| Restriction ends (T) | Norm year (full U_t) | Norm year (endogeneity-corrected) | Lag |
|---|---|---|---|
| T = 2022 (never extended) | 2023 | 2023 | +1 yr |
| T = 2023 | 2027 | 2024 | +4 yr / +1 yr |
| T = 2024 | **2027** | **2026** | **+3 yr / +2 yr** |
| T = 2025 | 2027 | 2026 | +2 yr / +1 yr |

*The factual restriction end year is T = 2024 (benchmark). Full U_t uses all abducted residuals including restriction-period values that may be endogenous; endogeneity-corrected replaces restriction-year U_t with interpolated values. The true lag is in the range [+2, +3] yr depending on correction approach.*

The finding is that prices remain elevated for 2–3 years after the restriction ends, even if it were removed immediately. This carries forward damage from the restriction period: inventory depletion, deferred investment in alternative supply, and speculative dynamics that accumulated during the 2023 restriction period cannot be instantly reversed. With τ_K = 7.83 yr, capacity adjustment is slow; the market takes years to rebuild cover to the target ratio, keeping prices elevated throughout.

**Operational stockpile implication**: begin drawdown at restriction onset; sustain drawdown for τ_K/2 ≈ 4 years beyond restriction end; replenish only once prices return within 10% of the no-restriction baseline. For a 2-year restriction (2023–2024), this implies maintaining drawdown through approximately 2028.

### 5.2.6 Reserve Context: A Structural Zero

The US has never held a strategic graphite reserve. Natural graphite has not appeared in the National Defense Stockpile (NDS) objectives list since the 1990s. This is not a gap that emerged recently — it reflects a sustained policy judgement that graphite was a commodity-class industrial input, available from multiple global sources, that did not require strategic storage. That judgement was reasonable when US graphite consumption was primarily industrial (steel, lubricants, refractories) and supply was dispersed across China, North Korea, and smaller producers. It became structurally wrong as EV battery manufacturing concentrated both demand and supply: battery-grade flake graphite is sourced almost exclusively from China, and the processing for spherical anode material is overwhelmingly Chinese.

The only US inventory buffer at the time of the October 2023 licence implementation was processor-level industry stock — typically 60 to 90 days. This is roughly one-quarter of the model's I₀ = 50 kt (6-month) starting assumption. Under the actual near-zero reserve condition, the model's finding that cover* is breached in 2024 (lead time = 1 yr from a 6-month starting reserve) translates to an essentially immediate breach from the actual starting position. The export licence system was already the de facto supply shock; the US had no buffer to absorb it.

### 5.2.7 Trade Flow Circumvention: 6% is Not a Lifeline

A natural policy response to a Chinese restriction is to source graphite through third countries — purchasing Chinese-origin material after it has been re-exported through a non-restricted intermediary. The transshipment analysis (Chapter 3.4, CEPII bilateral flows 2019–2024) tests whether this is viable.

**Pre-restriction graphite flows (2022 baseline):**

| Route | Tonnes/yr | Share of China exports |
|-------|-----------|----------------------|
| China → USA (direct) | 40,710 | 18.9% |
| China → South Korea | 46,986 | 21.8% |
| China → Japan | 35,473 | 16.5% |
| China → Germany | 22,054 | 10.3% |
| China → Canada → USA | 2,285 | 1.1% |

Post-2023 statistical detection identifies one significant rerouting hub: Poland, with a +186% increase in Chinese graphite inflows post-restriction (p = 0.065, borderline significant). The estimated circumvention rate is **6%** of the nominal restriction volume (95% CI: [0%, 8.8%]). South Korea and Japan — which receive large volumes of Chinese graphite — are not circumvention hubs because they add genuine manufacturing value (anode processing) rather than simply re-exporting.

The conclusion is that China's export licence system is approximately **94% effective** against the US. Third-country routing cannot substitute for direct supply at any meaningful scale. The 40,710 t/yr direct flow is effectively severed by the restriction; the circumvention channel recovers at most 2,400 t/yr.

### 5.2.8 Forward Projection: Escalation from the 2025 Baseline

Using the L2 do-calculus projection (Chapter 6, Table 6.5), a standardised 30% restriction starting 2025 (FULL_BAN scenario, 3-year duration 2025–2027) produces a peak price index of **1.59× baseline** in 2026, normalising approximately 3 years after the restriction ends (2030). The SEVERE_BAN scenario (50%, 4 years) peaks at **2.02× baseline**.

These projections use Euler-stabilised α_P = 1.158 (capped from 2.615 for multi-year stability), which is the structural long-run speed. The short-run amplification that occurred in 2023–2024 (captured by the crisis-calibrated α_P = 2.615) would produce larger near-term spikes in a real event — the L2 projection is deliberately conservative for multi-year planning.

---

## 5.3 Rare Earths: The NDS Liquidation Case (Secondary Case Study)

### 5.3.1 Market Context and the Processing Bottleneck

Rare earth elements (REEs) are a group of 17 metals critical for permanent magnets (NdFeB for EV motors and wind turbines), catalysts (automotive catalytic converters), defence electronics (precision-guided munitions, radar), and optical materials. China accounts for approximately 60% of global mine output and 85% of global REE separation and processing capacity. The concentration at the processing stage — not the mining stage — is the strategic vulnerability: even REE ore mined in the US, Australia, or Africa must currently be processed in China or through Chinese-licensed technology.

The USGS Mineral Commodity Summaries 2024 reports US net import reliance for REEs at 14% — a figure that substantially understates the strategic risk. This low figure reflects growing US domestic mining (Mountain Pass, California, operated by MP Materials) and sourcing from Australia (Lynas). But the 14% net import figure measures material flows, not processing capacity. For the specific REEs most critical to defence and clean energy (neodymium, dysprosium, terbium), US processing capacity was near zero as of 2023; virtually all US-mined REE concentrate is shipped to China for separation. The effective processing reliance on China is approximately 80%.

This distinction — mine reliance vs. processing reliance — is the central structural feature of the REE market and is not captured in the ODE's τ_K parameter, which reflects China's own rapid capacity ramp (τ_K = 0.505 yr). The actual US recovery timeline is the processing buildout timeline: approximately 10–15 years for domestic separation and magnet manufacturing capacity to reach meaningful scale.

### 5.3.2 The 2010 China Export Quota

China progressively reduced REE export quotas from 2009, reaching a 40% reduction by 2011. The stated justification was environmental protection and resource conservation. In practice, quotas selectively disadvantaged non-Chinese manufacturers who could not access below-quota domestic REE prices, while Chinese downstream manufacturers retained access. This created an effective subsidy for Chinese REE-consuming industries (magnets, motors, catalysts) relative to global competitors.

The WTO ruled against China's export restrictions in 2014 (DS431, DS432, DS433), and quotas were formally eliminated in 2015. Prices crashed after the ruling: the speculative inventory built up during the restriction period was released, collapsing prices to pre-restriction levels by 2016.

The model is calibrated to this episode with **DA = 1.000, Spearman ρ = 1.000** — the strongest validation result in the dataset. This near-perfect fit reflects that the REE 2010 episode is the cleanest historical example of an export restriction in the dataset: a single dominant supplier imposing explicit supply restrictions with a documented start date, removal date, and clear global price response.

### 5.3.3 Calibrated Parameters: The τ_K Asymmetry

| Parameter | Value | Interpretation |
|-----------|-------|---------------|
| α_P | 1.754 (stabilised: 0.965) | High price sensitivity: China concentration + inelastic downstream demand |
| η_D | −0.933 | High demand elasticity: more substitution options than graphite (motor design flexibility) |
| τ_K | **0.505 yr** | Fast capacity adjustment — China's ramp within existing quota infrastructure |
| g | 1.084/yr | Pre-EV growth in clean-tech demand |

The τ_K = 0.505 yr requires careful interpretation. This does not mean that global REE supply recovers in six months after a restriction ends. It reflects the speed at which China can adjust production within its existing processing capacity — essentially, the time to ramp quotas back up or down within established operations. From a US policy perspective the relevant τ_K is approximately 10–15 years: the time to rebuild domestic separation capacity, develop allied processing (Lynas in Australia, MP Materials Phase 2 magnet production). The model correctly captures the market's price recovery speed after the WTO ruling (fast, because China's capacity came back online quickly) but does not model the structural US vulnerability, which is determined by a τ_K an order of magnitude larger.

This is the clearest example in the dataset of a model parameter that is *correct for the calibration target* (global price recovery speed) while *understating the policy-relevant quantity* (US domestic capability restoration time).

### 5.3.4 L3 Duration Analysis

**Table 5.2b — Rare earths 2010: L3 normalisation lag by quota end year T**

| Quota ends (T) | Norm year (full U_t) | Norm year (endogeneity-corrected) | Lag |
|---|---|---|---|
| T = 2010 | 2017 | 2011 | +7 yr / +1 yr |
| T = 2011 | 2012 | 2016 | +1 yr / +5 yr |
| T = 2012 | 2013 | never | +1 yr / — |
| T = 2013 | 2014 | 2016 | **+1 yr / +3 yr** |
| T = 2014 | 2015 | 2016 | +1 yr / +2 yr |

*Factual benchmark: T = 2013 (quota restrictions substantively ended). The large divergence between full U_t and endogeneity-corrected results reflects that 2010–2013 residuals are heavily endogenous to the restriction (speculative dynamics, WTO proceedings, inventory building were all restriction-driven). The corrected result (+3 yr at T=2013) is the preferred estimate.*

Despite τ_K = 0.505 yr (China's fast ramp), the L3 analysis finds a 3-year normalization lag at T = 2013. This seemingly contradicts the fast τ_K but is explained by the large endogenous residuals: the speculative inventory building and WTO uncertainty during 2010–2013 created a price trajectory that could not immediately normalise even after the restriction ended, because unwinding speculative positions takes time independent of physical supply recovery. The L3 abduction correctly captures this carry-forward from the restriction period; a simple L2 projection (which would predict fast recovery based on τ_K alone) would underestimate the post-restriction scar by approximately 2 years.

### 5.3.5 Reserve Context: A Policy-Caused Vulnerability

The US National Defense Stockpile (NDS) held rare earth materials as a Cold War strategic asset through the 1980s. Congress authorised progressive liquidation from 1993 onward as part of the post-Cold War defence restructuring — rare earths were removed from the NDS strategic objectives list on the explicit assumption that US domestic production (principally Mountain Pass) and allied supply were adequate substitutes for strategic reserves. The NDS held essentially no rare earth materials by 2004 (DoD NDS Annual Report to Congress FY2004; GAO-02-116).

This was a deliberate and documented policy choice, not an oversight. Mountain Pass mine had been the world's largest REE producer in the 1980s–1990s; its operator (Molycorp) placed it on care-and-maintenance status in 2002 due to Chinese competition and environmental compliance costs. The combination — NDS liquidated, Mountain Pass offline — left the US with no government buffer and no domestic production when China implemented quotas in 2010.

Congressional hearings in September 2010 documented the gap explicitly. Molycorp CEO Mark Smith testified: *"The United States has no stockpile of rare earth materials"* (Senate Energy and Natural Resources Committee, September 30, 2010). The model's finding that the REE episode requires >24 months of pre-positioned reserves is therefore not a forward-looking prescription — it quantifies the actual deficit that existed and was exploited in 2010. The NDS liquidation decision of the 1990s is the causal antecedent of the 2010 crisis, sixteen years later.

### 5.3.6 The Processing Bottleneck: What the Model Cannot Capture

The ODE model captures REE market dynamics accurately for the calibration target (global price, 2008–2014) but cannot represent the structural processing bottleneck that makes the US vulnerable independently of the mine-level supply constraint. Even in a world where the export quota is removed (the L2 counterfactual), US manufacturers who need separated REE oxides or alloyed NdFeB magnets cannot source them domestically: there is no US separation plant at scale and no domestic magnet manufacturing supply chain.

This means the model's DA = 1.000 and the L3 causal effects are valid for the global price market, but the US-specific vulnerability is larger than the model implies. A complete US vulnerability model for REEs would require: (a) a supply chain model capturing the processing stages between mine output and end-use material; (b) a separate capacity constraint for each processing stage; and (c) stage-specific τ_K values. This is beyond the ODE's scope. The model correctly establishes that REE restrictions cause large, persistent price increases globally; it understates how much of that increase directly and unavoidably affects US defence and clean-energy manufacturers who have no alternative sourcing channel.

---

## 5.4 Supporting Mineral Cases

### 5.4.1 Cobalt: DRC Demand Surge and Chinese Refining Lock-In

**Market context.** The Democratic Republic of Congo (DRC) accounts for approximately 70% of global cobalt mine production. China processes approximately 65% of global cobalt output. This creates a dual dependency: the US is exposed to DRC political instability at the mining stage and to Chinese processing concentration at the refining stage — neither can be addressed by sourcing from the other.

**Episode: cobalt_2016 EV speculation demand surge.** The 2016–2018 cobalt price cycle was driven by speculative buying ahead of anticipated EV battery demand, not by a supply restriction. The model is calibrated with α_P = 2.784 (highest in the dataset after graphite_2022), η_D = −0.542, τ_K = 5.750 yr. The high α_P reflects the combination of high supply concentration and speculative demand dynamics. Out-of-sample: cobalt_2016 parameters transferred to lithium_2016 achieve DA = 0.500 (random baseline), confirming these are episode-specific crisis dynamics rather than transferable structural parameters.

**L3 result.** The L3 analysis for cobalt_2016 shows "never normalises" within the 10-year window at benchmark T = 2018. This reflects large speculative U_t residuals during the 2016–2019 period that dominate the structural ODE recovery. The correct interpretation is that the cobalt price cycle was primarily speculation-driven, and the L3 framework correctly identifies that speculative dynamics (captured in U_t) extended the elevated price period beyond what the structural supply/demand model alone predicts.

**Policy note.** Cobalt is a HIGH tier vulnerability (V = 0.269) primarily because of the DRC/China dual dependency structure and τ_K = 5.75 yr mine development cycle. LFP battery chemistry reduces cobalt demand for passenger EVs but does not eliminate it for high-energy-density applications (defence electronics, aircraft, extreme-performance EVs). DoD Strategic and Critical Materials Reserve should maintain ≥1 year of cobalt above NDS requirements.

---

### 5.4.2 Lithium: Demand-Driven Cycle with Structural Resilience

**Market context.** Lithium's supply structure is fundamentally different from the other minerals in this study. Australia (55%) and Chile (23%) are the dominant producers, and neither has a history of using export restrictions as geopolitical leverage. The US has emerging domestic production (Thacker Pass, Nevada, projected to produce 80,000 t/yr LCE by 2030). τ_K = 1.337 yr for brine extraction (the dominant production technology in Chile) reflects the fast ramp time of evaporation pond chemistry relative to hard-rock mining.

**Episode: lithium_2022 EV boom.** The 2022 lithium price cycle was a demand-surge episode, not a restriction. The model achieves DA = 1.000, correctly predicting the price spike (2021–2022) and the crash (2023–2024). The crash is largely explained by the model's fringe entry mechanism: high prices (>1.1× P_ref) trigger HPAL-based fringe capacity entry, which floods the market by 2023–2024. This is essentially the same HPAL dynamic that drove nickel's crash (Section 5.4.3) — a technology-driven supply response that existing price models cannot predict from price history alone but that the ODE captures through the fringe_capacity_share parameter.

**L3 result.** The L3 analysis shows "never normalises" within the projection window at T = 2022, driven by large U_t residuals from the 2022 speculative bubble. As with cobalt, this is a demand-surge episode where speculation dominated structural dynamics. The L2 forward scenario (Table 6.5) shows lithium normalising +5 yr post a hypothetical 2025 restriction — not from recovery speed (τ_K is fast) but from baseline growth (g = 1.11/yr) pushing both baseline and restricted trajectories upward.

**Policy note.** Lithium is the LOWEST structural vulnerability (V = 0.059) — multiple source countries, fast τ_K, no dominant geopolitical restrictor, emerging domestic production. Standard IEA 90-day reserve is adequate.

---

### 5.4.3 Nickel: Technology Circumvention Defeats the Restriction

**Market context.** Indonesia accounts for 37% of global nickel production. The Philippines (13%) and Russia (~8%) are secondary suppliers. Unlike graphite or REEs, nickel's supply structure is genuinely diversified — no single country holds more than 40% of production. The US import reliance of approximately 40% reflects this diversification.

**Episode: nickel_2020 Indonesia ore export ban + HPAL response.** Indonesia banned unprocessed nickel ore exports in January 2020, the second such ban after a 2014 precedent. The stated goal was to develop downstream nickel processing capacity inside Indonesia. China responded by investing massively in High Pressure Acid Leaching (HPAL) technology in Indonesia — building laterite ore processing plants inside the country, thereby complying with the ban while capturing the downstream value. By 2023, Indonesian HPAL capacity was flooding global markets with battery-grade nickel, crashing prices from approximately $25,000/t (2022 peak) to $16,000/t (2024).

The model is calibrated to this period with α_P = 1.621, η_D = −0.495, τ_K = 7.514 yr. CEPII Indonesia data is used for L3 abduction. The L3 result is **+3 yr normalisation lag at T = 2022** — matching graphite's lag despite lower import reliance (40% vs 100%). Both minerals share τ_K ≈ 7.5 yr, confirming that τ_K, not import reliance, governs post-restriction price persistence.

**The key finding: technology circumvention.** The abducted U_t residuals for 2023–2024 are large and negative (the ODE model, parameterised on the ban dynamics, predicts elevated prices; CEPII data shows the actual HPAL-driven crash). The L3 framework isolates this: the U_t captures all the forces the structural model cannot represent, and the large negative residuals in 2023–2024 indicate that HPAL technology was the dominant force — not the ban, not demand destruction, but a technology-enabled supply response that circumvented the restriction's intent. This is the only case in the dataset where a major export restriction ultimately caused prices to fall below the pre-restriction level.

**Policy note.** Nickel's HPAL story is a rare positive example: market adaptation through technology investment can defeat the intended effect of an export restriction. The policy implication is the converse for US vulnerability assessment: minerals where no equivalent technology circumvention exists (graphite, REEs, uranium) carry higher structural risk precisely because the HPAL escape valve is not available.

---

### 5.4.4 Uranium: Contractual Buffer and the 20-Year Geological Cycle

**Market context.** Uranium is unique in this study in two respects: it has the longest mine development cycle of any mineral studied (τ_K = 14.9–20.0 yr), and it is the only one where a meaningful contractual buffer exists between supply disruption and US consumer impact. US nuclear utilities hold approximately 18–36 months of forward fuel under long-term contracts (EIA Uranium Marketing Annual Report, annual). Spot market purchases represent fewer than 10% of US utility uranium procurement.

Kazakhstan accounts for 43% of global uranium production. Russia's TENEX/Rosatom subsidiary supplied approximately 15% of US enrichment services pre-sanctions. The 2024 ADVANCE Act restricted most Russian uranium imports, making Kazakhstan and Canada the key replacement sources — both allied nations, neither of which has signalled intent to restrict exports.

**Episodes: uranium_2007 (Cigar Lake) and uranium_2022 (Russia sanctions).** The two episodes have strikingly different parameter signatures:

| Parameter | uranium_2007 | uranium_2022 | Interpretation |
|-----------|-------------|-------------|----------------|
| α_P | 2.476 | 0.890 | 2007: mine flood drove speculative panic; 2022: sanctions were gradual and anticipated |
| η_D | −0.436 | **−0.001** | 2022: near-perfectly inelastic (nuclear fuel has no substitute) |
| τ_K | **20.000 yr** | **14.886 yr** | Longest geological cycles in the dataset |
| g | 1.087/yr | 1.037/yr | Modest growth from nuclear Renaissance / new reactor builds |

The η_D = −0.001 for uranium_2022 is the most extreme demand inelasticity in the dataset. Nuclear fuel has essentially no short-run substitute: a reactor cannot switch fuels mid-cycle. This means any supply restriction above the contractual buffer translates directly to price escalation with essentially no demand-side dampening — the physics of the technology remove any elasticity. The L2 forward scenario reflects this: under FULL_BAN (30%, 3yr), uranium prices reach **4.74× baseline** by 2031 without normalising within the 2032 window.

**Reserve context: the contractual buffer that uranium uniquely possesses.** EIA uranium marketing data shows that US utilities held approximately 126–140 million pounds of U₃O₈ equivalent in forward cover (inventory plus contracted deliveries) entering 2007 — equivalent to approximately 24–30 months of reactor demand. When Cigar Lake flooded in October 2006 (removing ~18 million pounds per year, approximately 10% of world production) and spot prices rose from ~$15/lb to $88/lb by mid-2007, US utilities were largely insulated: fewer than 10% of fuel purchases were on the spot market, with the remainder under long-term contracts at price ceilings of $20–40/lb locked in years earlier (EIA Uranium Marketing Annual Report 2007). The 2007 spot spike — severe in global terms — did not translate into a US reactor fuel crisis.

This is the single case in the dataset where the historical reserve/contractual situation exceeded the model's I₀ = 50 kt (6 months) baseline assumption. The model's conservative I₀ substantially understates the actual uranium buffer. This explains why the ODE achieves DA = 1.000 on uranium_2007 despite a complex multi-shock structure: the utility-side insulation made the US demand-side response consistent with the model's inelastic η_D calibration, and the price trajectory was primarily driven by the supply removal and speculative spot buying — both of which the ODE captures through its structural equations.

**Forward caution.** The contractual buffer that protected the US in 2007 is finite. Long-term contracts have 3–7 year terms and must eventually be renewed. If the Russia sanctions persist and Kazakhstan becomes the dominant supplier, US enrichment infrastructure (centrifuge capacity for HEU/LEU conversion) becomes a secondary bottleneck: enrichment services, not just uranium ore, are concentrated among a small number of suppliers. The ADVANCE Act's domestic enrichment provisions (Centrus HALEU plant) address this, but τ_K = 14.9 yr means new enrichment capacity takes a decade to reach commercial scale.

---

## 5.5 Cross-Case Synthesis

The six case studies together establish three structural findings that the aggregate validation statistics in Chapter 4 cannot separately reveal:

**1. The α_P regime signal identifies transition points before crises occur.** Graphite's α_P shifted from 0.500 (2008) to 2.615 (2022) — a 5× increase that coincides exactly with the EV-era battery demand transition. Cobalt showed the same transition in 2016 (α_P = 2.784) seven years before graphite's export restriction. Nickel_2006 (α_P = 2.100) preceded the HPAL supply revolution. In each case, the α_P crossing the 1.5 threshold signals that a mineral has entered a regime where price shocks are amplified rather than attenuated. This signal is observable before restrictions occur and provides the structural early warning that Chapter 6's three-tier framework formalises.

**2. τ_K, not import reliance, determines how long elevated prices persist after a shock ends.** Graphite (+3 yr, τ_K = 7.8 yr) and nickel (+3 yr, τ_K = 7.5 yr) have the same normalisation lag despite graphite's 100% vs. nickel's 40% import reliance. Rare earths (+1–3 yr, τ_K = 0.5 yr China ramp) recover faster. Uranium (never, τ_K = 14.9–20 yr) never normalises within a decade. The geological investment cycle governs the price scar, not the geopolitical dependency. This finding separates the *onset* of vulnerability (import reliance, supply concentration) from its *duration* (τ_K, capacity adjustment), which have different policy responses.

**3. Reserve levels at shock onset determine whether model predictions match experienced impact.** The case studies reveal three distinct historical situations: zero reserve (graphite 2022, REE 2010) where the model's >24 month finding describes a deficit that existed; insufficient reserve (cobalt NDS, nickel) where partial mitigation occurred; and adequate contractual buffer (uranium 2007) where the spot shock was absorbed without US consumer impact. The ODE model captures what *would* have happened without reserves; the historical record shows what *did* happen given the actual reserve positions. Together they provide a validated baseline for future stockpile planning.

**4. Calibrated parameters are structurally identifiable from first-principles evidence, independent of price data.** The table below decomposes each structural parameter into its economic meaning and corroborating source across three episodes with the strongest independent documentation. This constitutes the "second line of validation" developed in Chapter 4 Section 4.9.4: the model satisfies not only trajectory accuracy (DA scores, Spearman ρ) but also parameter interpretability against sources that were never used in calibration.

| Parameter | Structural meaning | REE 2010 | Graphite 2022 | Uranium 2007 | Independent identification source |
|---|---|---|---|---|---|
| α_P | Price adjustment speed; high when demand inelastic and supply concentrated | 1.754 | **2.615** | 2.476 | EV demand share (Benchmark Mineral Intelligence); China processing HHI (USGS); demand-inelasticity literature for nuclear fuel and battery anodes |
| η_D | Demand elasticity; near zero for non-substitutable industrial inputs | −0.933 | −0.777 | −0.436 | No substitute for REE in permanent magnets (EC Critical Materials); no short-run anode substitute (Benchmark); nuclear fuel utility contracts lock in demand |
| τ_K | Mine-to-market development cycle (years); primarily geological and capital constraint | 0.505 (China's fast refinery ramp, not new mining) | 7.830 | **20.000** | China REE refinery expansion documented 2010–2012; graphite project timelines Tanzania/Mozambique 7–9 yr (USGS); Cigar Lake: 1981 discovery → 2014 first production |
| g | Demand growth rate per year (episode-specific; not transferred across episodes) | 1.084 | 0.973 | 1.087 | IEA EV demand trajectories; USGS consumption series; EIA nuclear generation data |
| I₀ | Effective inventory buffer at shock onset (sets model initial conditions) | **≈ 0** (NDS liquidated 1993–2006; Mark Smith Congressional testimony 2010) | **≈ 0** (no US production since 1990; no NDS holdings; USGS MCS 2020–2023) | **~24 months contractual** (EIA UMAR Table S1b: utility forward coverage 2005–2006) | USGS NDS Annual Reports; EIA Uranium Marketing Annual Report; USGS Mineral Commodity Summaries |

The I₀ row is the most policy-consequential entry. The two largest price impacts in the dataset — rare earths 2010 (+>200% historically; model DA = 1.000) and graphite 2022 (+111.5 pp causal effect) — both arose from episodes where I₀ ≈ 0. This was not an accident: in both cases, documented US government or industry decisions (NDS liquidation 1993–2006; no strategic graphite reserve maintained) set I₀ to near zero before the exogenous shock arrived. The uranium case shows the counterfactual: with I₀ ≈ 24 months contractual coverage, the 2007 spot spike did not pass through to US electricity consumers. The model's structural identification means these three regimes — zero buffer, partial buffer, adequate buffer — are recoverable from historical records and can inform prospective stockpile targets for future minerals before a crisis occurs.
