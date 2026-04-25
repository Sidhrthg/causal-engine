# Causal Engine — Research Questions and Model Extensions

## How to read this document

Each question below is a genuine L2 or L3 causal query the model can answer
using existing infrastructure (ODE {K, I, P}, CEPII validation, Pearl hierarchy).
Questions are grouped by mineral and by question type:

- **Stockpile policy** — do(stockpile_release = X) scenarios
- **Supply chain resilience** — do(allied_processing_share = X) or do(circumvention_rate = X)
- **Demand side** — what if η_D changes, or demand growth g shifts?
- **Sensitivity** — how robust are results to parameter uncertainty?
- **New shock types** — capex shock, demand collapse, tariff vs quota

For each mineral the question is stated in Pearl L2/L3 form, then translated
into a policy question a policymaker would actually ask.

Cross-mineral applicability is noted where a question structure transfers
directly to other minerals — including germanium and gallium (Section 7).

---

## 1. GRAPHITE

### 1.1 Stockpile policy
**Causal question (L2):**
do(stockpile_release = R_t) for years t = 2026, 2027, 2028 under a 50% ban.
What release schedule R_t minimises the peak price index?

*Policy translation:* "If DoD establishes a 4-year graphite reserve, should
we front-load releases in year 1 of the ban, or spread them evenly?"

**Causal question (L2):**
What is the minimum strategic reserve (in months of consumption) such that
a 100% ban never causes prices to exceed 1.5× baseline?

*Policy translation:* "How much graphite does the NDS need to stockpile to
keep battery costs stable under a complete Chinese export ban?"

---

### 1.2 Supply chain resilience
**Causal question (L2):**
do(allied_processing_share = 0.20) by 2028 (Poland + Mozambique anode
processing).
Does the 5-year affected period reduce, and by how many years?

*Policy translation:* "If IRA subsidies bring 20% of anode processing outside
China by 2028, does that meaningfully shorten the disruption window?"

**Causal question (L2):**
At what allied processing share does the amplification regime (α_P ≥ 1.5)
break — i.e., the supply concentration drops enough that price signals no
longer amplify?

*Policy translation:* "What processing diversification threshold actually
changes the structural vulnerability, not just reduces peak by a few percent?"

---

### 1.3 Demand side
**Causal question (L2):**
do(silicon_anode_adoption = 0.15 by 2028): if 15% of EV batteries shift to
silicon anodes, reducing graphite demand growth g from 0.97 to ~0.80/yr,
does that break out of the amplification regime under a 50% ban?

*Policy translation:* "Can technology substitution (silicon anodes) solve the
graphite problem, or does China's processing dominance make it irrelevant?"

---

### 1.4 Sensitivity
**Causal question:**
τ_K calibrated at 7.83yr. How do the "years affected" results change across
τ_K ∈ {5, 7.83, 10, 15} years?

*Policy translation:* "How confident are we in the 5-year affected period
given that mine development timelines have historical variance?"

---

### 1.5 New shock type
**Causal question (L2):**
do(capex_shock = 0.5, 2026–2027): what if a major Mozambican or Chinese mine
disaster removes 20% of supply for 2 years — no policy trigger, pure accident.
How does the price path differ from the export restriction scenario?

*Policy translation:* "Is an accidental supply disruption more or less damaging
than a deliberate export restriction of the same magnitude?"

**Transfers to:** REE (Mountain Pass accident), uranium (Cigar Lake-style flood),
cobalt (DRC mine accident), nickel (Philippine typhoon).

---

## 2. RARE EARTH ELEMENTS

### 2.1 Stockpile policy
**Causal question (L2):**
What DoD NDS stockpile level (in years of separated oxide consumption) eliminates
the price spike under a 75% ban — the threshold at which "never normalises"
becomes "normalises within 5 years"?

*Policy translation:* "What should Congress appropriate for the REE NDS to
provide meaningful protection against a near-total Chinese export restriction?"

---

### 2.2 Supply chain resilience
**Causal question (L2):**
do(mp_materials_processing_share = 0.12 by 2027): MP Materials Phase 2
separation covers ~12% of US NdFeB magnet demand.
Does the never-normalises result become bounded?

*Policy translation:* "Is the DoD Section 232 offtake with MP Materials
sufficient to change the structural outcome, or is it too small?"

**Causal question (L2):**
do(japan_separation_share = 0.15): Japan's Sumitomo/Toyota Tsusho separation
capacity allocated to US supply under a bilateral agreement.
Combined with MP Materials, does allied processing exceed the stabilisation
threshold?

*Policy translation:* "Does the US-Japan Critical Minerals Agreement 2023
provide real supply security or marginal diversification?"

---

### 2.3 Demand side
**Causal question (L2):**
At what ban magnitude does ferrite magnet substitution (replacing NdFeB in
EV motors) become commercially viable (assumed at 2.5× reference price)?
How fast does that demand substitution bring prices down?

*Policy translation:* "Is the EV industry's dependence on rare earth magnets
structurally permanent, or does price pressure trigger a technology response?"

---

### 2.4 China ramp-speed question
**Causal question (L3):**
Given τ_K = 0.51yr (China can ramp production back in ~6 months), what is
the minimum restriction duration for China to cause lasting damage before
flooding the market to reassert dominance?

*Policy translation:* "Is a short Chinese REE ban (6 months) a credible
weapon, or does China's own ramp speed make it self-defeating?"

---

## 3. COBALT

### 3.1 Oversupply buffer question
**Causal question (L3):**
Given cobalt is in 2024 oversupply (U_2024 = −2.128), in what year does
the current oversupply buffer expire — i.e., at what year would a ban cause
immediate price spikes rather than first absorbing the inventory glut?

*Policy translation:* "Do we have a window to pre-position reserves before
cobalt oversupply exhausts and the market tightens again?"

---

### 3.2 LFP transition
**Causal question (L2):**
do(lfp_market_share = 0.60 by 2027): if LFP reaches 60% EV deployment,
the effective η_D rises from −0.542 toward −0.80 for passenger EV segment.
Does the never-normalises result change?

*Policy translation:* "Is the cobalt supply risk self-solving via battery
chemistry transition, or does defence/aviation demand maintain the vulnerability
regardless?"

---

### 3.3 Zambia diversification
**Causal question (L2):**
do(zambia_share = 0.15 by 2028): Chambishi Metals + Mopani expansion covers
15% of global cobalt.
Does DRC concentration drop below the binding threshold?

*Policy translation:* "What Zambia investment level is required to break DRC's
structural grip on cobalt supply?"

---

### 3.4 NDS drawdown interaction
**Causal question (L2):**
Given an 8-year affected period under a 50% ban: what continuous drawdown
rate from an NDS cobalt reserve suppresses prices below 2× baseline throughout
the entire 8-year window?

*Policy translation:* "What NDS cobalt reserve size and release rate provides
adequate protection for the full disruption duration?"

---

## 4. LITHIUM

### 4.1 Thacker Pass question
**Causal question (L2):**
do(thacker_pass_online = 2027, production_share = 0.08): Thacker Pass
(Nevada) covers ~8% of US consumption from 2027.
Under a 50% ban starting 2026, does domestic production reduce the >9yr
affected period?

*Policy translation:* "Does the US lithium mine investment pipeline change
the strategic calculus, or is it too small relative to demand growth?"

---

### 4.2 Processing restriction vs. mine restriction
**Causal question (L2):**
do(china_lioh_processing_restriction = 0.50): China restricts lithium hydroxide
conversion (the processing stage it dominates at 70%) rather than mine output.
Compare: does a processing restriction produce worse outcomes than a mine
restriction of the same magnitude?

*Policy translation:* "Should the US focus security investment on processing
capacity (IRA 45X) or mine diversification — which bottleneck is more dangerous
if exploited?"

**Transfers to:** graphite (processing vs. mine), REE (separation vs. mining),
cobalt (refining vs. DRC mine). This is the central structural question for
all five Chinese-processing-dominant minerals.

---

### 4.3 η_D threshold
**Causal question (L2):**
At what ban magnitude is demand inelasticity (η_D = −0.062) so extreme that
no demand substitution mechanism can prevent prices exceeding 3×?
Is there a safe ban threshold below which structural demand elasticity absorbs
the shock?

*Policy translation:* "Is there any 'small enough' lithium restriction that
the market can absorb without a severe price spike?"

---

## 5. NICKEL

### 5.1 HPAL technology replication
**Causal question (L2):**
do(hpal_response_tau = 2.5yr): if Western companies can replicate Indonesia's
HPAL investment model (2–3yr brownfield construction rather than 7.5yr
greenfield), how does the 6-year affected period change?

*Policy translation:* "Can the US build 'shovel-ready' HPAL capacity that
compresses our effective τ_K for nickel?"

---

### 5.2 Import reliance threshold
**Causal question (L2):**
At what US nickel import reliance level (currently 40%) does a 50% ban have
negligible consumer price impact (< 1.2× baseline)?

*Policy translation:* "How much domestic/allied nickel production is needed
before geopolitical restrictions become irrelevant to US industry?"

---

### 5.3 Indonesia ban reversal
**Causal question (L3):**
Given that Indonesia lifted ore export restrictions in certain conditions
historically: do(ban_lifted_year = T). At what T does early ban reversal,
combined with HPAL supply already entering, produce faster normalisation?

*Policy translation:* "What diplomatic timeline for Indonesia ban resolution
maximally benefits the US — and is there a window where HPAL overshoots
and actually drives prices below pre-ban levels?"

---

## 6. URANIUM

### 6.1 DOE Reserve sizing
**Causal question (L2):**
do(doe_reserve_drawdown = R_t): what reserve level (in years of SWU
consumption) and release schedule prevents the SEVERE_BAN scenario (100%
Russia + Kazakhstan disruption) from exceeding 3× baseline?

*Policy translation:* "What Congressional appropriation for the DOE Uranium
Reserve provides genuine nuclear fuel security under worst-case scenarios?"

---

### 6.2 Centrus HALEU capacity
**Causal question (L2):**
do(centrus_haleu_share = 0.08 by 2028): Centrus reaches full commercial
HALEU production covering ~8% of enrichment needs.
Does this plus Urenco US coverage reduce the effective restriction magnitude
below the "never normalises" threshold?

*Policy translation:* "Is the ADVANCE Act implementation sufficient to
prevent a sustained price crisis under a complete Rosatom ban?"

---

### 6.3 Perfect inelasticity paradox
**Causal question:**
Given η_D ≈ 0 (nuclear plants cannot reduce consumption), is there any
stockpile strategy (without demand-side response) that prevents prices
exceeding 5× under a 100% Kazakh + Russian ban?

*Policy translation:* "Is the US nuclear industry structurally exposed to
a price crisis no matter what stockpile policy is in place — requiring
new enrichment capacity not stockpiles?"

---

## 7. EXTENSION TO GERMANIUM AND GALLIUM

### 7.1 Why these minerals fit the framework

Germanium and gallium were subject to Chinese export controls from August 2023
— making them the most recent observable episodes with potential CEPII
calibration data. Both fit the ODE structural model because:

| Feature | Germanium | Gallium |
|---------|-----------|---------|
| China supply share | ~80% production | ~80–85% production |
| US import reliance | ~50% (estimated) | ~87% |
| Demand elasticity | Very low (semiconductor optics, IR) | Very low (GaN semiconductors, LEDs) |
| τ_K structure | Very long: byproduct of zinc/coal | Moderate-long: byproduct of aluminum |
| Export control start | August 2023 | August 2023 |
| CEPII HS code | 2804.70, 8112.92 | 2805.30 |

Both minerals share China's August 2023 export control episode as the
calibration event — meaning the framework can run a joint validation study.

---

### 7.2 Structural differences requiring model adaptation

**Byproduct τ_K structure**: Neither germanium nor gallium can be expanded
independently. Germanium supply is constrained by zinc smelting economics;
gallium by aluminum refining capacity. This means:

- τ_K for germanium ≈ τ_K(zinc mines) + processing time ≈ 10–15yr
- τ_K for gallium ≈ τ_K(aluminum refining expansion) ≈ 5–8yr

The ODE captures this correctly via τ_K — it does not require knowing the
primary metal. But the τ_K identification should use zinc/aluminum development
timelines as priors, not direct germanium/gallium mine data.

**Very small market volumes**: Germanium global production is ~130 tonnes/yr;
gallium ~300 tonnes/yr. Price volatility is extreme (small markets) and
speculative dynamics dominate. σ_P will be large — L3 U_t residuals will
be significant, consistent with what we observed for lithium/cobalt.

**No fringe supply**: Unlike lithium (Australian hard-rock) or graphite
(Mozambique), there is no credible non-Chinese fringe supply for germanium
or gallium at scale. The fringe_entry_price parameter should be set very
high (> 10× P_ref) or the fringe mechanism disabled.

---

### 7.3 Research questions for germanium

**Causal question (L2):**
do(china_export_control = 0.80, 2023–2025): calibrate from the Aug 2023
restriction. What are the implied τ_K, α_P, η_D from the 2023–2025 CEPII
trajectory?

*Purpose:* parameter identification for the most recent episode.

**Causal question (L2):**
do(china_ban = {0.25, 0.50, 0.75, 1.00} from 2026): given calibrated params,
how long does the US semiconductor and defense optics industry remain affected?

*Policy translation:* "Is germanium a strategic vulnerability for US infrared
defense systems (missile seekers, thermal imaging) comparable to graphite for
batteries?"

**Causal question (L2):**
do(us_recycling_share = 0.20 by 2028): Indium recycling in the US accounts
for ~15% of supply; similar germanium recycling programs could cover ~20%.
Does secondary supply meaningfully compress τ_K?

*Policy translation:* "Can recycling programs provide a faster supply response
than mine development for byproduct metals?"

---

### 7.4 Research questions for gallium

**Causal question (L2):**
do(china_ban = {0.25, 0.50, 0.75, 1.00} from 2026):
GaN semiconductors (5G, EV power electronics, defense radar) are the primary
demand driver. How long does the US GaN manufacturing sector remain disrupted?

*Policy translation:* "Is China's gallium export control a semiconductor supply
chain weapon comparable to its graphite battery supply chain control?"

**Causal question (L2):**
do(japan_gallium_production_share = 0.10 by 2027): Japan has secondary gallium
recovery from aluminum refining waste. A bilateral agreement could provide
~10% of US demand.
Does this combined with US recycling change the structural outcome?

*Policy translation:* "What allied supply agreements are needed to reduce
gallium vulnerability to below the 'critically exposed' threshold?"

**Causal question — compound scenario (L2):**
do(china_bans_gallium = 0.80 AND china_bans_germanium = 0.80 from 2026):
Both minerals are used in GaAs and GaN devices. A simultaneous restriction
would compound through the same downstream manufacturing supply chain.
What is the combined impact on US defense electronics production?

*Policy translation:* "Could China simultaneously restrict gallium and
germanium as a targeted attack on US defense semiconductor supply chains?"

---

### 7.5 Framework extension requirements

To run germanium and gallium through the model, three additions are needed:

**1. CEPII data ingestion**
Download BACI bilateral trade data for:
- HS 2804.70 (germanium) and HS 8112.92 (germanium compounds)
- HS 2805.30 (gallium, indium, thallium)
Run through `scripts/ingest_cepii.py` to produce
`data/canonical/cepii_germanium.csv` and `data/canonical/cepii_gallium.csv`.

**2. Calibration runs**
Run `src/minerals/predictability.py` with new episode configs:
- `germanium_2023_export_control`
- `gallium_2023_export_control`
Calibrate τ_K, α_P, η_D, g from Aug 2023 CEPII trajectory.

**3. HippoRAG document ingestion**
Add USGS MCS entries for germanium and gallium to the document corpus.
Run `scripts/index_hipporag.py` to incorporate.
The KG extractor will then identify causal triples for both minerals.

**4. Parameter priors for byproduct minerals**
Set τ_K prior from zinc/aluminum development timelines (not direct mine data).
Set fringe_entry_price = 5.0+ (no credible non-Chinese fringe at current prices).
Set fringe_capacity_share = 0.05 (near-zero — essentially disable fringe).

---

## 8. Cross-mineral transferability summary

| Question type | Transfers to | Key parameter that varies |
|---------------|-------------|--------------------------|
| Optimal stockpile release schedule | All minerals | τ_K (determines recovery lag) |
| Allied processing offset threshold | Graphite, REE, cobalt, lithium, Ge, Ga | China processing share |
| Processing vs. mine restriction comparison | All 5 China-processing minerals | Binding stage in KG |
| HPAL/brownfield τ_K compression | Nickel, cobalt, gallium (Al byproduct) | Effective τ_K under investment |
| LFP/substitution demand elasticity shift | Cobalt, REE (magnet substitution) | η_D trajectory |
| η_D threshold for inelastic demand | Lithium, uranium, Ge, Ga | η_D value |
| Byproduct τ_K structure | Germanium (Zn), gallium (Al), cobalt (Cu) | Primary metal τ_K |
| Capex shock (accident) | All minerals | Same ODE, different shock type |
| Demand collapse scenario | All minerals | Negative g or demand_surge |
| Sensitivity to τ_K uncertainty | All minerals | τ_K prior distribution |
