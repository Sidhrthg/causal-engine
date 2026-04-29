# Required Data Pulls

This document lists data files that are NOT in the repository but are
required for full reproduction of analyses. Each entry includes the
provenance, expected location, and download instructions.

---

## CEPII BACI HS92 — Germanium and Gallium

**Status:** NOT IN REPO. Required for calibrating gallium/germanium forward
scenarios beyond the rare_earths_2010 prior used in `scripts/byproduct_forward_2026.py`.

**HS codes needed:**
- **2804.70** — Germanium (and other group V/VI elements; filter to Germanium row)
- **2805.30** — Rare earth metals, scandium, yttrium *(this code's main use here is for cross-check; rare earths are already covered)*
- **8112.92** — Germanium articles, raw form (waste/scrap and unwrought)
- **2818.30** — Gallium oxide (proxy for processed gallium)
- **2805.40** — Mercury — irrelevant (do not download)

For gallium specifically, CEPII does not have a dedicated HS6 code at the
metal level. The closest is processed gallium chemistry intermediates.
Gallium may need to be tracked via UN Comtrade at HS6 = 281000 or 8112.99.

**Where to download:**
- CEPII portal: http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37
- Required: free registration
- Data structure: BACI HS92, one .csv (or .txt) per year, 1995-2024
- File naming convention: `BACI_HS92_Y{YEAR}_V{VERSION}.csv`
- Recommended version: V202601 or latest

**Storage:**
- Raw downloads: `data/documents/cepii/baci_hs92/`
  - One file per year, naming: `BACI_HS92_Y{YEAR}_V202601.csv` (full HS6 panel)
- Processed canonical: `data/canonical/cepii_germanium.csv` and `cepii_gallium.csv`
  - Filtered to relevant HS codes only
  - Long format: year, exporter, importer, product, value_kusd, quantity_tonnes

**Processing pipeline:**
After raw downloads land, run:
```python
python3 scripts/ingest_cepii_byproduct.py --commodity germanium
python3 scripts/ingest_cepii_byproduct.py --commodity gallium
```
(Script not yet written; create when raw data lands. Pattern follows
existing CEPII ingestion logic.)

## EIA Uranium Marketing Annual Report (already in repo)

**Status:** IN REPO at `data/canonical/eia_uranium_spot_price.csv` (2002-2024)
**Source:** EIA UMAR 2024 (released Sep 2025), Table S1b spot-contract column.
Update annually via EIA portal.

## GPR Country Annual (in repo, predictive branch only)

**Status:** Built on `predictive-experimental` branch only.
**Source:** Caldara & Iacoviello, https://www.matteoiacoviello.com/gpr.htm
**Path:** `data/canonical/gpr_country_annual.csv`
**Re-fetch:** `python3 src/minerals/predictive/fetch_gpr.py`

## S&P Country Risk (NOT IN REPO)

**Status:** Pending user pull from Capital IQ Pro.
**Target path:** `data/canonical/sp_country_risk.csv`
**See:** `src/minerals/predictive/SP_DATA_REQUEST.md` for column spec
and download instructions.

## Document corpus (in repo)

**Status:** IN REPO at `data/documents/`
- USGS Mineral Commodity Summaries 2020-2024
- IEA Critical Minerals reports (2021, 2022, 2023, 2024)
- World Bank Commodity Markets Outlook archives

---

## Reproducibility Note

Scripts that depend on missing data will print a clear error message and
exit cleanly. The forward scenarios in `scripts/byproduct_forward_2026.py`
use parameter PRIORS and run without CEPII Ge/Ga data — they are clearly
labeled PROVISIONAL in output. When CEPII data lands, the same script can
be re-run with `_GERMANIUM_2023_PARAMS` (calibrated) replacing
`_GERMANIUM_2023_PRIOR` to produce the calibrated version of the same
analysis.
