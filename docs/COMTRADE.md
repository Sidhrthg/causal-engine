# How to start using Comtrade data

This project uses **UN Comtrade** data for graphite (HS 2504) in validation, synthetic control, **and to train/calibrate the causal scenario model**. You can get data in two ways.

---

## 1. API download (recommended)

You need a **Comtrade API key** and the project’s download script.

### Setup

1. **Get an API key** from [UN Comtrade](https://comtradeplus.un.org/) (subscription).
2. **Put it in `.env`** in the project root:
   ```bash
   COMTRADE_API_KEY=your_key_here
   ```
3. **Activate the venv** and run the downloader:
   ```bash
   source .venv/bin/activate
   python -m scripts.download_comtrade_specific
   ```
4. Choose an option from the menu:
   - **USA imports** from Africa, South America, East Asia → `data/canonical/comtrade_usa_imports_africa_samerica_eastasia.csv`
   - **China imports** from the same regions → `data/canonical/comtrade_china_imports_africa_samerica_eastasia.csv`
   - **Intra-region flows** (trade between those countries) → `data/canonical/comtrade_intra_region_flows.csv`

Those CSVs are **panel** data (year × reporter × partner × trade_value_usd) and can be used to build a **synthetic-control** panel (e.g. one country as treated, others as controls).

---

## 2. Manual export + convert (no API key)

If you export data from the Comtrade website instead of using the API:

1. **Export** from [UN Comtrade](https://comtradeplus.un.org/) (e.g. USA imports, HS 2504, annual).
2. **Save the CSV** under:
   ```text
   data/raw/comtrade/
   ```
   Example: `TradeData_1_20_2026_18_16_50.csv` (or any name).
3. **Point the converter** at your file: edit `scripts/convert_comtrade.py` and set:
   ```python
   IN_PATH = Path("data/raw/comtrade/YOUR_FILE.csv")
   ```
4. **Run the converter**:
   ```bash
   python -m scripts.convert_comtrade
   ```
   This writes **`data/canonical/comtrade_graphite_trade.csv`** with columns: `date`, `series`, `value`, `unit`, `material`, `from_entity`, `to_entity`, `source`.

---

## What each file is used for

| File | Used by |
|------|--------|
| **`data/canonical/comtrade_graphite_trade.csv`** | Validate with RAG, `validate_historical.py`, `validate_aggregate.py`. Single or multi-series (date, value, from_entity, to_entity). |
| **`data/canonical/comtrade_graphite_trade.normalized.csv`** | Synthetic Control tab, `test_synthetic_control.py`, `demo_graphite_trade_shock.py`. Expects either: (a) **TimeSeriesSchema** (`year`, `P`, `D`, `Q`, `I`) for the demo, or (b) a **panel** with `year`, `to_entity` (or unit), `trade_value_usd` for synthetic control (multiple countries). |
| **`comtrade_usa_imports_...`.csv / `comtrade_china_imports_...`.csv** | Panel data; you can aggregate or reshape to build `comtrade_graphite_trade.normalized.csv` for synthetic control (treated unit + control units). |

---

## Quick path (you already have raw data)

If you already have **`data/raw/comtrade/TradeData_1_20_2026_18_16_50.csv`** (USA graphite imports):

```bash
# 1. Convert raw → canonical
python -m scripts.convert_comtrade
# → creates data/canonical/comtrade_graphite_trade.csv

# 2. Optional: create normalized (year, P) for demo / single-series use
#    Run the normalizer on the canonical file, or copy/rename if you already have
#    data/canonical/comtrade_graphite_trade.normalized.csv (year,P,D,Q,I).
#    For synthetic control with multiple countries, use API download and build a panel.
```

Then:

- **Validate with RAG** and **Validate with RAG** tab use `comtrade_graphite_trade.csv`.
- **Synthetic Control** tab and **Run synthetic control** use `comtrade_graphite_trade.normalized.csv`. The repo includes a single-series normalized file (USA only); for full synthetic control (USA vs DEU, JPN, IND, etc.) you need a panel, e.g. from the API download and then reshape to `year`, `to_entity`, `trade_value_usd`.

---

## Training the causal model on Comtrade

By default, scenario parameters (e.g. `tau_K`, `alpha_P`, `eta_D`) are set in the YAML and are **not** fitted to data. To have the **causal dynamics model use and train on Comtrade**:

1. Get Comtrade data (e.g. `data/canonical/comtrade_graphite_trade.csv`) as above.
2. Run **calibration** so the scenario’s simulated series (P×Q) fits Comtrade trade value over the same years:
   ```bash
   python -m scripts.calibrate_from_comtrade \
     --scenario scenarios/graphite_baseline.yaml \
     --comtrade data/canonical/comtrade_graphite_trade.csv \
     --out scenarios/graphite_comtrade_calibrated.yaml
   ```
3. This optimizes `tau_K`, `alpha_P`, `eta_D` (by default) to minimize RMSE between scaled P×Q and Comtrade, and writes a new scenario YAML. Use **`graphite_comtrade_calibrated.yaml`** (or your `--out` file) when running scenarios so the causal model is trained on Comtrade.

**Synthetic control** also trains on Comtrade: it fits weights on pre-treatment panel data to estimate treatment effects. So you have two ways the causal pipeline uses Comtrade: (1) **calibrated scenario** (dynamics parameters fitted to trade series), (2) **synthetic control** (treatment effect from panel data).

---

## Summary

1. **API**: Set `COMTRADE_API_KEY` in `.env` → run `python -m scripts.download_comtrade_specific` → pick USA/China/intra-region → CSVs in `data/canonical/`.
2. **Manual**: Export from Comtrade → save under `data/raw/comtrade/` → set `IN_PATH` in `convert_comtrade.py` → run `python -m scripts.convert_comtrade` → `comtrade_graphite_trade.csv` in `data/canonical/`.
3. **Train causal model on Comtrade**: Run `python -m scripts.calibrate_from_comtrade` → use the generated scenario YAML for runs.
4. Use **`comtrade_graphite_trade.csv`** for validation and calibration; use **`comtrade_graphite_trade.normalized.csv`** (panel or single-series) for the Synthetic Control tab and demos.
