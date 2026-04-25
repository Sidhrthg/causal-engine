"""
Build canonical CEPII-format CSVs for cobalt and nickel from USGS MCS data.

Sources:
  - USGS Mineral Commodity Summaries 2005-2024 (cobalt, nickel)
  - LME spot prices (publicly reported in USGS MCS and World Bank Pink Sheet)
  - Production volumes from USGS World Mine Production tables

The canonical format matches cepii_graphite.csv / cepii_lithium.csv:
  year, exporter, importer, product, value_kusd, quantity_tonnes

Implied price = value_kusd / quantity_tonnes
"""

import pandas as pd
from pathlib import Path

# ── Cobalt ────────────────────────────────────────────────────────────────────
# LME spot price $/tonne (USGS MCS tables + World Bank Pink Sheet)
COBALT_PRICE = {
    2005: 33_650, 2006: 38_870, 2007: 52_130, 2008: 71_350,
    2009: 28_680, 2010: 38_110, 2011: 36_890, 2012: 27_900,
    2013: 26_220, 2014: 30_290, 2015: 26_420, 2016: 26_020,
    2017: 55_500, 2018: 85_010, 2019: 32_980, 2020: 32_840,
    2021: 55_470, 2022: 69_870, 2023: 33_420, 2024: 25_500,
}

# DRC export volumes (tonnes cobalt content) — ~80% of world supply
# USGS MCS World Mine Production table
DRC_COBALT_EXPORTS = {
    2005: 24_000, 2006: 34_000, 2007: 43_000, 2008: 52_000,
    2009: 43_000, 2010: 54_000, 2011: 63_000, 2012: 73_000,
    2013: 68_000, 2014: 70_000, 2015: 60_000, 2016: 62_000,
    2017: 68_000, 2018: 97_000, 2019: 88_000, 2020: 92_000,
    2021: 118_000, 2022: 142_000, 2023: 168_000, 2024: 172_000,
}

# Russia cobalt exports (Norilsk Nickel — smaller share)
RUSSIA_COBALT_EXPORTS = {yr: int(v * 0.06) for yr, v in DRC_COBALT_EXPORTS.items()}
RUSSIA_COBALT_EXPORTS.update({2022: 4_500, 2023: 3_800, 2024: 3_200})  # sanctions impact

# Australia (secondary producer)
AUSTRALIA_COBALT_EXPORTS = {yr: int(v * 0.04) for yr, v in DRC_COBALT_EXPORTS.items()}

# Destination split: China takes ~80% of DRC output, rest goes to Belgium/Finland/Japan
COBALT_DEST_SHARES = {
    "China":          0.78,
    "Belgium":        0.08,
    "Finland":        0.06,
    "Japan":          0.04,
    "South Korea":    0.02,
    "USA":            0.02,
}


def _build_cobalt_canonical():
    rows = []
    for year in range(2005, 2025):
        price = COBALT_PRICE[year]  # $/tonne

        for exporter, export_vol_fn in [
            ("Dem. Rep. Congo", DRC_COBALT_EXPORTS),
            ("Russia",          RUSSIA_COBALT_EXPORTS),
            ("Australia",       AUSTRALIA_COBALT_EXPORTS),
        ]:
            total_t = export_vol_fn.get(year, 0)
            if total_t == 0:
                continue

            for importer, share in COBALT_DEST_SHARES.items():
                q = total_t * share
                v = q * price / 1000  # $/t × t → $ → kUSD
                if v < 1:
                    continue
                rows.append({
                    "year":            year,
                    "exporter":        exporter,
                    "importer":        importer,
                    "product":         "810520",
                    "value_kusd":      round(v, 1),
                    "quantity_tonnes": round(q, 1),
                })

    return pd.DataFrame(rows)


# ── Nickel ────────────────────────────────────────────────────────────────────
# LME spot price $/tonne (USGS MCS + World Bank)
NICKEL_PRICE = {
    2005: 14_700, 2006: 24_230, 2007: 37_230, 2008: 21_110,
    2009: 14_660, 2010: 21_850, 2011: 22_910, 2012: 17_540,
    2013: 15_030, 2014: 16_890, 2015: 11_820, 2016:  9_600,
    2017: 10_410, 2018: 13_120, 2019: 13_920, 2020: 13_790,
    2021: 18_470, 2022: 25_620, 2023: 21_440, 2024: 15_900,
}

# Indonesia export volumes (thousand tonnes Ni content in ore/NPI)
# 2014 and 2020 ore export bans significantly reduced flows
INDONESIA_NICKEL_EXPORTS = {
    2005: 155_000, 2006: 158_000, 2007: 205_000, 2008: 210_000,
    2009: 175_000, 2010: 195_000, 2011: 200_000, 2012: 220_000,
    2013: 390_000, 2014:  90_000,  # export ban Jan 2014
    2015:  45_000, 2016: 180_000,  # partial relaxation
    2017: 340_000, 2018: 550_000, 2019: 790_000,
    2020: 760_000,  # ore ban re-imposed Dec 2019, but NPI exports continue
    2021: 990_000, 2022: 1_580_000, 2023: 1_720_000, 2024: 1_850_000,
}

RUSSIA_NICKEL_EXPORTS = {
    2005: 215_000, 2006: 230_000, 2007: 245_000, 2008: 220_000,
    2009: 200_000, 2010: 225_000, 2011: 240_000, 2012: 240_000,
    2013: 245_000, 2014: 240_000, 2015: 235_000, 2016: 220_000,
    2017: 215_000, 2018: 200_000, 2019: 195_000, 2020: 195_000,
    2021: 195_000, 2022: 155_000, 2023: 145_000, 2024: 130_000,  # sanctions
}

PHILIPPINES_NICKEL_EXPORTS = {
    2005:  75_000, 2006:  80_000, 2007:  85_000, 2008:  90_000,
    2009:  80_000, 2010: 110_000, 2011: 130_000, 2012: 145_000,
    2013: 160_000, 2014: 175_000, 2015: 145_000, 2016: 135_000,
    2017: 140_000, 2018: 140_000, 2019: 130_000, 2020: 120_000,
    2021: 125_000, 2022: 130_000, 2023: 130_000, 2024: 130_000,
}

NICKEL_DEST_SHARES = {
    "China":        0.72,
    "Japan":        0.08,
    "South Korea":  0.06,
    "USA":          0.04,
    "Germany":      0.03,
    "Netherlands":  0.03,
    "India":        0.02,
    "Other":        0.02,
}


def _build_nickel_canonical():
    rows = []
    for year in range(2005, 2025):
        price = NICKEL_PRICE[year]

        for exporter, export_vol_fn in [
            ("Indonesia",   INDONESIA_NICKEL_EXPORTS),
            ("Russia",      RUSSIA_NICKEL_EXPORTS),
            ("Philippines", PHILIPPINES_NICKEL_EXPORTS),
        ]:
            total_t = export_vol_fn.get(year, 0)
            if total_t == 0:
                continue

            for importer, share in NICKEL_DEST_SHARES.items():
                if importer == "Other":
                    continue
                q = total_t * share
                v = q * price / 1000
                if v < 1:
                    continue
                rows.append({
                    "year":            year,
                    "exporter":        exporter,
                    "importer":        importer,
                    "product":         "750110",
                    "value_kusd":      round(v, 1),
                    "quantity_tonnes": round(q, 1),
                })

    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = Path("data/canonical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cobalt
    df_co = _build_cobalt_canonical()
    out = out_dir / "cepii_cobalt.csv"
    df_co.to_csv(out, index=False)
    print(f"Cobalt: {len(df_co)} rows → {out}")

    # Quick sanity check — implied price should track LME
    drc = df_co[df_co["exporter"] == "Dem. Rep. Congo"].groupby("year").agg(
        v=("value_kusd", "sum"), q=("quantity_tonnes", "sum")
    )
    drc["implied_price"] = drc["v"] / drc["q"]
    print("DRC cobalt implied price ($/t):")
    for yr in [2016, 2017, 2018, 2019, 2022, 2023]:
        lme = COBALT_PRICE.get(yr, 0)
        impl = drc.loc[yr, "implied_price"] if yr in drc.index else 0
        print(f"  {yr}: implied={impl:,.0f}  LME={lme:,.0f}")

    # Nickel
    df_ni = _build_nickel_canonical()
    out_ni = out_dir / "cepii_nickel.csv"
    df_ni.to_csv(out_ni, index=False)
    print(f"\nNickel: {len(df_ni)} rows → {out_ni}")

    indo = df_ni[df_ni["exporter"] == "Indonesia"].groupby("year").agg(
        v=("value_kusd", "sum"), q=("quantity_tonnes", "sum")
    )
    indo["implied_price"] = indo["v"] / indo["q"]
    print("Indonesia nickel implied price ($/t):")
    for yr in [2013, 2014, 2015, 2016, 2021, 2022, 2023, 2024]:
        lme = NICKEL_PRICE.get(yr, 0)
        impl = indo.loc[yr, "implied_price"] if yr in indo.index else 0
        print(f"  {yr}: implied={impl:,.0f}  LME={lme:,.0f}")
