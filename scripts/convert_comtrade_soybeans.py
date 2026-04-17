#!/usr/bin/env python3
"""
Convert UN Comtrade annual soybean export data (HS 1201) to CEPII canonical format.

Merges two Comtrade downloads to maximise year coverage:
  data/raw/comtrade_soybeans_annual.csv  — 2014–2025 (v1)
  data/raw/comtrade_soybeans_v2.csv      — 2009–2020 (v2)

Years with known Brazil reporting gaps are dropped (world total ~half expected):
  2012, 2013, 2019 — Brazil absent from Comtrade
  2025             — incomplete year at time of download

Output: data/canonical/cepii_soybeans.csv
Columns: year, exporter, importer, product, value_kusd, quantity_tonnes

Usage:
    python scripts/convert_comtrade_soybeans.py
    python scripts/convert_comtrade_soybeans.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_V1 = ROOT / "data" / "raw" / "comtrade_soybeans_annual.csv"   # 2014–2025
RAW_V2 = ROOT / "data" / "raw" / "comtrade_soybeans_v2.csv"       # 2009–2020
OUT = ROOT / "data" / "canonical" / "cepii_soybeans.csv"

# Years where Brazil (world's largest exporter) is absent from Comtrade reporting
DROP_YEARS = [2012, 2013, 2019, 2025]


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1", index_col=False)
    print(f"  {path.name}: {len(df):,} rows, years {df['refYear'].min()}–{df['refYear'].max()}")
    return df


def convert(out_path: Path = OUT, dry_run: bool = False) -> pd.DataFrame:
    print("Loading raw files:")
    frames = []
    for path in [RAW_V1, RAW_V2]:
        if path.exists():
            frames.append(_load_raw(path))
        else:
            print(f"  WARNING: {path.name} not found — skipping")

    if not frames:
        raise FileNotFoundError("No raw Comtrade files found in data/raw/")

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(df):,} rows")

    # Keep export flows only
    exports = df[df["flowDesc"] == "Export"].copy()

    # Drop rows with missing weight or value
    exports = exports.dropna(subset=["netWgt", "primaryValue"])
    exports = exports[exports["netWgt"] > 0]
    exports = exports[exports["primaryValue"] > 0]

    # Build canonical columns
    canonical = pd.DataFrame({
        "year":            exports["refYear"].astype(int),
        "exporter":        exports["reporterDesc"],
        "importer":        "World",   # country→world totals; bilateral not available
        "product":         "1201",    # HS code for soybeans
        "value_kusd":      exports["primaryValue"] / 1000.0,  # USD → thousands USD
        "quantity_tonnes": exports["netWgt"] / 1000.0,         # kg → tonnes
    })

    # Deduplicate overlapping years between v1 and v2 (keep first occurrence)
    canonical = canonical.drop_duplicates(subset=["year", "exporter"])
    canonical = canonical.sort_values(["year", "exporter"]).reset_index(drop=True)

    # Drop years with known reporting gaps
    before = len(canonical)
    canonical = canonical[~canonical["year"].isin(DROP_YEARS)]
    print(f"Dropped years {DROP_YEARS}: {before - len(canonical)} rows removed")

    # Diagnostics
    world_annual = canonical.groupby("year").agg(
        world_qty_mt=("quantity_tonnes", lambda x: x.sum() / 1e6),
        world_val_busd=("value_kusd", lambda x: x.sum() / 1e6),
        n_reporters=("exporter", "count"),
    )
    world_annual["implied_price_usd_t"] = (
        world_annual["world_val_busd"] * 1e9 / (world_annual["world_qty_mt"] * 1e6)
    )
    print("\nAnnual world totals:")
    print(world_annual.round(2).to_string())

    usa = canonical[canonical["exporter"] == "USA"].groupby("year")["quantity_tonnes"].sum() / 1e6
    print("\nUS exports (Mt):")
    print(usa.round(2).to_string())

    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_csv(out_path, index=False)
        print(f"\nWrote {len(canonical):,} rows → {out_path}")
    else:
        print("\n(dry-run — no file written)")

    return canonical


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    convert(dry_run=args.dry_run)
