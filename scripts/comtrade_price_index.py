"""
Compare UN Comtrade implied prices against CEPII for all critical minerals.

Usage:
    python scripts/comtrade_price_index.py                  # all minerals, data/comtrade/
    python scripts/comtrade_price_index.py cobalt nickel    # specific minerals
    python scripts/comtrade_price_index.py data/comtrade/my_file.csv  # specific files

Minerals supported: cobalt, nickel, lithium, graphite, rare_earths, soybeans
"""

import sys
import glob
from pathlib import Path

import pandas as pd
import numpy as np

BASE_YEAR = 2018
COMTRADE_DIR = "data/comtrade"
CEPII_DIR = "data/canonical"

# HS codes and CEPII file for each mineral
MINERALS = {
    "cobalt":      dict(hs=["810520"],        cepii="cepii_cobalt.csv"),
    "nickel":      dict(hs=["750210"],        cepii="cepii_nickel.csv"),
    "lithium":     dict(hs=["283691"],        cepii="cepii_lithium.csv"),
    "graphite":    dict(hs=["250410"],        cepii="cepii_graphite.csv"),
    "rare_earths": dict(hs=["284610"],        cepii="cepii_rare_earths.csv"),
    "soybeans":    dict(hs=["1201"],          cepii="cepii_soybeans.csv"),
}


def load_comtrade(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False, encoding="latin-1", index_col=False)
            frames.append(df)
        except Exception as e:
            print(f"  warning: could not read {p}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def comtrade_annual_price(raw: pd.DataFrame, hs_codes: list[str]) -> pd.Series:
    cmd = raw["cmdCode"].str.strip()
    flow = raw["flowCode"].str.strip()
    partner = raw["partnerISO"].str.strip()

    mask = cmd.isin(hs_codes) & flow.isin(["X", "Export"]) & partner.isin(["W00", "World"])
    df = raw[mask].copy()
    if df.empty:
        return pd.Series(dtype=float)

    df["year"] = pd.to_numeric(df["refYear"], errors="coerce")
    df["fob"] = pd.to_numeric(df["fobvalue"], errors="coerce")
    df["qty_kg"] = pd.to_numeric(df["qty"], errors="coerce")
    df = df[(df["qty_kg"] > 0) & (df["fob"] > 0)].dropna(subset=["year", "fob", "qty_kg"])

    annual = df.groupby("year").agg(fob=("fob", "sum"), qty=("qty_kg", "sum"))
    return (annual["fob"] / annual["qty"]).sort_index()  # USD/kg


def cepii_annual_price(cepii_file: str) -> pd.Series:
    path = Path(CEPII_DIR) / cepii_file
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path)
    world = df.groupby("year").agg(
        value_kusd=("value_kusd", "sum"),
        qty_tonnes=("quantity_tonnes", "sum"),
    )
    # kusd/tonne == USD/kg numerically
    return (world["value_kusd"] / world["qty_tonnes"]).sort_index()


def print_comparison(name: str, ct: pd.Series, cp: pd.Series) -> None:
    overlap = sorted(set(ct.index) & set(cp.index))
    if len(overlap) < 2:
        print(f"  {name}: insufficient overlap ({len(overlap)} years)")
        return

    base = BASE_YEAR if BASE_YEAR in overlap else overlap[0]

    ct_idx = ct / ct.loc[base] if base in ct.index else ct / ct.iloc[0]
    cp_idx = cp / cp.loc[base] if base in cp.index else cp / cp.iloc[0]

    agree = total = 0
    shock_rows = []
    yoy_rows = []

    prev_c = prev_t = prev_yr = None
    for yr in overlap:
        c = cp.get(yr, np.nan)
        t = ct.get(yr, np.nan)
        if prev_c is not None and not np.isnan(c) and not np.isnan(t):
            yoy_c = c / prev_c - 1
            yoy_t = t / prev_t - 1
            match = yoy_c * yoy_t > 0
            agree += int(match)
            total += 1
            tag = "YES" if match else "NO "
            yoy_rows.append(f"  {int(yr)}  CEPII {yoy_c:+6.1%}  Comtrade {yoy_t:+6.1%}  {tag}")
        prev_c, prev_t, prev_yr = c, t, yr

    da = agree / total if total else 0
    print(f"\n{'='*55}")
    print(f"  {name.upper()}  —  directional agreement: {agree}/{total} = {da:.0%}")
    print(f"{'='*55}")
    for row in yoy_rows:
        print(row)
    print(f"\n  Comtrade years: {sorted(ct.index.astype(int).tolist())}")
    print(f"  CEPII years:    {sorted(cp.index.astype(int).tolist())}")


def resolve_paths(args: list[str]) -> list[str]:
    paths = []
    for arg in args:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])
    resolved = []
    for p in paths:
        if Path(p).is_dir():
            resolved.extend(glob.glob(str(Path(p) / "*.csv")))
        else:
            resolved.append(p)
    return resolved


def main():
    args = sys.argv[1:]

    # Determine which minerals to run and which files to load
    mineral_filter = [a for a in args if a in MINERALS]
    file_args = [a for a in args if a not in MINERALS]

    targets = mineral_filter if mineral_filter else list(MINERALS.keys())

    if file_args:
        paths = resolve_paths(file_args)
    else:
        paths = resolve_paths([COMTRADE_DIR])

    print(f"Loading {len(paths)} Comtrade file(s)...")
    raw = load_comtrade(paths)
    if raw.empty:
        raise SystemExit("No Comtrade data loaded.")
    print(f"  {len(raw):,} rows total")
    print(f"  HS codes present: {sorted(raw['cmdCode'].str.strip().unique().tolist())}")

    for mineral in targets:
        cfg = MINERALS[mineral]
        ct = comtrade_annual_price(raw, cfg["hs"])
        cp = cepii_annual_price(cfg["cepii"])
        if ct.empty:
            print(f"\n  {mineral}: no Comtrade rows found for HS {cfg['hs']}")
            continue
        if cp.empty:
            print(f"\n  {mineral}: CEPII file not found ({cfg['cepii']})")
            continue
        print_comparison(mineral, ct, cp)


if __name__ == "__main__":
    main()
