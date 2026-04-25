"""
Download bilateral trade data from UN Comtrade public API for any commodity.

Outputs a canonical CSV matching the format expected by predictability.py:
  year, exporter, importer, product, value_kusd, quantity_tonnes

Usage:
  python scripts/download_cepii_commodity.py --commodity cobalt --hs 810520
  python scripts/download_cepii_commodity.py --commodity nickel --hs 750110
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

# Comtrade ISO3 numeric → country name
_COUNTRY_CODES = {
    "004": "Afghanistan", "008": "Albania", "012": "Algeria", "024": "Angola",
    "032": "Argentina", "036": "Australia", "040": "Austria", "050": "Bangladesh",
    "056": "Belgium", "068": "Bolivia", "076": "Brazil", "100": "Bulgaria",
    "116": "Cambodia", "120": "Cameroon", "124": "Canada", "144": "Sri Lanka",
    "152": "Chile", "156": "China", "170": "Colombia", "180": "Dem. Rep. Congo",
    "188": "Costa Rica", "191": "Croatia", "203": "Czech Republic", "208": "Denmark",
    "218": "Ecuador", "818": "Egypt", "231": "Ethiopia", "246": "Finland",
    "250": "France", "266": "Gabon", "276": "Germany", "288": "Ghana",
    "320": "Guatemala", "348": "Hungary", "356": "India", "360": "Indonesia",
    "364": "Iran", "368": "Iraq", "372": "Ireland", "376": "Israel",
    "380": "Italy", "388": "Jamaica", "392": "Japan", "400": "Jordan",
    "398": "Kazakhstan", "404": "Kenya", "410": "South Korea", "417": "Kyrgyzstan",
    "426": "Lesotho", "430": "Liberia", "434": "Libya", "458": "Malaysia",
    "484": "Mexico", "496": "Mongolia", "504": "Morocco", "508": "Mozambique",
    "516": "Namibia", "524": "Nepal", "528": "Netherlands", "554": "New Zealand",
    "566": "Nigeria", "578": "Norway", "586": "Pakistan", "591": "Panama",
    "604": "Peru", "608": "Philippines", "616": "Poland", "620": "Portugal",
    "630": "Puerto Rico", "634": "Qatar", "642": "Romania", "643": "Russia",
    "682": "Saudi Arabia", "686": "Senegal", "694": "Sierra Leone",
    "706": "Somalia", "710": "South Africa", "724": "Spain", "752": "Sweden",
    "756": "Switzerland", "762": "Tajikistan", "764": "Thailand", "788": "Tunisia",
    "792": "Turkey", "800": "Uganda", "804": "Ukraine", "784": "UAE",
    "826": "United Kingdom", "840": "USA", "842": "USA", "858": "Uruguay",
    "860": "Uzbekistan", "862": "Venezuela", "704": "Vietnam", "887": "Yemen",
    "894": "Zambia", "716": "Zimbabwe",
    # Special codes used by Comtrade
    "0": "World", "899": "Areas NES", "697": "State of Palestine",
}


def _resolve(code) -> str:
    s = str(int(float(code))).zfill(3)
    return _COUNTRY_CODES.get(s, s)


BASE_URL = "https://comtradeapi.un.org/public/v1/get"


def _fetch(reporter: str, hs: str, year: int, flow: str = "X") -> list[dict]:
    """Fetch one reporter × year × flow from Comtrade public API."""
    url = f"{BASE_URL}/{year}/{flow}/{reporter}/all/{hs}"
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("data", []) or []
        return []
    except Exception:
        return []


def download_commodity(
    hs: str,
    commodity: str,
    start_year: int = 2005,
    end_year: int = 2024,
    key_exporters: list[str] = None,
    out_csv: Path = None,
) -> pd.DataFrame:
    """
    Download export data for all key_exporters for a given HS code and year range.
    Returns a canonical DataFrame and writes it to out_csv.
    """
    # Comtrade reporter codes for key mineral exporters
    exporter_codes = {
        "China":                 "156",
        "Dem. Rep. Congo":       "180",
        "Australia":             "036",
        "Chile":                 "152",
        "Indonesia":             "360",
        "Philippines":           "608",
        "Russia":                "643",
        "South Africa":          "710",
        "Zambia":                "894",
        "Zimbabwe":              "716",
        "Brazil":                "076",
        "USA":                   "840",
        "Canada":                "124",
        "Finland":               "246",
        "Norway":                "578",
        "Morocco":               "504",
        "Madagascar":            "450",
        "Mozambique":            "508",
        "Kazakhstan":            "398",
    }

    if key_exporters:
        codes = {k: v for k, v in exporter_codes.items() if k in key_exporters}
    else:
        codes = exporter_codes

    rows = []
    total = len(codes) * (end_year - start_year + 1)
    done = 0

    for country, code in codes.items():
        for year in range(start_year, end_year + 1):
            done += 1
            print(f"\r[{done}/{total}] {country} {year}...", end="", flush=True)
            records = _fetch(code, hs, year, flow="X")
            for rec in records:
                partner_code = str(rec.get("partnerCode", "0"))
                partner = _resolve(partner_code)
                val = float(rec.get("primaryValue", 0) or 0)
                qty = float(rec.get("netWgt", 0) or 0)
                if val < 1:
                    continue
                rows.append({
                    "year":             year,
                    "exporter":         country,
                    "importer":         partner,
                    "product":          hs,
                    "value_kusd":       val / 1000,   # USD → kUSD
                    "quantity_tonnes":  qty / 1000,   # kg  → tonnes
                })
            time.sleep(0.6)   # public API rate limit

    print()
    if not rows:
        print("WARNING: no data downloaded")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["importer"] != "World"]   # drop aggregate rows

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"Saved {len(df)} rows → {out_csv}")

    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--commodity", required=True, help="e.g. cobalt, nickel")
    p.add_argument("--hs", required=True, help="HS6 code, e.g. 810520")
    p.add_argument("--start", type=int, default=2005)
    p.add_argument("--end",   type=int, default=2024)
    p.add_argument("--exporters", default=None,
                   help="Comma-separated list of exporters to fetch (default: all key exporters)")
    args = p.parse_args()

    exporters = [e.strip() for e in args.exporters.split(",")] if args.exporters else None
    out = Path(f"data/canonical/cepii_{args.commodity}.csv")

    df = download_commodity(
        hs=args.hs,
        commodity=args.commodity,
        start_year=args.start,
        end_year=args.end,
        key_exporters=exporters,
        out_csv=out,
    )

    if not df.empty:
        print(f"\nTop exporters by value:")
        top = (df.groupby("exporter")["value_kusd"].sum()
                 .sort_values(ascending=False).head(8))
        for country, val in top.items():
            print(f"  {country}: ${val/1e6:,.1f}M")

        print(f"\nYears covered: {sorted(df['year'].unique())}")


if __name__ == "__main__":
    main()
