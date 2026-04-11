#!/usr/bin/env python3
"""
Ingest CEPII bilateral trade flow data (BACI / CHELEM / gravity datasets).

Converts CSVs → text summaries for RAG indexing and a canonical CSV
for scenario calibration.

Supported CEPII formats:
  BACI:    t, i (exporter), j (importer), k (HS6 code), v (value kUSD), q (tonnes)
  CHELEM:  year, exporter, importer, product, value
  Generic: any CSV with year, exporter/reporter, importer/partner, value columns

Usage:
  python scripts/ingest_cepii.py --input data/cepii/ --mineral graphite --hs 250410
  python scripts/ingest_cepii.py --input data/cepii/BACI_HS02_Y2005.csv --mineral graphite
  python scripts/ingest_cepii.py --input data/cepii/ --list-products   # inspect HS codes
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

# ---------------------------------------------------------------------------
# Country code → name lookup (ISO3 numeric for BACI)
# ---------------------------------------------------------------------------
_ISO3_NUM = {
    "156": "China", "842": "USA", "276": "Germany", "392": "Japan",
    "410": "South Korea", "356": "India", "076": "Brazil", "124": "Canada",
    "036": "Australia", "710": "South Africa", "504": "Morocco",
    "404": "Kenya", "508": "Mozambique", "800": "Uganda",
    "578": "Norway", "752": "Sweden", "246": "Finland", "208": "Denmark",
    "056": "Belgium", "528": "Netherlands", "250": "France",
    "380": "Italy", "724": "Spain", "620": "Portugal",
    "616": "Poland", "203": "Czech Republic", "348": "Hungary",
    "642": "Romania", "100": "Bulgaria", "191": "Croatia",
    "792": "Turkey", "804": "Ukraine", "643": "Russia",
    "398": "Kazakhstan", "860": "Uzbekistan", "417": "Kyrgyzstan",
    "496": "Mongolia", "408": "North Korea", "764": "Thailand",
    "360": "Indonesia", "458": "Malaysia", "608": "Philippines",
    "704": "Vietnam", "116": "Cambodia", "764": "Thailand",
    "050": "Bangladesh", "586": "Pakistan", "144": "Sri Lanka",
    "818": "Egypt", "566": "Nigeria", "288": "Ghana", "012": "Algeria",
    "788": "Tunisia", "682": "Saudi Arabia", "784": "UAE",
    "364": "Iran", "368": "Iraq", "376": "Israel",
    "484": "Mexico", "170": "Colombia", "604": "Peru",
    "152": "Chile", "032": "Argentina", "858": "Uruguay",
    "218": "Ecuador", "862": "Venezuela", "591": "Panama",
    "320": "Guatemala", "188": "Costa Rica",
}

# ISO2 alpha codes (CHELEM / some CEPII variants)
_ISO2 = {
    "CN": "China", "US": "USA", "DE": "Germany", "JP": "Japan",
    "KR": "South Korea", "IN": "India", "BR": "Brazil", "CA": "Canada",
    "AU": "Australia", "ZA": "South Africa", "MA": "Morocco",
    "NO": "Norway", "SE": "Sweden", "FI": "Finland", "FR": "France",
    "IT": "Italy", "ES": "Spain", "GB": "United Kingdom",
    "RU": "Russia", "UA": "Ukraine", "TR": "Turkey",
    "MX": "Mexico", "CO": "Colombia", "PE": "Peru", "CL": "Chile",
    "AR": "Argentina", "NG": "Nigeria", "GH": "Ghana", "EG": "Egypt",
    "SA": "Saudi Arabia", "AE": "UAE", "TH": "Thailand",
    "ID": "Indonesia", "MY": "Malaysia", "PH": "Philippines",
    "VN": "Vietnam", "MN": "Mongolia", "KZ": "Kazakhstan",
    "MZ": "Mozambique", "TZ": "Tanzania", "KE": "Kenya",
}

_ALL_CODES = {**_ISO3_NUM, **_ISO2}


def resolve_country(code: str) -> str:
    code = str(code).strip().zfill(3) if str(code).isdigit() else str(code).strip().upper()
    return _ALL_CODES.get(code, code)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
def _detect_columns(header: List[str]) -> Dict[str, str]:
    """Return mapping role → actual column name."""
    h_lower = {c.lower(): c for c in header}
    mapping = {}

    # Year
    for cand in ["t", "year", "yr", "period", "ref_year"]:
        if cand in h_lower:
            mapping["year"] = h_lower[cand]
            break

    # Exporter
    for cand in ["i", "exporter", "reporter", "origin", "source", "exp"]:
        if cand in h_lower:
            mapping["exporter"] = h_lower[cand]
            break

    # Importer
    for cand in ["j", "importer", "partner", "destination", "dest", "imp"]:
        if cand in h_lower:
            mapping["importer"] = h_lower[cand]
            break

    # Product
    for cand in ["k", "hs6", "hs", "product", "commodity", "cmdcode", "cmd_code", "sitc"]:
        if cand in h_lower:
            mapping["product"] = h_lower[cand]
            break

    # Value
    for cand in ["v", "value", "trade_value", "tradevalue", "fob", "cif", "v_usd"]:
        if cand in h_lower:
            mapping["value"] = h_lower[cand]
            break

    # Quantity
    for cand in ["q", "qty", "quantity", "weight", "net_wgt", "netweight", "tonnes"]:
        if cand in h_lower:
            mapping["quantity"] = h_lower[cand]
            break

    return mapping


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------
def ingest_cepii(
    input_path: Path,
    mineral: str,
    hs_codes: Optional[List[str]],
    out_docs_dir: Path,
    out_csv: Optional[Path],
    max_rows: int = 500_000,
) -> Dict:
    """
    Read one or more CEPII CSVs, optionally filter by HS code,
    produce text documents and a canonical CSV.
    """
    files = []
    if input_path.is_dir():
        files = sorted(input_path.glob("*.csv")) + sorted(input_path.glob("*.CSV"))
    else:
        files = [input_path]

    if not files:
        print(f"No CSV files found at {input_path}")
        sys.exit(1)

    print(f"Found {len(files)} CSV file(s)")

    out_docs_dir.mkdir(parents=True, exist_ok=True)
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
    canonical_rows = []
    docs_written = 0
    total_rows = 0

    # Normalise HS codes
    hs_set = set()
    if hs_codes:
        for h in hs_codes:
            hs_set.add(h.lstrip("0"))   # e.g. "250410" → "250410" and "10410" for partial
            hs_set.add(h)

    for csv_path in files:
        print(f"\nProcessing: {csv_path.name}")
        try:
            rows = _process_file(csv_path, hs_set, max_rows)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not rows:
            print("  No matching rows after filter")
            continue

        total_rows += len(rows)
        canonical_rows.extend(rows)

        # Write text document per file
        doc_text = _rows_to_text(rows, mineral, csv_path.stem)
        doc_path = out_docs_dir / f"cepii_{mineral}_{csv_path.stem}.txt"
        doc_path.write_text(doc_text, encoding="utf-8")
        docs_written += 1
        print(f"  → {len(rows)} rows → {doc_path.name}")

    # Also write aggregated year summaries (useful for RAG)
    year_summaries = _build_year_summaries(canonical_rows, mineral)
    for year, text in year_summaries.items():
        ypath = out_docs_dir / f"cepii_{mineral}_summary_{year}.txt"
        ypath.write_text(text, encoding="utf-8")
        docs_written += 1

    print(f"\nWrote year summaries for {len(year_summaries)} years")

    # Write canonical CSV
    if out_csv and canonical_rows:
        fieldnames = ["year", "exporter", "importer", "product", "value_kusd", "quantity_tonnes"]
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(canonical_rows)
        print(f"\nCanonical CSV → {out_csv}  ({len(canonical_rows)} rows)")

    return {
        "files_processed": len(files),
        "total_rows": total_rows,
        "docs_written": docs_written,
        "years": sorted(set(r["year"] for r in canonical_rows)),
        "top_exporters": _top_n(canonical_rows, "exporter", 10),
        "top_importers": _top_n(canonical_rows, "importer", 10),
    }


def _process_file(csv_path: Path, hs_set: set, max_rows: int) -> List[Dict]:
    """Read CSV, detect columns, optionally filter, return normalised rows."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        cols = _detect_columns(list(reader.fieldnames))
        missing = [r for r in ("year", "exporter", "importer", "value") if r not in cols]
        if missing:
            raise ValueError(f"Could not detect columns {missing}. "
                             f"Headers found: {reader.fieldnames}")

        for i, row in enumerate(reader):
            if i >= max_rows:
                print(f"  (truncated at {max_rows} rows)")
                break

            # Filter by HS code
            if hs_set:
                prod = str(row.get(cols.get("product", ""), "")).strip().lstrip("0")
                if prod not in hs_set:
                    continue

            year = str(row[cols["year"]]).strip()
            exp = resolve_country(row[cols["exporter"]])
            imp = resolve_country(row[cols["importer"]])

            try:
                val = float(str(row[cols["value"]]).replace(",", "") or 0)
            except ValueError:
                val = 0.0

            qty = 0.0
            if "quantity" in cols:
                try:
                    qty = float(str(row[cols["quantity"]]).replace(",", "") or 0)
                except ValueError:
                    qty = 0.0

            product = row.get(cols.get("product", ""), "")

            rows.append({
                "year": year,
                "exporter": exp,
                "importer": imp,
                "product": str(product).strip(),
                "value_kusd": val,
                "quantity_tonnes": qty,
            })

    return rows


def _rows_to_text(rows: List[Dict], mineral: str, source: str) -> str:
    """Convert rows to a RAG-friendly text document."""
    lines = [
        f"CEPII Trade Flow Data — {mineral.title()}",
        f"Source file: {source}",
        f"Total records: {len(rows)}",
        "",
    ]

    # Group by year
    by_year: Dict[str, List[Dict]] = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)

    for year in sorted(by_year):
        year_rows = by_year[year]
        total_val = sum(r["value_kusd"] for r in year_rows)
        total_qty = sum(r["quantity_tonnes"] for r in year_rows)

        lines.append(f"Year {year}: {len(year_rows)} bilateral flows, "
                     f"total value {total_val:,.0f} kUSD, "
                     f"total quantity {total_qty:,.0f} tonnes")

        # Top exporters
        exp_totals: Dict[str, float] = {}
        for r in year_rows:
            exp_totals[r["exporter"]] = exp_totals.get(r["exporter"], 0) + r["value_kusd"]
        top_exp = sorted(exp_totals.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Top exporters: " +
                     ", ".join(f"{c} ({v:,.0f} kUSD)" for c, v in top_exp))

        # Top importers
        imp_totals: Dict[str, float] = {}
        for r in year_rows:
            imp_totals[r["importer"]] = imp_totals.get(r["importer"], 0) + r["value_kusd"]
        top_imp = sorted(imp_totals.items(), key=lambda x: -x[1])[:5]
        lines.append(f"  Top importers: " +
                     ", ".join(f"{c} ({v:,.0f} kUSD)" for c, v in top_imp))

        # Notable bilateral flows
        top_flows = sorted(year_rows, key=lambda r: -r["value_kusd"])[:5]
        for r in top_flows:
            lines.append(f"  {r['exporter']} → {r['importer']}: "
                         f"{r['value_kusd']:,.0f} kUSD  {r['quantity_tonnes']:,.0f} t")
        lines.append("")

    return "\n".join(lines)


def _build_year_summaries(rows: List[Dict], mineral: str) -> Dict[str, str]:
    """One rich text doc per year across all files (good for RAG retrieval by year)."""
    by_year: Dict[str, List[Dict]] = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)

    summaries = {}
    for year, year_rows in by_year.items():
        total_val = sum(r["value_kusd"] for r in year_rows) / 1000  # → MUSD
        total_qty = sum(r["quantity_tonnes"] for r in year_rows) / 1000  # → kt

        exp_totals: Dict[str, float] = {}
        imp_totals: Dict[str, float] = {}
        pair_totals: Dict[str, float] = {}

        for r in year_rows:
            exp_totals[r["exporter"]] = exp_totals.get(r["exporter"], 0) + r["value_kusd"]
            imp_totals[r["importer"]] = imp_totals.get(r["importer"], 0) + r["value_kusd"]
            pair = f"{r['exporter']}→{r['importer']}"
            pair_totals[pair] = pair_totals.get(pair, 0) + r["value_kusd"]

        top_exp = sorted(exp_totals.items(), key=lambda x: -x[1])[:10]
        top_imp = sorted(imp_totals.items(), key=lambda x: -x[1])[:10]
        top_pairs = sorted(pair_totals.items(), key=lambda x: -x[1])[:10]

        lines = [
            f"{mineral.title()} Global Trade — {year} (CEPII data)",
            f"Total world trade: {total_val:,.1f} million USD, {total_qty:,.1f} thousand tonnes",
            f"Number of bilateral trade relationships: {len(year_rows)}",
            "",
            "Top exporters by value:",
        ]
        for c, v in top_exp:
            share = 100 * v / sum(exp_totals.values()) if exp_totals else 0
            lines.append(f"  {c}: {v/1000:,.1f} MUSD ({share:.1f}% of world exports)")

        lines += ["", "Top importers by value:"]
        for c, v in top_imp:
            share = 100 * v / sum(imp_totals.values()) if imp_totals else 0
            lines.append(f"  {c}: {v/1000:,.1f} MUSD ({share:.1f}% of world imports)")

        lines += ["", "Top bilateral trade corridors:"]
        for pair, v in top_pairs:
            lines.append(f"  {pair}: {v/1000:,.1f} MUSD")

        summaries[year] = "\n".join(lines)

    return summaries


def _top_n(rows: List[Dict], field: str, n: int) -> List[str]:
    totals: Dict[str, float] = {}
    for r in rows:
        totals[r[field]] = totals.get(r[field], 0) + r["value_kusd"]
    return [c for c, _ in sorted(totals.items(), key=lambda x: -x[1])[:n]]


def list_products(input_path: Path, n: int = 30) -> None:
    """Print top HS codes found in the files to help identify the right filter."""
    files = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    counts: Dict[str, int] = {}
    for csv_path in files[:3]:  # sample first 3 files
        try:
            with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                cols = _detect_columns(list(reader.fieldnames))
                if "product" not in cols:
                    print(f"No product column in {csv_path.name}")
                    continue
                for i, row in enumerate(reader):
                    if i > 50_000:
                        break
                    k = str(row[cols["product"]]).strip()
                    counts[k] = counts.get(k, 0) + 1
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    print(f"\nTop {n} HS codes (by frequency):")
    for code, cnt in sorted(counts.items(), key=lambda x: -x[1])[:n]:
        print(f"  {code:>10}  {cnt:>8} rows")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Ingest CEPII trade flow CSVs into RAG corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True,
                   help="Path to CEPII CSV file or directory of CSVs")
    p.add_argument("--mineral", default="graphite",
                   help="Mineral name label (default: graphite)")
    p.add_argument("--hs", default=None,
                   help="HS product code(s) to filter, comma-separated "
                        "(e.g. 250410 for natural graphite). Omit to keep all rows.")
    p.add_argument("--out-docs", default="data/documents/cepii",
                   help="Output directory for text docs (default: data/documents/cepii)")
    p.add_argument("--out-csv", default=None,
                   help="Optional canonical CSV output path "
                        "(default: data/canonical/cepii_<mineral>.csv)")
    p.add_argument("--list-products", action="store_true",
                   help="List HS codes found in the files then exit")
    p.add_argument("--max-rows", type=int, default=500_000,
                   help="Max rows per file (default: 500000)")

    args = p.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} does not exist")
        sys.exit(1)

    if args.list_products:
        list_products(input_path)
        return

    hs_codes = [h.strip() for h in args.hs.split(",")] if args.hs else None
    out_docs = Path(args.out_docs)
    out_csv = Path(args.out_csv) if args.out_csv else \
              Path(f"data/canonical/cepii_{args.mineral}.csv")

    stats = ingest_cepii(
        input_path=input_path,
        mineral=args.mineral,
        hs_codes=hs_codes,
        out_docs_dir=out_docs,
        out_csv=out_csv,
        max_rows=args.max_rows,
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(json.dumps(stats, indent=2))
    print()
    print("Next steps:")
    print(f"  1. Re-index RAG:  python scripts/reindex_rag.py")
    print(f"  2. Or in the app: Just RAG tab → 'Rebuild search index'")
    print(f"  3. Calibrate scenario from canonical CSV:")
    print(f"     python scripts/calibrate_from_comtrade.py --input {out_csv}")


if __name__ == "__main__":
    main()
