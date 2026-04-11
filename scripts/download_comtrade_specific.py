"""
Download SPECIFIC bilateral trade flows.
Explicit country-to-country requests (reporter + partner codes).
"""

import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class ComtradeSpecific:
    def __init__(self) -> None:
        self.api_key = os.getenv("COMTRADE_API_KEY")
        if not self.api_key:
            raise ValueError("COMTRADE_API_KEY not set. Add to .env")

        self.endpoints = [
            "https://comtradeapi.un.org/data/v1/get/C/A/HS",
            "https://comtradeapi.un.org/public/v1/get/C/A/HS",
        ]

    def get_bilateral_flow(
        self,
        reporter_code: str,
        partner_code: str,
        year: str,
        endpoint_idx: int = 0,
    ) -> dict[str, Any] | None:
        """Get one specific bilateral flow (reporter imports from partner)."""
        params = {
            "reporterCode": reporter_code,
            "partnerCode": partner_code,
            "cmdCode": "2504",
            "period": year,
            "flowCode": "M",
            "subscription-key": self.api_key,
        }

        try:
            response = requests.get(
                self.endpoints[endpoint_idx],
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            if endpoint_idx == 0 and len(self.endpoints) > 1:
                return self.get_bilateral_flow(
                    reporter_code, partner_code, year, endpoint_idx=1
                )
            print(f"      ❌ Error {response.status_code}: {response.text[:100]}")
            return None

        except Exception as e:
            print(f"      ❌ Exception: {e}")
            return None

    # Reporter codes. Partner sets: Africa, South America, East Asia.
    USA_CODE = "842"
    USA_NAME = "USA"
    CHINA_CODE = "156"
    CHINA_NAME = "China"

    # Years for all downloads (1995–2026 inclusive).
    YEARS: list[str] = [str(y) for y in range(1995, 2027)]

    AFRICA: dict[str, str] = {
        "12": "Algeria",
        "24": "Angola",
        "204": "Benin",
        "72": "Botswana",
        "854": "Burkina Faso",
        "108": "Burundi",
        "132": "Cabo Verde",
        "120": "Cameroon",
        "140": "Central African Rep.",
        "148": "Chad",
        "174": "Comoros",
        "178": "Congo",
        "180": "Dem. Rep. of the Congo",
        "384": "Côte d'Ivoire",
        "262": "Djibouti",
        "818": "Egypt",
        "226": "Equatorial Guinea",
        "232": "Eritrea",
        "748": "Eswatini",
        "231": "Ethiopia",
        "266": "Gabon",
        "270": "Gambia",
        "288": "Ghana",
        "324": "Guinea",
        "624": "Guinea-Bissau",
        "404": "Kenya",
        "426": "Lesotho",
        "430": "Liberia",
        "434": "Libya",
        "450": "Madagascar",
        "454": "Malawi",
        "466": "Mali",
        "478": "Mauritania",
        "480": "Mauritius",
        "504": "Morocco",
        "508": "Mozambique",
        "516": "Namibia",
        "562": "Niger",
        "566": "Nigeria",
        "646": "Rwanda",
        "678": "São Tomé and Príncipe",
        "686": "Senegal",
        "690": "Seychelles",
        "694": "Sierra Leone",
        "706": "Somalia",
        "710": "South Africa",
        "728": "South Sudan",
        "729": "Sudan",
        "834": "Tanzania",
        "768": "Togo",
        "788": "Tunisia",
        "800": "Uganda",
        "894": "Zambia",
        "716": "Zimbabwe",
    }

    SOUTH_AMERICA: dict[str, str] = {
        "32": "Argentina",
        "68": "Bolivia",
        "76": "Brazil",
        "152": "Chile",
        "170": "Colombia",
        "218": "Ecuador",
        "328": "Guyana",
        "600": "Paraguay",
        "604": "Peru",
        "740": "Suriname",
        "858": "Uruguay",
        "862": "Venezuela",
    }

    EAST_ASIA: dict[str, str] = {
        "156": "China",
        "344": "China, Hong Kong SAR",
        "446": "China, Macao SAR",
        "392": "Japan",
        "408": "Dem. People's Rep. of Korea",
        "410": "South Korea",
        "496": "Mongolia",
        "158": "Taiwan",
    }

    def _region_partners(self) -> dict[str, str]:
        """Combined Africa + South America + East Asia (excl. China as reporter)."""
        out: dict[str, str] = {}
        out.update(self.AFRICA)
        out.update(self.SOUTH_AMERICA)
        out.update(self.EAST_ASIA)
        return out

    def _download_imports_from_regions(
        self,
        reporter_code: str,
        reporter_name: str,
        partners: dict[str, str],
        years: list[str],
        out_filename: str,
        title: str,
    ) -> pd.DataFrame | None:
        """Generic: one reporter's imports from each partner, all years."""
        print(f"\n{title}")
        print("-" * 50)
        all_data: list[dict[str, Any]] = []
        for part_code, part_name in sorted(partners.items(), key=lambda x: x[1]):
            print(f"  {part_name:35s}", end=" ", flush=True)
            flow_data: list[dict[str, Any]] = []
            for year in years:
                data = self.get_bilateral_flow(reporter_code, part_code, year)
                if data and "data" in data and data["data"]:
                    for rec in data["data"]:
                        trade_value = rec.get("primaryValue", 0)
                        if trade_value and trade_value > 0:
                            flow_data.append(
                                {
                                    "year": year,
                                    "reporter": reporter_name,
                                    "reporter_code": reporter_code,
                                    "partner": part_name,
                                    "partner_code": part_code,
                                    "flow": "import",
                                    "commodity_code": "2504",
                                    "trade_value_usd": trade_value,
                                    "quantity_kg": rec.get("netWgt", 0),
                                }
                            )
                time.sleep(0.5)
            if flow_data:
                all_data.extend(flow_data)
                total_value = sum(f["trade_value_usd"] for f in flow_data)
                print(f"✅ ${total_value:,.0f}")
            else:
                print("- (no trade)")

        if not all_data:
            print("\n❌ No data retrieved.")
            return None
        df = pd.DataFrame(all_data)
        output_path = Path("data/canonical") / out_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  → {output_path} ({len(df)} rows)")
        return df

    def download_usa_imports_regions(self) -> pd.DataFrame | None:
        """
        Download USA imports (HS 2504) from ALL African, South American,
        and East Asian countries.
        """
        partners = self._region_partners()
        years = self.YEARS
        print("=" * 70)
        print("USA IMPORTS FROM AFRICA, SOUTH AMERICA & EAST ASIA")
        print("=" * 70)
        print(f"Partners: {len(partners)}. API calls: {len(partners) * len(years)} (rate limited)\n")
        df = self._download_imports_from_regions(
            self.USA_CODE,
            self.USA_NAME,
            partners,
            years,
            "comtrade_usa_imports_africa_samerica_eastasia.csv",
            "USA imports by partner",
        )
        if df is not None:
            print("\n📊 TOP 10 PARTNERS:")
            top = (
                df.groupby("partner", as_index=False)["trade_value_usd"]
                .sum()
                .nlargest(10, "trade_value_usd")
            )
            print(top.to_string(index=False))
        return df

    def download_china_imports_regions(self) -> pd.DataFrame | None:
        """
        Download China imports (HS 2504) from ALL African, South American,
        and East Asian countries (excluding China itself).
        """
        partners = {
            k: v for k, v in self._region_partners().items() if k != self.CHINA_CODE
        }
        years = self.YEARS
        print("=" * 70)
        print("CHINA IMPORTS FROM AFRICA, SOUTH AMERICA & EAST ASIA")
        print("=" * 70)
        print(f"Partners: {len(partners)}. API calls: {len(partners) * len(years)} (rate limited)\n")
        df = self._download_imports_from_regions(
            self.CHINA_CODE,
            self.CHINA_NAME,
            partners,
            years,
            "comtrade_china_imports_africa_samerica_eastasia.csv",
            "China imports by partner",
        )
        if df is not None:
            print("\n📊 TOP 10 PARTNERS:")
            top = (
                df.groupby("partner", as_index=False)["trade_value_usd"]
                .sum()
                .nlargest(10, "trade_value_usd")
            )
            print(top.to_string(index=False))
        return df

    def download_intra_region_flows(self) -> pd.DataFrame | None:
        """
        Download trade between all region countries: each country's imports
        from every other (Africa + South America + East Asia). ~74×73×4 calls.
        """
        partners = self._region_partners()
        years = self.YEARS
        n = len(partners)
        total_calls = n * (n - 1) * len(years)
        print("=" * 70)
        print("TRADE BETWEEN REGION COUNTRIES (IMPORTS BY REPORTER)")
        print("=" * 70)
        print(f"Reporters & partners: {n} countries. API calls: {total_calls}")
        print("(Rate limited; expect ~2–3 hours)\n")

        all_data: list[dict[str, Any]] = []
        for rep_code, rep_name in sorted(partners.items(), key=lambda x: x[1]):
            print(f"\n📥 {rep_name} imports from:", flush=True)
            for part_code, part_name in sorted(partners.items(), key=lambda x: x[1]):
                if rep_code == part_code:
                    continue
                flow_data: list[dict[str, Any]] = []
                for year in years:
                    data = self.get_bilateral_flow(rep_code, part_code, year)
                    if data and "data" in data and data["data"]:
                        for rec in data["data"]:
                            trade_value = rec.get("primaryValue", 0)
                            if trade_value and trade_value > 0:
                                flow_data.append(
                                    {
                                        "year": year,
                                        "reporter": rep_name,
                                        "reporter_code": rep_code,
                                        "partner": part_name,
                                        "partner_code": part_code,
                                        "flow": "import",
                                        "commodity_code": "2504",
                                        "trade_value_usd": trade_value,
                                        "quantity_kg": rec.get("netWgt", 0),
                                    }
                                )
                    time.sleep(0.5)
                if flow_data:
                    all_data.extend(flow_data)
                    tot = sum(f["trade_value_usd"] for f in flow_data)
                    print(f"  {part_name:35s} ✅ ${tot:,.0f}", flush=True)

        if not all_data:
            print("\n❌ No data retrieved.")
            return None
        df = pd.DataFrame(all_data)
        output_path = Path("data/canonical/comtrade_intra_region_flows.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print("\n" + "=" * 70)
        print("✅ INTRA-REGION FLOWS DOWNLOADED")
        print("=" * 70)
        print(f"Total flows: {len(df)}")
        print(f"Output: {output_path}")
        return df

    def download_complete_network(self) -> pd.DataFrame | None:
        """
        Download complete bilateral network by requesting each reporter–partner
        pair explicitly (no "all" partners).
        """
        print("=" * 70)
        print("DOWNLOADING COMPLETE BILATERAL NETWORK")
        print("=" * 70)

        countries = {
            "842": "USA",
            "156": "China",
            "484": "Mexico",
            "76": "Brazil",
            "124": "Canada",
            "392": "Japan",
            "410": "South Korea",
            "356": "India",
            "508": "Mozambique",
            "276": "Germany",
            "826": "United Kingdom",
            "250": "France",
            "702": "Singapore",
            "36": "Australia",
            "710": "South Africa",
        }

        years = self.YEARS

        all_data: list[dict[str, Any]] = []
        n_countries = len(countries)
        total_requests = n_countries * (n_countries - 1) * len(years)
        current = 0

        print(f"\nTotal bilateral flows to request: {total_requests}")
        print("(Rate limited; expect ~20–30 minutes)\n")

        for rep_code, rep_name in countries.items():
            print(f"\n📥 {rep_name} imports from:")
            print("-" * 50)

            for part_code, part_name in countries.items():
                if rep_code == part_code:
                    continue

                print(f"  {part_name:20s}", end=" ", flush=True)
                flow_data: list[dict[str, Any]] = []

                for year in years:
                    current += 1
                    data = self.get_bilateral_flow(rep_code, part_code, year)

                    if data and "data" in data and data["data"]:
                        for rec in data["data"]:
                            trade_value = rec.get("primaryValue", 0)
                            if trade_value and trade_value > 0:
                                flow_data.append(
                                    {
                                        "year": year,
                                        "reporter": rep_name,
                                        "reporter_code": rep_code,
                                        "partner": part_name,
                                        "partner_code": part_code,
                                        "flow": "import",
                                        "commodity_code": "2504",
                                        "trade_value_usd": trade_value,
                                        "quantity_kg": rec.get("netWgt", 0),
                                    }
                                )

                    time.sleep(0.5)

                if flow_data:
                    all_data.extend(flow_data)
                    total_value = sum(f["trade_value_usd"] for f in flow_data)
                    print(f"✅ ${total_value:,.0f}")
                else:
                    print("- (no trade)")

        df = pd.DataFrame(all_data)

        if len(df) == 0:
            print("\n❌ No data retrieved. Check API key and endpoint.")
            return None

        output_path = Path("data/canonical/comtrade_bilateral_complete.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("✅ COMPLETE BILATERAL NETWORK DOWNLOADED")
        print("=" * 70)
        print(f"Total flows: {len(df)}")
        print(f"Reporters: {df['reporter'].nunique()}")
        print(f"Partners: {df['partner'].nunique()}")
        print(f"Years: {sorted(df['year'].unique())}")
        print(f"Total trade: ${df['trade_value_usd'].sum():,.0f}")
        print(f"Output: {output_path}")

        print("\n📊 TOP 10 BILATERAL FLOWS:")
        top = df.nlargest(10, "trade_value_usd")[
            ["year", "reporter", "partner", "trade_value_usd"]
        ]
        print(top.to_string(index=False))

        return df


if __name__ == "__main__":
    print("1) USA imports from Africa + South America + East Asia")
    print("2) China imports from Africa + South America + East Asia")
    print("3) Trade between these countries (each other) — ~3 hours")
    print("4) All of the above (USA + China + intra-region)")
    print("5) Full bilateral network (other preset countries)")
    choice = input("Choice (1–5) [1]: ").strip() or "1"
    response = input("Continue? (y/n): ").strip().lower()
    if response != "y":
        print("Cancelled.")
        raise SystemExit(0)

    downloader = ComtradeSpecific()
    if choice == "2":
        downloader.download_china_imports_regions()
    elif choice == "3":
        downloader.download_intra_region_flows()
    elif choice == "4":
        downloader.download_usa_imports_regions()
        downloader.download_china_imports_regions()
        downloader.download_intra_region_flows()
    elif choice == "5":
        downloader.download_complete_network()
    else:
        downloader.download_usa_imports_regions()
