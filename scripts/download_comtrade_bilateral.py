"""
Download BILATERAL trade data (country-to-country).
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class ComtradeBilateral:
    def __init__(self) -> None:
        self.api_key = os.getenv("COMTRADE_API_KEY")
        if not self.api_key:
            raise ValueError("COMTRADE_API_KEY not set. Add to .env")
        self.base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

    def download_bilateral(self) -> pd.DataFrame:
        """Download bilateral flows: each reporter with each specific partner."""
        print("=" * 70)
        print("DOWNLOADING BILATERAL TRADE FLOWS")
        print("=" * 70)

        all_data = []

        reporters = {
            "842": "USA",
            "156": "China",
            "484": "Mexico",
            "76": "Brazil",
            "124": "Canada",
        }

        years = ["2020", "2021", "2022", "2023"]

        for rep_code, rep_name in reporters.items():
            print(f"\n📥 {rep_name} bilateral imports")
            print("-" * 50)

            for year in years:
                print(f"  {year}...", end=" ", flush=True)

                params = {
                    "reporterCode": rep_code,
                    "partnerCode": "all",
                    "cmdCode": "2504",
                    "period": year,
                    "flowCode": "M",
                    "subscription-key": self.api_key,
                }

                try:
                    response = requests.get(self.base_url, params=params, timeout=30)

                    if response.status_code == 200:
                        data = response.json()

                        if data and "data" in data:
                            records = data["data"]
                            print(f"✅ {len(records)} partners")

                            for rec in records:
                                if rec.get("partnerCode") == 0:
                                    continue

                                all_data.append(
                                    {
                                        "year": year,
                                        "reporter": rep_name,
                                        "reporter_code": rep_code,
                                        "partner": rec.get("partnerDesc", "Unknown"),
                                        "partner_code": rec.get("partnerCode", ""),
                                        "flow": "import",
                                        "commodity_code": "2504",
                                        "trade_value_usd": rec.get("primaryValue", 0),
                                        "quantity_kg": rec.get("netWgt", 0),
                                    }
                                )
                        else:
                            print("❌ No data")
                    else:
                        print(f"❌ Error {response.status_code}")

                except Exception as e:
                    print(f"❌ {e}")

                time.sleep(1)

        df = pd.DataFrame(all_data)
        output_path = Path("data/canonical/comtrade_bilateral_flows.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("✅ BILATERAL DATA DOWNLOADED")
        print("=" * 70)
        print(f"Records: {len(df)}")
        if len(df) > 0:
            print(f"Reporters: {df['reporter'].nunique()}")
            print(f"Partners: {df['partner'].nunique()}")
        print(f"Output: {output_path}")

        if len(df) > 0:
            print("\n📊 Sample flows:")
            sample = df.nlargest(10, "trade_value_usd")[
                ["year", "reporter", "partner", "trade_value_usd"]
            ]
            print(sample.to_string(index=False))

        return df


if __name__ == "__main__":
    downloader = ComtradeBilateral()
    downloader.download_bilateral()
