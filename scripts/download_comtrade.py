"""
Download partner-level trade data from UN Comtrade.
"""

import requests
import pandas as pd
import time
from pathlib import Path


def download_bilateral_trade():
    """
    Download graphite trade data for multiple partners.
    """

    # UN Comtrade API v2 (free, no key needed)
    BASE_URL = "https://comtradeplus.un.org/api/data"

    partners = {
        "China": "CHN",
        "Mexico": "MEX",
        "Canada": "CAN",
        "Brazil": "BRA",
        "Mozambique": "MOZ",
        "India": "IND",
        "South Korea": "KOR",
        "Japan": "JPN",
    }

    all_data = []

    for country_name, country_code in partners.items():
        print(f"\n📥 Downloading {country_name} → USA...")

        for year in range(2000, 2025):
            try:
                # Build query
                params = {
                    "year": year,
                    "reporter": "USA",
                    "partner": country_code,
                    "commodity": "2504",  # Natural graphite
                    "flow": "import",
                    "format": "json",
                }

                response = requests.get(BASE_URL, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if "data" in data and data["data"]:
                        for record in data["data"]:
                            all_data.append(
                                {
                                    "year": year,
                                    "partner": country_name,
                                    "partner_code": country_code,
                                    "trade_value_usd": record.get("primaryValue", 0),
                                    "quantity_kg": record.get("netWgt", 0),
                                    "commodity": "graphite",
                                    "hs_code": "2504",
                                }
                            )
                        print(f"  ✓ {year}: {record.get('primaryValue', 0):,} USD")
                    else:
                        print(f"  - {year}: No data")
                else:
                    print(f"  ✗ {year}: Error {response.status_code}")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"  ✗ {year}: {e}")
                continue

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save
    output_path = Path("data/canonical/comtrade_graphite_bilateral.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Downloaded {len(df)} records")
    print(f"📁 Saved to: {output_path}")
    print(f"\n📊 Summary:")
    print(df.groupby("partner")["trade_value_usd"].sum().sort_values(ascending=False))

    return df


if __name__ == "__main__":
    download_bilateral_trade()
