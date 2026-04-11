"""
Download Comtrade data with API key authentication.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()


class ComtradeAuthAPI:
    """
    Authenticated Comtrade API access.
    """

    def __init__(self) -> None:
        self.api_key = os.getenv("COMTRADE_API_KEY")

        if not self.api_key:
            raise ValueError("COMTRADE_API_KEY not found in .env file!")

        self.base_url = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

        print(f"✅ API Key loaded (ends with: ...{self.api_key[-4:]})")

    def get_trade_data(
        self,
        reporter_code: str,
        partner_code: str = "0",  # 0 = World/All
        commodity_code: str = "2504",
        year: str = "2023",
        flow_code: str = "M",  # M = imports
    ):
        """Get trade data with authentication."""
        params = {
            "reporterCode": reporter_code,
            "partnerCode": partner_code,
            "cmdCode": commodity_code,
            "period": year,
            "flowCode": flow_code,
            "subscription-key": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            print(f"❌ Error {response.status_code}: {response.text}")
            return None

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    def download_global_network(
        self,
        commodity: str = "2504",
        years: list[str] | None = None,
    ) -> pd.DataFrame:
        """Download complete global trade network."""
        if years is None:
            years = [str(y) for y in range(1996, 2025)]

        print("=" * 70)
        print("DOWNLOADING GLOBAL TRADE NETWORK (AUTHENTICATED)")
        print("=" * 70)
        print(f"Commodity: HS {commodity}")
        print(f"Years: {min(years)}-{max(years)}")
        print("=" * 70 + "\n")

        all_data = []

        reporters = {
            "842": "USA",
            "156": "China",
            "484": "Mexico",
            "76": "Brazil",
            "124": "Canada",
            "508": "Mozambique",
            "356": "India",
            "392": "Japan",
            "410": "South Korea",
        }

        for reporter_code, reporter_name in reporters.items():
            print(f"\n📥 {reporter_name} IMPORTS FROM ALL PARTNERS")
            print("-" * 50)

            for year in years:
                print(f"  Year {year}...", end=" ", flush=True)

                data = self.get_trade_data(
                    reporter_code=reporter_code,
                    partner_code="0",
                    commodity_code=commodity,
                    year=year,
                    flow_code="M",
                )

                if data and "data" in data:
                    records = data["data"]
                    print(f"✅ {len(records)} flows")

                    for record in records:
                        all_data.append(
                            {
                                "year": year,
                                "reporter": reporter_name,
                                "reporter_code": reporter_code,
                                "partner": record.get("partnerDesc", "Unknown"),
                                "partner_code": record.get("partnerCode", ""),
                                "flow": "import",
                                "commodity_code": commodity,
                                "trade_value_usd": record.get("primaryValue", 0),
                                "quantity_kg": record.get("netWgt", 0),
                            }
                        )
                else:
                    print("❌ No data")

                time.sleep(1)

        df = pd.DataFrame(all_data)

        output_path = Path("data/canonical/comtrade_network_graphite.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Total records: {len(df)}")
        if len(df) > 0:
            print(f"Unique reporters: {df['reporter'].nunique()}")
            print(f"Unique partners: {df['partner'].nunique()}")
            print(f"Total trade value: ${df['trade_value_usd'].sum():,.0f}")
        print(f"Output: {output_path}")

        if len(df) > 0:
            print("\n📊 TOP 10 TRADE FLOWS:")
            top = df.nlargest(10, "trade_value_usd")[
                ["year", "reporter", "partner", "trade_value_usd"]
            ]
            print(top.to_string(index=False))

        return df


if __name__ == "__main__":
    api = ComtradeAuthAPI()

    df = api.download_global_network(
        commodity="2504",
        years=[str(y) for y in range(1996, 2025)],
    )

    print("\n✅ Success! Network data saved to:")
    print("   data/canonical/comtrade_network_graphite.csv")
