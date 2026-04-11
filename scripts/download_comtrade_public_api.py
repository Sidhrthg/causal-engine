"""
Download global trade data using Comtrade FREE public API.
No subscription key required!
"""

import time
from pathlib import Path

import pandas as pd
import requests


class ComtradePublicAPI:
    """
    Use UN Comtrade public (free) API.
    Documentation: https://bit.ly/42JNSaZ
    """

    def __init__(self) -> None:
        self.base_url = "https://comtradeapi.un.org/public/v1"
        self.rate_limit_delay = 1  # seconds between requests

    def get_trade_data(
        self,
        reporter_code: str,
        partner_code: str,
        commodity_code: str,
        year: int,
        flow_code: str = "M",  # M = imports, X = exports
    ):
        """
        Get bilateral trade data.

        Args:
            reporter_code: ISO3 code (e.g., 'USA')
            partner_code: ISO3 code or 'all'
            commodity_code: HS code (e.g., '2504' for graphite)
            year: Year (e.g., 2023)
            flow_code: 'M' for imports, 'X' for exports
        """
        endpoint = (
            f"{self.base_url}/get/{year}/{flow_code}"
            f"/{reporter_code}/{partner_code}/{commodity_code}"
        )

        try:
            response = requests.get(endpoint, timeout=30)

            if response.status_code == 200:
                return response.json()
            print(f"Error {response.status_code}: {response.text}")
            return None

        except Exception as e:
            print(f"Request failed: {e}")
            return None

    def download_global_matrix(
        self,
        commodity_code: str = "2504",  # Graphite
        years: range = None,
        output_file: str = "data/comtrade_global_graphite.csv",
    ) -> pd.DataFrame:
        """
        Download complete global trade matrix.

        Strategy: Get USA imports from all partners, then major exporters to all.
        """
        if years is None:
            years = range(2018, 2024)

        print("=" * 70)
        print("DOWNLOADING GLOBAL TRADE MATRIX (FREE PUBLIC API)")
        print("=" * 70)
        print(f"Commodity: HS {commodity_code}")
        print(f"Years: {min(years)}-{max(years)}")
        print("=" * 70 + "\n")

        all_data = []

        # Step 1: Get USA imports from all partners
        print("\n📥 USA IMPORTS FROM ALL COUNTRIES")
        print("-" * 40)

        for year in years:
            print(f"\nYear {year}:")

            data = self.get_trade_data(
                reporter_code="USA",
                partner_code="all",
                commodity_code=commodity_code,
                year=year,
                flow_code="M",
            )

            if data and "data" in data:
                records = data["data"]
                print(f"  ✅ Retrieved {len(records)} trade flows")

                for record in records:
                    all_data.append(
                        {
                            "year": year,
                            "reporter": "USA",
                            "reporter_code": record.get("reporterCode", "USA"),
                            "partner": record.get("partnerDesc", "Unknown"),
                            "partner_code": record.get("partnerCode", ""),
                            "flow": "import",
                            "commodity": f"HS{commodity_code}",
                            "trade_value_usd": record.get("primaryValue", 0),
                            "quantity_kg": record.get("netWgt", 0),
                            "quantity_unit": record.get("qtUnitCode", ""),
                        }
                    )
            else:
                print(f"  ❌ No data for {year}")

            time.sleep(self.rate_limit_delay)

        # Step 2: Get major exporters to all countries
        major_exporters = ["CHN", "BRA", "IND", "MEX", "CAN", "MOZ", "JPN"]

        print("\n\n📤 MAJOR EXPORTERS TO ALL COUNTRIES")
        print("-" * 40)

        for exporter in major_exporters:
            print(f"\n{exporter} exports:")

            for year in years:
                data = self.get_trade_data(
                    reporter_code=exporter,
                    partner_code="all",
                    commodity_code=commodity_code,
                    year=year,
                    flow_code="X",
                )

                if data and "data" in data:
                    records = data["data"]
                    print(f"  {year}: {len(records)} flows")

                    for record in records:
                        all_data.append(
                            {
                                "year": year,
                                "reporter": exporter,
                                "reporter_code": record.get("reporterCode", exporter),
                                "partner": record.get("partnerDesc", "Unknown"),
                                "partner_code": record.get("partnerCode", ""),
                                "flow": "export",
                                "commodity": f"HS{commodity_code}",
                                "trade_value_usd": record.get("primaryValue", 0),
                                "quantity_kg": record.get("netWgt", 0),
                                "quantity_unit": record.get("qtUnitCode", ""),
                            }
                        )
                else:
                    print(f"  {year}: No data")

                time.sleep(self.rate_limit_delay)

        df = pd.DataFrame(all_data)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print("\n" + "=" * 70)
        print("✅ DOWNLOAD COMPLETE")
        print("=" * 70)
        print(f"Total records: {len(df)}")
        if len(df) > 0:
            print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
            print(f"Unique reporters: {df['reporter'].nunique()}")
            print(f"Unique partners: {df['partner'].nunique()}")
        print(f"Output file: {output_path}")
        print("=" * 70)

        if len(df) > 0:
            print("\n📊 TOP TRADE FLOWS:")
            top_flows = df.nlargest(10, "trade_value_usd")[
                ["year", "reporter", "partner", "trade_value_usd"]
            ]
            print(top_flows.to_string(index=False))

        return df


if __name__ == "__main__":
    api = ComtradePublicAPI()

    df = api.download_global_matrix(
        commodity_code="2504",
        years=range(2018, 2024),
        output_file="data/canonical/comtrade_global_graphite.csv",
    )

    print("\n✅ Done! Check data/canonical/comtrade_global_graphite.csv")
