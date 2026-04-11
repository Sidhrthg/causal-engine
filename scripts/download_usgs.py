"""
Download USGS Mineral Commodity Summaries for graphite.
"""

import time
from pathlib import Path

import requests


def download_usgs_summaries():
    """
    Download USGS graphite summaries 2000-2024.
    """

    output_dir = Path("data/documents/usgs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading USGS Mineral Commodity Summaries...")

    downloaded = 0
    failed = []

    for year in range(2000, 2025):
        # USGS URL pattern
        url = f"https://pubs.usgs.gov/periodicals/mcs{year}/mcs{year}-graphite.pdf"
        output_path = output_dir / f"graphite_mcs_{year}.pdf"

        if output_path.exists():
            print(f"✓ {year} (already exists)")
            downloaded += 1
            continue

        try:
            print(f"📥 Downloading {year}...", end=" ")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                f.write(response.content)

            print(f"✅ ({len(response.content)/1024:.1f} KB)")
            downloaded += 1
            time.sleep(1)  # Be nice to USGS servers

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print("❌ (not found)")
            else:
                print(f"❌ (error {e.response.status_code})")
            failed.append(year)
        except Exception as e:
            print(f"❌ ({e})")
            failed.append(year)

    print(f"\n✅ Downloaded {downloaded} PDFs to {output_dir}")

    if failed:
        print(f"⚠️  Failed years: {failed}")
        print("   (Some years may not have separate graphite reports)")

    return downloaded


if __name__ == "__main__":
    download_usgs_summaries()
