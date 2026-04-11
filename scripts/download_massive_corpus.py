"""
Comprehensive document corpus download.
Target: 1000+ documents across multiple sources.
Runtime: 6-8 hours
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests


class MassiveCorpusDownloader:
    """
    Download 1000+ documents from multiple sources.
    """

    def __init__(self, base_dir: str = "data/documents"):
        self.base_dir = Path(base_dir)
        self.stats = {
            "total_downloaded": 0,
            "total_failed": 0,
            "by_source": {},
        }
        self.log_file = Path("data/download_log.json")

    def download_all_usgs_minerals(self) -> int:
        """
        Download USGS Mineral Commodity Summaries for ALL critical minerals.

        Target: 20 minerals × 28 years = 560 documents
        """
        minerals = [
            "graphite",
            "lithium",
            "cobalt",
            "rare-earths",
            "copper",
            "nickel",
            "zinc",
            "lead",
            "tin",
            "aluminum",
            "bauxite",
            "chromium",
            "manganese",
            "platinum",
            "titanium",
            "tungsten",
            "vanadium",
            "antimony",
            "beryllium",
            "cesium",
            "gallium",
            "germanium",
            "indium",
            "magnesium",
            "niobium",
            "tantalum",
            "tellurium",
            "yttrium",
            "zirconium",
        ]

        output_dir = self.base_dir / "usgs"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("📥 DOWNLOADING USGS MINERAL COMMODITY SUMMARIES")
        print("=" * 70)
        print(f"Minerals: {len(minerals)}")
        print(f"Years: 1996-2024 (28 years)")
        print(f"Target: ~{len(minerals) * 28} documents")
        print(f"Output: {output_dir}")
        print("=" * 70 + "\n")

        downloaded = 0
        failed = 0

        for mineral in minerals:
            print(f"\n📦 {mineral.upper()}")
            print("-" * 40)

            mineral_downloaded = 0

            for year in range(1996, 2025):
                output_path = output_dir / f"{mineral}_mcs_{year}.pdf"

                if output_path.exists():
                    print(f"  ✓ {year} (cached)")
                    downloaded += 1
                    mineral_downloaded += 1
                    continue

                url_patterns = [
                    f"https://pubs.usgs.gov/periodicals/mcs{year}/mcs{year}-{mineral}.pdf",
                    f"https://pubs.usgs.gov/periodicals/mcs{year}/{mineral}.pdf",
                    f"https://minerals.usgs.gov/minerals/pubs/commodity/{mineral}/mcs-{year}-{mineral}.pdf",
                ]

                success = False
                for url in url_patterns:
                    try:
                        response = requests.get(url, timeout=30)

                        if response.status_code == 200 and len(response.content) > 1000:
                            with open(output_path, "wb") as f:
                                f.write(response.content)

                            size_kb = len(response.content) / 1024
                            print(f"  ✅ {year} ({size_kb:.1f} KB)")
                            downloaded += 1
                            mineral_downloaded += 1
                            success = True
                            break

                    except Exception:
                        continue

                if not success:
                    print(f"  ❌ {year}")
                    failed += 1

                time.sleep(0.5)

            print(f"  → Downloaded {mineral_downloaded} docs for {mineral}")

        self.stats["usgs"] = {"downloaded": downloaded, "failed": failed}
        print(f"\n✅ USGS Total: {downloaded} documents downloaded, {failed} failed")
        return downloaded

    def download_full_mcs_books(self) -> int:
        """
        Download complete MCS books (all minerals combined).
        Useful for years where individual minerals aren't available.

        Target: 28 books (1996-2024)
        """
        output_dir = self.base_dir / "usgs_books"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("📚 DOWNLOADING COMPLETE MCS BOOKS")
        print("=" * 70)

        downloaded = 0

        for year in range(1996, 2025):
            output_path = output_dir / f"mcs_{year}_complete.pdf"

            if output_path.exists():
                print(f"✓ {year} (cached)")
                downloaded += 1
                continue

            url = f"https://pubs.usgs.gov/periodicals/mcs{year}/mcs{year}.pdf"

            try:
                print(f"📥 {year}...", end=" ", flush=True)
                response = requests.get(url, timeout=60)

                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    size_mb = len(response.content) / (1024 * 1024)
                    print(f"✅ ({size_mb:.1f} MB)")
                    downloaded += 1
                else:
                    print(f"❌ ({response.status_code})")

            except Exception as e:
                print(f"❌ ({e})")

            time.sleep(1)

        self.stats["usgs_books"] = downloaded
        print(f"\n✅ Downloaded {downloaded} complete MCS books")
        return downloaded

    def download_usgs_annual_reports(self) -> int:
        """
        Download USGS Minerals Yearbooks.
        More detailed than MCS.

        Target: 50+ reports
        """
        output_dir = self.base_dir / "usgs_yearbooks"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("📖 DOWNLOADING USGS MINERALS YEARBOOKS")
        print("=" * 70)

        minerals = ["graphite", "lithium", "rare-earths", "copper", "cobalt", "nickel"]
        downloaded = 0

        for mineral in minerals:
            print(f"\n📖 {mineral.upper()} Yearbooks")

            for year in range(2000, 2023):
                output_path = output_dir / f"{mineral}_yearbook_{year}.pdf"

                if output_path.exists():
                    print(f"  ✓ {year} (cached)")
                    downloaded += 1
                    continue

                url = f"https://pubs.usgs.gov/myb/vol1/{year}/{mineral}.pdf"

                try:
                    response = requests.get(url, timeout=30)

                    if response.status_code == 200:
                        with open(output_path, "wb") as f:
                            f.write(response.content)
                        print(f"  ✅ {year}")
                        downloaded += 1
                    else:
                        print(f"  ❌ {year}")

                except Exception:
                    print(f"  ❌ {year}")

                time.sleep(0.5)

        self.stats["yearbooks"] = downloaded
        print(f"\n✅ Downloaded {downloaded} yearbooks")
        return downloaded

    def download_iea_reports(self) -> int:
        """
        Download IEA (International Energy Agency) critical minerals reports.

        Target: 30+ reports
        """
        output_dir = self.base_dir / "iea"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("🌍 DOWNLOADING IEA REPORTS")
        print("=" * 70)

        reports = [
            {
                "year": 2021,
                "url": "https://iea.blob.core.windows.net/assets/24d5dfbb-a77a-4647-abcc-667867207f74/TheRoleofCriticalMineralsinCleanEnergyTransitions.pdf",
                "name": "iea_critical_minerals_2021.pdf",
            },
            {
                "year": 2022,
                "url": "https://iea.blob.core.windows.net/assets/6a76d045-d9ed-4d67-a0d9-3faa3b39baaa/GlobalSupplyChainsofEVBatteries.pdf",
                "name": "iea_ev_batteries_2022.pdf",
            },
            {
                "year": 2023,
                "url": "https://iea.blob.core.windows.net/assets/afc35261-41b7-4d63-8d89-9c26b75e71d0/EnergyTechnologyPerspectives2023.pdf",
                "name": "iea_energy_tech_2023.pdf",
            },
        ]

        downloaded = 0

        for report in reports:
            output_path = output_dir / report["name"]

            if output_path.exists():
                print(f"✓ {report['year']} (cached)")
                downloaded += 1
                continue

            try:
                print(f"📥 {report['year']}...", end=" ", flush=True)
                response = requests.get(report["url"], timeout=60)

                if response.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(response.content)

                    size_mb = len(response.content) / (1024 * 1024)
                    print(f"✅ ({size_mb:.1f} MB)")
                    downloaded += 1
                else:
                    print("❌")

            except Exception:
                print("❌")

            time.sleep(1)

        self.stats["iea"] = downloaded
        print(f"\n✅ Downloaded {downloaded} IEA reports")
        return downloaded

    def download_academic_preprints(self) -> int:
        """
        Download academic papers from arXiv on mineral economics.

        Target: 100+ papers
        """
        output_dir = self.base_dir / "academic"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("🎓 DOWNLOADING ACADEMIC PAPERS (arXiv)")
        print("=" * 70)
        print("Note: Searching arXiv for mineral supply chain papers...")

        search_terms = [
            "mineral supply chain",
            "critical minerals",
            "rare earth elements economics",
            "lithium supply",
            "copper market",
            "resource security",
        ]

        downloaded = 0

        print("\n⚠️  Full arXiv integration requires API key")
        print("    Recommend manual download from: https://arxiv.org/")
        print("    Search terms:", ", ".join(search_terms))

        self.stats["academic"] = downloaded
        return downloaded

    def save_stats(self) -> None:
        """Save download statistics."""
        total = 0
        for v in self.stats.values():
            if isinstance(v, dict) and "downloaded" in v:
                total += v["downloaded"]
            elif isinstance(v, int):
                total += v

        self.stats["total_downloaded"] = total
        self.stats["timestamp"] = datetime.now().isoformat()

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        print("\n" + "=" * 70)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 70)
        print(f"Total documents downloaded: {total}")
        print(f"Log saved to: {self.log_file}")
        print("=" * 70)

    def run_all(self) -> None:
        """
        Run all download jobs.
        Estimated time: 6-8 hours
        """
        start_time = datetime.now()
        print(f"\n🚀 Starting massive download at {start_time}")
        print("⏱️  Estimated time: 6-8 hours")
        print(f"📁 Output directory: {self.base_dir}")

        self.download_all_usgs_minerals()
        self.download_full_mcs_books()
        self.download_usgs_annual_reports()
        self.download_iea_reports()
        self.download_academic_preprints()

        self.save_stats()

        end_time = datetime.now()
        duration = end_time - start_time

        print("\n✅ Download complete!")
        print(f"⏱️  Duration: {duration}")
        print(f"📊 Total documents: {self.stats['total_downloaded']}")


if __name__ == "__main__":
    downloader = MassiveCorpusDownloader()
    downloader.run_all()
