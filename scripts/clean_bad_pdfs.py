"""
Remove non-PDF files that were downloaded as PDFs.
"""

from pathlib import Path


def is_valid_pdf(pdf_path: Path) -> bool:
    """Check if file is actually a PDF (starts with %PDF)."""
    try:
        with open(pdf_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


def clean_fake_pdfs() -> list:
    """Find and remove fake PDFs (e.g. HTML error pages saved as .pdf)."""
    print("🔍 Scanning for fake PDFs...")

    docs_root = Path("data/documents")
    if not docs_root.exists():
        print(f"   Directory not found: {docs_root}")
        return []

    all_pdfs = list(docs_root.rglob("*.pdf"))
    print(f"   Found {len(all_pdfs)} .pdf files")

    valid: list[Path] = []
    invalid: list[Path] = []

    for pdf_path in all_pdfs:
        if is_valid_pdf(pdf_path):
            valid.append(pdf_path)
        else:
            invalid.append(pdf_path)
            pdf_path.unlink()

    print(f"\n✅ Valid PDFs: {len(valid)}")
    print(f"❌ Fake PDFs removed: {len(invalid)}")

    if valid:
        print("\n📄 Valid PDFs by directory:")
        by_dir: dict[str, int] = {}
        for pdf in valid:
            parent = pdf.parent.name
            by_dir[parent] = by_dir.get(parent, 0) + 1
        for dir_name, count in sorted(by_dir.items()):
            print(f"   {dir_name}: {count} PDFs")

    return valid


if __name__ == "__main__":
    valid_pdfs = clean_fake_pdfs()

    if valid_pdfs:
        print(f"\n✅ Ready to convert {len(valid_pdfs)} valid PDFs to text")
        print("   Run: python scripts/extract_pdf_text.py")
    else:
        print("\n⚠️  No valid PDFs found!")
