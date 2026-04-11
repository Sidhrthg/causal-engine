"""
Extract text from USGS PDFs for RAG indexing.
"""

from pathlib import Path
import subprocess

def extract_pdf_text():
    """Convert PDFs to text files."""
    
    pdf_dir = Path("data/documents/usgs")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDFs found")
        return
    
    print(f"📄 Found {len(pdf_files)} PDFs")
    print("Converting to text...\n")
    
    for pdf_path in pdf_files:
        txt_path = pdf_path.with_suffix('.txt')
        
        if txt_path.exists():
            print(f"✓ {pdf_path.name} (already converted)")
            continue
        
        try:
            result = subprocess.run(
                ['pdftotext', str(pdf_path), str(txt_path)],
                capture_output=True,
                timeout=10
            )
            
            if result.returncode == 0:
                # Check file s              size_kb = txt_path.stat().st_size / 1024
                print(f"✅ {pdf_path.name} → {txt_path.name} ({size_kb:.1f} KB)")
            else:
                print(f"❌ {pdf_path.name} (conversion failed)")
                
        except FileNotFoundError:
            print("❌ pdftotext not installed!")
            print("   Run: brew install poppler")
            return
        except Exception as e:
            print(f"❌ {pdf_path.name}: {e}")
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 Text files in: {pdf_dir}")

if __name__ == "__main__":
    extract_pdf_text()
