#!/usr/bin/env python3
"""
Ingest documents (PDF/text) and build a local knowledge index.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Try to import pypdf, but fail gracefully if not available
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    logger.warning("pypdf not available, PDF parsing will be disabled")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF file."""
    if not HAS_PYPDF:
        raise ImportError("pypdf not installed. Install with: pip install pypdf")
    
    text_parts = []
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            text_parts.append(page.extract_text())
        return "\n".join(text_parts)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""


def load_text_file(text_path: Path) -> str:
    """Load text from a text file."""
    with open(text_path, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end >= len(text):
            break
        
        start = end - overlap
    
    return chunks


def build_tfidf_index(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build TF-IDF index from text chunks.
    
    Args:
        chunks: List of dicts with 'text', 'source', 'chunk_id' keys
        
    Returns:
        Dict with 'vectorizer' (fitted) and 'chunks' (list)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    texts = [chunk["text"] for chunk in chunks]
    
    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # If no features, use simpler vectorizer
        logger.warning("TF-IDF fitting failed, using simpler vectorizer")
        vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 1),
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Convert to dense for serialization (small datasets only)
    # For larger datasets, you'd want to store sparse matrices
    tfidf_dense = tfidf_matrix.toarray()
    
    index = {
        "vectorizer": {
            "vocabulary": vectorizer.vocabulary_,
            "idf": vectorizer.idf_.tolist(),
        },
        "chunks": chunks,
        "tfidf_matrix": tfidf_dense.tolist(),
    }
    
    return index


def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build knowledge index")
    parser.add_argument("--path", type=str, required=True,
                       help="Path to directory containing PDF/text files")
    parser.add_argument("--out", type=str, default="artifacts/knowledge/index.jsonl",
                       help="Output path for index (JSONL format)")
    parser.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200,
                       help="Overlap between chunks")
    
    args = parser.parse_args()
    
    docs_dir = Path(args.path)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {args.path}")
    
    # Find all documents
    pdf_files = list(docs_dir.glob("*.pdf"))
    txt_files = list(docs_dir.glob("*.txt"))
    md_files = list(docs_dir.glob("*.md"))
    
    logger.info(f"Found {len(pdf_files)} PDFs, {len(txt_files)} TXT files, {len(md_files)} MD files")
    
    # Extract text from documents
    chunks = []
    
    for pdf_path in pdf_files:
        logger.info(f"Processing PDF: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            doc_chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
            for i, chunk_text_content in enumerate(doc_chunks):
                chunk_id = f"{pdf_path.stem}_{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": str(pdf_path),
                    "text": chunk_text_content,
                    "type": "pdf",
                })
    
    for txt_path in txt_files + md_files:
        logger.info(f"Processing text file: {txt_path.name}")
        text = load_text_file(txt_path)
        if text:
            doc_chunks = chunk_text(text, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
            for i, chunk_text_content in enumerate(doc_chunks):
                chunk_id = f"{txt_path.stem}_{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "source": str(txt_path),
                    "text": chunk_text_content,
                    "type": txt_path.suffix[1:],
                })
    
    if not chunks:
        logger.warning("No text chunks extracted from documents")
        return
    
    logger.info(f"Extracted {len(chunks)} text chunks")
    
    # Build TF-IDF index
    logger.info("Building TF-IDF index")
    index = build_tfidf_index(chunks)
    
    # Save to JSONL (one JSON object per line for streaming)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # For simplicity, save as single JSON file (can be split into JSONL if needed)
    with open(out_path, "w") as f:
        json.dump(index, f, indent=2)
    
    logger.info(f"Saved index to {out_path}")
    logger.info(f"Index contains {len(chunks)} chunks with {len(index['vectorizer']['vocabulary'])} vocabulary terms")


if __name__ == "__main__":
    main()


