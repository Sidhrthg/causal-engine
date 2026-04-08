"""RAG retrieval system for document-based validation."""

import os
import json
import pickle
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import requests
except ImportError:
    requests = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


class DocumentChunk:
    """Represents a chunk of a document."""

    def __init__(self, text: str, metadata: dict):
        self.text = text
        self.metadata = metadata
        self.chunk_id = hashlib.md5(text.encode()).hexdigest()[:16]

    def to_dict(self):
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
        }


class SimpleRAGRetriever:
    """
    Document retrieval system using keyword search and LLM re-ranking.

    For production, would use proper embeddings + vector DB (e.g., Chroma, Pinecone).
    This implementation uses keyword search + LLM relevance scoring.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        index_path: str = "data/documents/index.json",
        api_key: str = None,
    ):
        self.documents_dir = Path(documents_dir)
        self.index_path = Path(index_path)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key) if self.api_key and Anthropic else None

        self.chunks: List[DocumentChunk] = []
        self.index: Dict = {}
        self.embeddings: Optional[np.ndarray] = None

        # Load or create index
        if self.index_path.exists():
            self._load_index()

    def _get_embeddings(self, texts: List[str], model: str = "voyage-3") -> np.ndarray:
        """
        Get embeddings for texts using Voyage AI API.

        Alternative: Use sentence-transformers locally (no API needed)
        """
        voyage_key = os.getenv("VOYAGE_API_KEY")

        if voyage_key and requests is not None:
            return self._voyage_embeddings(texts, voyage_key)
        else:
            return self._local_embeddings(texts)

    def _voyage_embeddings(self, texts: List[str], api_key: str) -> np.ndarray:
        """Get embeddings from Voyage AI."""
        if requests is None:
            raise ImportError("requests required for Voyage AI. pip install requests")
        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"input": texts, "model": "voyage-3"},
        )

        if response.status_code != 200:
            raise Exception(f"Voyage API error: {response.text}")

        embeddings = [item["embedding"] for item in response.json()["data"]]
        return np.array(embeddings)

    def _local_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings locally using sentence-transformers.
        No API key needed, runs on your machine.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("⚠️  sentence-transformers not installed")
            print("Install with: pip install sentence-transformers")
            return np.zeros((len(texts), 384))

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, show_progress_bar=False)
        return embeddings

    def _build_embedding_index(self):
        """Build vector embeddings for all chunks."""
        print("🧮 Building embedding index...")

        texts = [chunk.text for chunk in self.chunks]

        self.embeddings = self._get_embeddings(texts)

        self.index["has_embeddings"] = True

        embeddings_path = self.index_path.parent / "embeddings.pkl"
        with open(embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

        print(f"✅ Generated {len(self.embeddings)} embeddings")

    def ingest_documents(self, force_reindex: bool = False, build_embeddings: bool = True):
        """
        Ingest all documents from documents directory.

        Args:
            force_reindex: Rebuild index even if exists
            build_embeddings: Generate vector embeddings for semantic search
        """
        if not force_reindex and self.index_path.exists():
            print("Index exists. Use force_reindex=True to rebuild.")
            embeddings_path = self.index_path.parent / "embeddings.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, "rb") as f:
                    self.embeddings = pickle.load(f)
            return

        print(f"📚 Ingesting documents from {self.documents_dir}...")

        self.chunks = []

        text_files = list(self.documents_dir.rglob("*.txt")) + list(
            self.documents_dir.rglob("*.md")
        )

        if len(text_files) == 0:
            print("⚠️  No documents found. Add .txt or .md files to data/documents/")
            return

        for file_path in text_files:
            print(f"  Processing: {file_path.name}")
            self._ingest_file(file_path)

        self._build_keyword_index()

        if build_embeddings:
            self._build_embedding_index()

        self._save_index()

        print(f"✅ Indexed {len(self.chunks)} chunks from {len(text_files)} documents\n")

    def _ingest_file(self, file_path: Path):
        """Ingest a single file and create chunks."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract metadata from path
        try:
            rel = file_path.relative_to(self.documents_dir)
        except ValueError:
            rel = file_path
        metadata = {
            "source_file": str(rel),
            "category": file_path.parent.name,
            "filename": file_path.name,
        }

        # Try to extract year from filename or content
        year_match = re.search(r"(19|20)\d{2}", file_path.name)
        if year_match:
            metadata["year"] = int(year_match.group())

        # Chunk the document (simple: by paragraph)
        chunks = self._chunk_text(content, chunk_size=500, overlap=50)

        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i

            chunk = DocumentChunk(chunk_text, chunk_metadata)
            self.chunks.append(chunk)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Chunk text into overlapping segments.

        Simple implementation - for production would use semantic chunking.
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para_length = len(para)

            if current_length + para_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append("\n\n".join(current_chunk))

                # Start new chunk with overlap
                if overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += para_length

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _build_keyword_index(self):
        """Build simple keyword index for fast retrieval."""
        self.index = {
            "chunks": [chunk.to_dict() for chunk in self.chunks],
            "keywords": {},
        }

        for i, chunk in enumerate(self.chunks):
            keywords = self._extract_keywords(chunk.text)
            for keyword in keywords:
                if keyword not in self.index["keywords"]:
                    self.index["keywords"][keyword] = []
                self.index["keywords"][keyword].append(i)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text (simple implementation)."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())

        stopwords = {
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
            "was", "one", "our", "out", "day", "get", "has", "him", "his", "how",
            "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did",
            "its", "let", "put", "say", "she", "too", "use",
        }

        keywords = [w for w in words if w not in stopwords]
        return list(set(keywords))

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict = None,
    ) -> List[Dict]:
        """
        Retrieve most relevant document chunks for query.

        Args:
            query: Search query
            top_k: Number of chunks to return
            filters: Optional filters (e.g., {'year': 2008})

        Returns:
            List of relevant chunks with metadata
        """
        if len(self.chunks) == 0:
            print("⚠️  No documents indexed. Run ingest_documents() first.")
            return []

        print(f"🔍 Retrieving documents for: '{query}'")

        candidate_indices = self._keyword_search(query)

        if filters:
            candidate_indices = [
                i for i in candidate_indices
                if self._matches_filters(self.chunks[i], filters)
            ]

        if self.client and len(candidate_indices) > 0:
            ranked_chunks = self._llm_rerank(query, candidate_indices, top_k)
        else:
            ranked_chunks = [self.chunks[i].to_dict() for i in candidate_indices[:top_k]]

        print(f"  → Retrieved {len(ranked_chunks)} relevant chunks\n")

        return ranked_chunks

    def retrieve_semantic(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict = None,
    ) -> List[Dict]:
        """
        Semantic retrieval using embeddings.
        Better than keyword search for conceptual queries.
        """
        if not hasattr(self, "embeddings") or self.embeddings is None:
            print("⚠️  No embeddings found. Using keyword search instead.")
            return self.retrieve(query, top_k, filters)

        print(f"🔍 Semantic search for: '{query}'")

        query_embedding = self._get_embeddings([query])[0]

        norms = np.linalg.norm(self.embeddings, axis=1)
        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            similarities = np.zeros(len(self.chunks))
        else:
            similarities = np.dot(self.embeddings, query_embedding) / (norms * q_norm)

        if filters:
            valid_indices = [
                i for i in range(len(self.chunks))
                if self._matches_filters(self.chunks[i], filters)
            ]
            mask = np.ones(len(similarities), dtype=float) * -1
            mask[valid_indices] = similarities[valid_indices]
            similarities = mask

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = [
            {**self.chunks[i].to_dict(), "similarity": float(similarities[i])}
            for i in top_indices
            if similarities[i] > 0
        ]

        if results:
            avg_sim = float(np.mean([r["similarity"] for r in results]))
            print(f"  → Retrieved {len(results)} chunks (avg similarity: {avg_sim:.3f})\n")
        else:
            print(f"  → No matching chunks (filters or empty index)\n")

        return results

    def _keyword_search(self, query: str) -> List[int]:
        """Fast keyword-based candidate retrieval."""
        query_keywords = self._extract_keywords(query)

        scores = {}
        for keyword in query_keywords:
            if keyword in self.index["keywords"]:
                for chunk_idx in self.index["keywords"][keyword]:
                    scores[chunk_idx] = scores.get(chunk_idx, 0) + 1

        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def _matches_filters(self, chunk: DocumentChunk, filters: Dict) -> bool:
        """Check if chunk matches filter criteria."""
        for key, value in filters.items():
            if key not in chunk.metadata:
                return False
            if chunk.metadata[key] != value:
                return False
        return True

    def _llm_rerank(self, query: str, candidate_indices: List[int], top_k: int) -> List[Dict]:
        """Use LLM to re-rank candidates by relevance."""
        candidates = [self.chunks[i] for i in candidate_indices[:20]]

        candidates_text = "\n\n---\n\n".join(
            [f"[CHUNK {i}]\n{chunk.text[:300]}..." for i, chunk in enumerate(candidates)]
        )

        prompt = f"""Given this query: "{query}"

Rate the relevance of each chunk below on a scale of 0-10.
Return ONLY a JSON array of scores in order: [score_0, score_1, ...]

Chunks:
{candidates_text}

JSON scores:"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )

            scores_text = response.content[0].text.strip()
            json_match = re.search(r"\[[\d,\s]+\]", scores_text)
            if json_match:
                scores = json.loads(json_match.group())
            else:
                scores = [5] * len(candidates)
        except Exception as e:
            print(f"  ⚠️  LLM reranking failed: {e}")
            scores = [5] * len(candidates)

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        return [chunk.to_dict() for chunk, score in ranked[:top_k]]

    def _save_index(self):
        """Save index to disk."""
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def _load_index(self):
        """Load index from disk."""
        with open(self.index_path, "r") as f:
            raw = f.read()
        if not raw.strip():
            self.index = {"chunks": [], "keywords": {}}
            self.chunks = []
            self.embeddings = None
            return
        self.index = json.loads(raw)
        self.chunks = [
            DocumentChunk(chunk_data["text"], chunk_data["metadata"])
            for chunk_data in self.index.get("chunks", [])
        ]
        embeddings_path = self.index_path.parent / "embeddings.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, "rb") as f:
                self.embeddings = pickle.load(f)
        else:
            self.embeddings = None


def demo():
    """Demo semantic search."""

    retriever = SimpleRAGRetriever()

    # Ingest with embeddings
    retriever.ingest_documents(force_reindex=True, build_embeddings=True)

    if len(retriever.chunks) > 0:
        print("\n" + "=" * 70)
        print("KEYWORD SEARCH:")
        print("=" * 70)
        keyword_results = retriever.retrieve(
            query="price volatility supply disruptions",
            top_k=3
        )
        for i, chunk in enumerate(keyword_results):
            print(f"[{i+1}] {chunk['metadata']['source_file']}")
            print(f"    {chunk['text'][:100]}...\n")

        print("\n" + "=" * 70)
        print("SEMANTIC SEARCH (with embeddings):")
        print("=" * 70)
        semantic_results = retriever.retrieve_semantic(
            query="What causes prices to spike during shortages?",
            top_k=3
        )
        for i, chunk in enumerate(semantic_results):
            print(f"[{i+1}] {chunk['metadata']['source_file']} (similarity: {chunk['similarity']:.3f})")
            print(f"    {chunk['text'][:100]}...\n")


if __name__ == "__main__":
    demo()
