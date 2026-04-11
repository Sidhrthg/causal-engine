"""
Industrial-scale RAG with ChromaDB vector database.
Scales to 1000+ documents.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class IndustrialRAG:
    """
    Production-grade RAG system using ChromaDB.

    Features:
    - Persistent vector storage
    - Batch processing
    - Metadata filtering
    - Semantic + keyword hybrid search
    """

    def __init__(
        self,
        collection_name: str = "minerals_corpus",
        persist_directory: str = "data/chroma_db",
        embedding_model: str = "all-mpnet-base-v2",
    ):
        """
        Initialize industrial RAG system.

        Args:
            collection_name: Name of ChromaDB collection
            persist_directory: Where to store vector database
            embedding_model: Sentence-transformers model name
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install with: pip install chromadb")
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name

        try:
            settings = Settings(anonymized_telemetry=False)
        except Exception:
            settings = None
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=settings,
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"📥 Loading embedding model: {embedding_model}")
        self.encoder = SentenceTransformer(embedding_model)
        print(f"✅ Model loaded (dimension: {self.encoder.get_sentence_embedding_dimension()})")

        self.chunk_size = 500
        self.chunk_overlap = 50

    def index_corpus(
        self,
        doc_directory: str = "data/documents",
        batch_size: int = 100,
        force_reindex: bool = False,
    ) -> int:
        """
        Index entire document corpus.

        Args:
            doc_directory: Root directory containing documents
            batch_size: Documents to process at once
            force_reindex: If True, clear existing index

        Returns:
            Total number of chunks indexed.
        """
        if force_reindex:
            print("🔄 Clearing existing index...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        doc_path = Path(doc_directory)
        all_docs = list(doc_path.rglob("*.txt"))

        print(f"\n📚 Indexing {len(all_docs)} documents...")
        print(f"📁 Source: {doc_directory}")
        print(f"💾 Database: {self.persist_directory}")
        print(f"🔢 Batch size: {batch_size}")

        total_chunks = 0

        for i in tqdm(range(0, len(all_docs), batch_size), desc="Batches"):
            batch_docs = all_docs[i : i + batch_size]

            chunks: List[str] = []
            metadatas: List[Dict] = []
            ids: List[str] = []

            for fp in batch_docs:
                try:
                    text = fp.read_text(encoding="utf-8")
                except Exception as e:
                    print(f"⚠️  Failed to read {fp.name}: {e}")
                    continue

                doc_chunks = self._chunk_text(text)
                metadata_base = self._extract_metadata(fp)

                for j, chunk in enumerate(doc_chunks):
                    chunks.append(chunk)
                    metadatas.append(
                        {
                            **metadata_base,
                            "chunk_id": j,
                            "total_chunks": len(doc_chunks),
                        }
                    )
                    ids.append(f"{fp.stem}_chunk_{j}")

            if chunks:
                embeddings = self.encoder.encode(
                    chunks,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )

                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    embeddings=embeddings.tolist(),
                    ids=ids,
                )

                total_chunks += len(chunks)

        n_docs = len(all_docs)
        print(f"\n✅ Indexing complete!")
        print(f"📊 Statistics:")
        print(f"   - Documents: {n_docs}")
        print(f"   - Total chunks: {total_chunks}")
        print(f"   - Average chunks per doc: {total_chunks / n_docs:.1f}" if n_docs else "   - N/A")
        print(f"   - Database size: {self._get_db_size()}")

        return total_chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant documents.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters, e.g. {"mineral": "graphite"},
                {"year": {"$gte": 2020}}, {"source_type": {"$in": ["usgs", "iea"]}}

        Returns:
            List of retrieved chunks with metadata and similarity.
        """
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)

        kwargs = {
            "query_embeddings": [query_embedding.tolist()],
            "n_results": top_k,
        }
        if filters is not None:
            kwargs["where"] = filters

        results = self.collection.query(**kwargs)

        retrieved = []
        n = len(results["ids"][0])
        distances = results.get("distances")
        # Chroma returns distances (lower = more similar for cosine)
        for i in range(n):
            sim = 1.0 - distances[0][i] if distances else 0.0
            retrieved.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": sim,
                }
            )

        return retrieved

    def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        semantic_weight: float = 0.7,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Hybrid semantic + keyword search.
        """
        semantic_results = self.retrieve(query, top_k=top_k * 2, filters=filters)
        keywords = set(query.lower().split())

        for result in semantic_results:
            text_lower = result["text"].lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            keyword_score = keyword_matches / len(keywords) if keywords else 0.0

            result["semantic_score"] = result["similarity"]
            result["keyword_score"] = keyword_score
            result["hybrid_score"] = (
                semantic_weight * result["similarity"]
                + (1 - semantic_weight) * keyword_score
            )

        semantic_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return semantic_results[:top_k]

    def search_by_mineral(
        self, mineral: str, query: str, top_k: int = 10
    ) -> List[Dict]:
        """Search within specific mineral."""
        return self.retrieve(query=query, top_k=top_k, filters={"mineral": mineral})

    def search_by_year_range(
        self,
        query: str,
        start_year: int,
        end_year: int,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search within year range."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            filters={"year": {"$gte": start_year, "$lte": end_year}},
        )

    def get_statistics(self) -> Dict:
        """Get corpus statistics."""
        count = self.collection.count()
        sample = self.collection.get(limit=min(1000, count))

        minerals: set = set()
        years: set = set()
        sources: set = set()

        for metadata in sample.get("metadatas", []):
            if "mineral" in metadata:
                minerals.add(metadata["mineral"])
            if "year" in metadata:
                years.add(metadata["year"])
            if "source_type" in metadata:
                sources.add(metadata["source_type"])

        return {
            "total_chunks": count,
            "unique_minerals": len(minerals),
            "year_range": (
                f"{min(years)}-{max(years)}" if years else "N/A"
            ),
            "source_types": list(sources),
            "database_size": self._get_db_size(),
        }

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if len(chunk.split()) > 50:
                chunks.append(chunk)

        return chunks

    def _extract_metadata(self, doc_path: Path) -> Dict:
        """Extract metadata from document path and filename."""
        metadata: Dict = {
            "source": doc_path.name,
            "filepath": str(doc_path),
        }

        path_str = str(doc_path).lower()
        # Ordered: check multi-word names before single-word to avoid partial matches
        _MINERALS = [
            "rare-earths", "rare_earths",
            "graphite", "lithium", "cobalt", "copper", "nickel",
            "antimony", "beryllium", "cesium", "gallium", "germanium",
            "indium", "niobium", "platinum", "tantalum", "tellurium",
            "titanium", "tungsten", "vanadium", "yttrium",
        ]
        for part in _MINERALS:
            if part in path_str:
                metadata["mineral"] = part.replace("_", "-")
                break

        year_match = re.search(r"(19|20)\d{2}", doc_path.name)
        if year_match:
            metadata["year"] = int(year_match.group())

        if "usgs" in path_str.lower():
            metadata["source_type"] = "usgs"
        elif "iea" in path_str.lower():
            metadata["source_type"] = "iea"
        elif "academic" in path_str.lower():
            metadata["source_type"] = "academic"
        else:
            metadata["source_type"] = "other"

        return metadata

    def _get_db_size(self) -> str:
        """Get database size on disk."""
        if self.persist_directory.exists():
            total_size = sum(
                f.stat().st_size for f in self.persist_directory.rglob("*") if f.is_file()
            )
            size_mb = total_size / (1024 * 1024)
            return f"{size_mb:.1f} MB"
        return "Unknown"


def main() -> None:
    """Test industrial RAG system."""
    print("=" * 70)
    print("INDUSTRIAL RAG SYSTEM TEST")
    print("=" * 70)

    rag = IndustrialRAG()

    total_chunks = rag.index_corpus(
        doc_directory="data/documents",
        batch_size=50,
        force_reindex=False,
    )

    stats = rag.get_statistics()
    print(f"\n📊 Corpus Statistics:")
    print(json.dumps(stats, indent=2))

    print(f"\n🔍 Test Query: 'graphite supply disruption 2008'")
    results = rag.retrieve("graphite supply disruption 2008", top_k=5)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['metadata']['source']} (similarity: {result['similarity']:.3f})")
        print(f"   {result['text'][:150]}...")

    print(f"\n🔍 Hybrid Search: 'price volatility export restrictions'")
    hybrid_results = rag.hybrid_search("price volatility export restrictions", top_k=5)

    for i, result in enumerate(hybrid_results, 1):
        print(f"\n{i}. {result['metadata']['source']}")
        print(
            f"   Hybrid: {result['hybrid_score']:.3f} "
            f"(semantic: {result['semantic_score']:.3f}, keyword: {result['keyword_score']:.3f})"
        )
        print(f"   {result['text'][:150]}...")


if __name__ == "__main__":
    main()
