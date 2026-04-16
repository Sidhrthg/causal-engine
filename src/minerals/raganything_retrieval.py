"""
RAG-Anything multimodal retriever and KG bridge.

Wraps RAGAnything (LightRAG + multimodal processors) with:
1. ``retrieve(query, top_k)``  — same interface as HippoRAGRetriever / SimpleRAGRetriever
2. ``index(doc_paths)``        — ingest PDFs, images, Word docs into the multimodal KG
3. ``get_networkx_graph()``    — return the raw NetworkX graph (.graphml)
4. ``bridge_to_causal_kg(kg)`` — import LightRAG entities/relations into CausalKnowledgeGraph

RAGAnything builds a **unified multimodal knowledge graph** that stores text entities,
image descriptions, table summaries, and equation descriptions as first-class nodes,
linked by LLM-extracted relationships.  The underlying graph is a standard NetworkX
MultiDiGraph persisted as ``<working_dir>/graph_chunk_entity_relation.graphml``.

Usage
-----
    from src.minerals.raganything_retrieval import RAGAnythingRetriever

    ret = RAGAnythingRetriever(working_dir="data/raganything_index")
    ret.index(["data/documents/report.pdf", "data/documents/chart.png"])

    chunks = ret.retrieve("graphite supply disruption 2023", top_k=5)

    # Bridge the multimodal KG into the causal KG
    from src.minerals.knowledge_graph import build_critical_minerals_kg
    kg = build_critical_minerals_kg()
    n = ret.bridge_to_causal_kg(kg)
    dag = kg.to_causal_dag()
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    import networkx as nx
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    RAGANYTHING_AVAILABLE = True
except ImportError:
    RAGANYTHING_AVAILABLE = False

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_WORKING_DIR = "data/documents/raganything_index"


# ---------------------------------------------------------------------------
# Async helper
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run *coro* whether or not an event loop is already running."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Inside Jupyter / FastAPI — schedule as a task (caller must await)
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Predicate mapping (LightRAG keyword → RelationType)
# ---------------------------------------------------------------------------

def _map_keywords_to_relation_type(keywords: str, description: str = ""):
    """
    Map LightRAG edge keywords/description to a ``RelationType``.
    Returns None if no match — those edges are skipped in the causal bridge.
    """
    # Import here to avoid circular imports
    from src.minerals.knowledge_graph import RelationType

    text = f"{keywords} {description}".lower()

    _KEYWORD_MAP = [
        (RelationType.CAUSES,       ["causes", "leads to", "results in", "drives", "triggers", "impacts"]),
        (RelationType.PRODUCES,     ["produces", "mines", "extracts", "manufactures", "outputs"]),
        (RelationType.EXPORTS_TO,   ["exports", "ships to", "sells to", "sends to"]),
        (RelationType.IMPORTS_FROM, ["imports", "buys from", "receives from", "sources from"]),
        (RelationType.SUPPLIES,     ["supplies", "provides", "delivers", "furnishes"]),
        (RelationType.PROCESSES,    ["processes", "refines", "smelts", "converts", "transforms"]),
        (RelationType.DEPENDS_ON,   ["depends on", "relies on", "requires", "needs"]),
        (RelationType.SUBSTITUTES,  ["substitutes", "replaces", "alternative to", "competes with"]),
        (RelationType.DISRUPTS,     ["disrupts", "interrupts", "halts", "suspends", "stops"]),
        (RelationType.REGULATES,    ["regulates", "restricts", "bans", "controls", "limits", "sanctions"]),
        (RelationType.MITIGATES,    ["mitigates", "reduces", "alleviates", "offsets"]),
        (RelationType.LOCATED_IN,   ["located in", "based in", "situated in", "found in"]),
        (RelationType.PART_OF,      ["part of", "component of", "belongs to", "member of"]),
        (RelationType.CONSUMES,     ["consumes", "uses", "utilises", "absorbs"]),
        (RelationType.ENABLES,      ["enables", "facilitates", "supports", "allows"]),
    ]
    for rel_type, keywords_list in _KEYWORD_MAP:
        if any(kw in text for kw in keywords_list):
            return rel_type
    return None


def _classify_entity_type(entity_type_str: str, name: str):
    """Map LightRAG entity_type string to our EntityType enum."""
    from src.minerals.knowledge_graph import EntityType

    s = entity_type_str.lower()
    n = name.lower()

    if "image" in s or "figure" in s or "chart" in s:
        return EntityType.ECONOMIC_INDICATOR  # visual data → indicator proxy
    if "table" in s:
        return EntityType.ECONOMIC_INDICATOR
    if "country" in s or "nation" in s or any(c in n for c in ["china", "australia", "drc", "chile", "india", "usa"]):
        return EntityType.COUNTRY
    if "company" in s or "corp" in n or "ltd" in n or "inc" in n or "mine" in n:
        return EntityType.COMPANY
    if "policy" in s or "regulation" in s or "tariff" in n or "ban" in n or "sanction" in n:
        return EntityType.POLICY
    if "commodity" in s or any(m in n for m in ["graphite", "lithium", "cobalt", "copper", "nickel", "antimony"]):
        return EntityType.COMMODITY
    if "price" in n or "demand" in n or "supply" in n or "index" in n:
        return EntityType.ECONOMIC_INDICATOR
    if "technology" in s or "tech" in n or "process" in n:
        return EntityType.TECHNOLOGY
    if "event" in s or "disruption" in n or "shock" in n:
        return EntityType.EVENT
    return EntityType.COMMODITY  # safe default


# ---------------------------------------------------------------------------
# Main retriever class
# ---------------------------------------------------------------------------

class RAGAnythingRetriever:
    """
    Multimodal RAG retriever backed by RAGAnything / LightRAG.

    Accepts the same ``retrieve(query, top_k)`` interface as HippoRAGRetriever
    and SimpleRAGRetriever, so it plugs into RAGPipeline and KGExtractor
    without code changes.

    Additionally exposes:
    - ``index(doc_paths)`` for multimodal document ingestion
    - ``get_networkx_graph()`` for the raw KG
    - ``bridge_to_causal_kg(kg)`` to import LightRAG entities into CausalKnowledgeGraph

    Args:
        working_dir: Where LightRAG stores its KG and vector indexes.
        llm_model_name: OpenAI LLM model for text/entity extraction.
        embedding_model_name: OpenAI embedding model.
        vision_model_name: OpenAI vision model for image description (``None`` disables image processing).
        openai_api_key: Falls back to ``OPENAI_API_KEY`` env var.
        retrieve_mode: LightRAG query mode for ``retrieve()`` — ``"naive"`` (vector only)
            or ``"mix"`` (KG + vector, richer but slower).
    """

    def __init__(
        self,
        working_dir: str = DEFAULT_WORKING_DIR,
        llm_model_name: str = "gpt-4o-mini",
        embedding_model_name: str = "text-embedding-3-large",
        vision_model_name: Optional[str] = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        retrieve_mode: str = "mix",
    ):
        if not RAGANYTHING_AVAILABLE:
            raise RuntimeError(
                "raganything not installed. Run: pip install raganything"
            )

        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._retrieve_mode = retrieve_mode

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self._llm_model_name = llm_model_name
        self._embedding_model_name = embedding_model_name
        self._vision_model_name = vision_model_name

        # Build LLM + embedding functions for LightRAG
        llm_func = self._make_llm_func(llm_model_name)
        embed_func = self._make_embed_func(embedding_model_name)
        vision_func = self._make_llm_func(vision_model_name) if vision_model_name else None

        config = RAGAnythingConfig(
            working_dir=str(self.working_dir),
            enable_image_processing=vision_func is not None,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        self._rag = RAGAnything(
            llm_model_func=llm_func,
            vision_model_func=vision_func,
            embedding_func=embed_func,
            config=config,
        )

        logger.info(
            f"RAGAnythingRetriever ready — working_dir={self.working_dir}, "
            f"llm={llm_model_name}, embed={embedding_model_name}, "
            f"vision={'enabled' if vision_func else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # LightRAG function builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_llm_func(model_name: str):
        """Return an async LLM callable for LightRAG using openai_complete_if_cache."""
        from functools import partial
        return partial(openai_complete_if_cache, model_name)

    @staticmethod
    def _make_embed_func(model_name: str) -> "EmbeddingFunc":
        """Return an EmbeddingFunc for LightRAG."""
        from functools import partial

        # Embedding dims by model
        _DIMS = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002":  1536,
        }
        dim = _DIMS.get(model_name, 1536)

        return EmbeddingFunc(
            embedding_dim=dim,
            max_token_size=8192,
            func=partial(openai_embed, model=model_name),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index(self, doc_paths: List[str], output_dir: Optional[str] = None) -> str:
        """
        Ingest a list of document paths into the multimodal KG.

        Supports PDFs, images (PNG/JPG/…), Word, PowerPoint, Excel, plain text.
        Each document is parsed with MinerU, then text/image/table/equation content
        is extracted and inserted into the LightRAG knowledge graph.

        Args:
            doc_paths: List of file paths to ingest.
            output_dir: Directory for intermediate MinerU parse output.

        Returns:
            Status string.
        """
        out_dir = output_dir or str(self.working_dir / "parser_output")
        ok, failed = 0, []

        for path in doc_paths:
            p = Path(path)
            if not p.exists():
                logger.warning(f"File not found, skipping: {path}")
                failed.append(path)
                continue
            try:
                logger.info(f"Indexing: {p.name}")
                _run_async(
                    self._rag.process_document_complete(
                        file_path=str(p),
                        output_dir=out_dir,
                    )
                )
                ok += 1
                logger.info(f"Indexed: {p.name}")
            except Exception as exc:
                logger.error(f"Failed to index {p.name}: {exc}")
                failed.append(path)

        status = f"RAGAnything indexed {ok}/{len(doc_paths)} documents."
        if failed:
            status += f" Failed: {[Path(f).name for f in failed]}"
        logger.info(status)
        return status

    def index_directory(self, documents_dir: str = "data/documents") -> str:
        """Ingest all supported documents under *documents_dir*."""
        config = self._rag.config
        exts = set(config.supported_file_extensions)
        doc_dir = Path(documents_dir)
        paths = [
            str(f) for f in sorted(doc_dir.rglob("*"))
            if f.is_file() and f.suffix.lower() in exts
            and "raganything_index" not in str(f)
        ]
        if not paths:
            return f"No supported documents found in {documents_dir}"
        return self.index(paths)

    # ------------------------------------------------------------------
    # Retrieval  (same interface as HippoRAGRetriever / SimpleRAGRetriever)
    # ------------------------------------------------------------------

    @property
    def chunks(self) -> List[Any]:
        """Compatibility shim — returns empty list (LightRAG manages its own chunks)."""
        return []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for *query* using LightRAG's KG + vector retrieval.

        Args:
            query: Natural language query.
            top_k: Approximate number of text passages to return.
            filters: Ignored (API parity with other retrievers).

        Returns:
            List of ``{"text": str, "metadata": dict, "score": float}``.
        """
        try:
            raw_context: str = _run_async(
                self._rag.aquery(
                    query,
                    mode=self._retrieve_mode,
                    only_need_context=True,
                    chunk_top_k=top_k,
                )
            )
        except Exception as exc:
            logger.error(f"RAGAnything retrieve failed: {exc}")
            return [{"text": f"[RAGAnything error: {exc}]", "metadata": {"source_file": "?"}}]

        return self._parse_context_to_chunks(raw_context, top_k)

    def _parse_context_to_chunks(self, context: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Split LightRAG context string into individual chunk dicts.

        LightRAG ``only_need_context`` returns sections:
        ``-----Entities-----``, ``-----Relationships-----``, ``-----Sources-----``

        We extract the Sources section (raw text passages) and also include
        the Entities/Relationships block as a structured summary chunk.
        """
        chunks: List[Dict[str, Any]] = []

        # Split on LightRAG section headers
        sections: Dict[str, str] = {}
        current_section = "preamble"
        buf: List[str] = []
        for line in context.splitlines():
            header_match = re.match(r"^-----(.+?)-----", line.strip())
            if header_match:
                sections[current_section] = "\n".join(buf).strip()
                current_section = header_match.group(1).strip().lower()
                buf = []
            else:
                buf.append(line)
        sections[current_section] = "\n".join(buf).strip()

        # Add Sources as individual chunks
        sources_text = sections.get("sources", sections.get("text", ""))
        if sources_text:
            passages = re.split(r"\n{2,}", sources_text)
            for i, passage in enumerate(passages[:top_k]):
                passage = passage.strip()
                if len(passage) < 30:
                    continue
                chunks.append({
                    "text": passage,
                    "metadata": {"source_file": f"raganything_source_{i}", "backend": "raganything"},
                    "score": 1.0 - (i * 0.05),
                })

        # Add KG summary (Entities + Relationships) as one enriched chunk
        kg_parts = []
        for section_name in ("entities", "relationships"):
            section_text = sections.get(section_name, "").strip()
            if section_text:
                kg_parts.append(section_text)
        if kg_parts and len(chunks) < top_k:
            chunks.append({
                "text": "\n\n".join(kg_parts),
                "metadata": {"source_file": "raganything_kg_context", "backend": "raganything"},
                "score": 0.9,
            })

        # Fallback: return the whole context as one chunk
        if not chunks and context.strip():
            chunks.append({
                "text": context.strip(),
                "metadata": {"source_file": "raganything_context", "backend": "raganything"},
                "score": 1.0,
            })

        return chunks[:top_k]

    # ------------------------------------------------------------------
    # KG access
    # ------------------------------------------------------------------

    def _graphml_path(self) -> Path:
        return self.working_dir / "graph_chunk_entity_relation.graphml"

    def get_networkx_graph(self) -> Optional["nx.Graph"]:
        """
        Return the raw LightRAG NetworkX graph loaded from the ``.graphml`` file.

        Nodes have attributes like ``entity_type``, ``description``, ``source_id``.
        Edges have attributes like ``keywords``, ``description``, ``weight``.

        Returns ``None`` if the graph file does not exist yet.
        """
        gml = self._graphml_path()
        if not gml.exists():
            logger.warning(f"No graph file yet at {gml} — run index() first.")
            return None
        G = nx.read_graphml(str(gml))
        logger.info(f"Loaded LightRAG graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G

    # ------------------------------------------------------------------
    # Bridge: LightRAG KG → CausalKnowledgeGraph
    # ------------------------------------------------------------------

    def bridge_to_causal_kg(self, kg, min_edge_weight: float = 0.0) -> int:
        """
        Import the LightRAG multimodal KG into an existing ``CausalKnowledgeGraph``.

        Reads the ``.graphml`` file and:
        - Creates ``Entity`` objects for every LightRAG node (text, image, table, …)
        - Maps LightRAG edge keywords/descriptions to ``RelationType`` values
        - Merges everything into *kg* (existing edges get confidence reinforced)

        Args:
            kg: ``CausalKnowledgeGraph`` to enrich in-place.
            min_edge_weight: Skip LightRAG edges below this weight (0 = keep all).

        Returns:
            Number of relationships merged.
        """
        from src.minerals.knowledge_graph import Entity, Relationship

        G = self.get_networkx_graph()
        if G is None:
            return 0

        new_kg_entities: Dict[str, "Entity"] = {}
        new_kg_rels: List["Relationship"] = []

        # Nodes → Entities
        for node_id, node_data in G.nodes(data=True):
            eid = node_id.strip().upper()
            if not eid:
                continue
            etype = _classify_entity_type(
                node_data.get("entity_type", ""),
                node_id,
            )
            new_kg_entities[eid] = Entity(
                id=eid,
                entity_type=etype,
                properties={
                    "description": node_data.get("description", ""),
                    "source_id":   node_data.get("source_id", ""),
                    "auto_extracted": True,
                    "origin": "raganything",
                },
            )

        # Edges → Relationships
        for src, dst, edge_data in G.edges(data=True):
            weight = float(edge_data.get("weight", 1.0))
            if weight < min_edge_weight:
                continue

            keywords    = edge_data.get("keywords", "") or ""
            description = edge_data.get("description", "") or ""
            rel_type    = _map_keywords_to_relation_type(keywords, description)
            if rel_type is None:
                continue

            src_id = src.strip().upper()
            dst_id = dst.strip().upper()
            if not src_id or not dst_id:
                continue

            conf = min(weight / 10.0, 1.0)  # LightRAG weights are typically 1-10+
            new_kg_rels.append(
                Relationship(
                    source_id=src_id,
                    target_id=dst_id,
                    relation_type=rel_type,
                    properties={
                        "confidence_score": conf,
                        "confidence": "HIGH" if conf >= 0.7 else ("MEDIUM" if conf >= 0.4 else "LOW"),
                        "evidence": description[:500],
                        "keywords": keywords,
                        "weight": weight,
                        "provenance": "raganything",
                        "auto_extracted": True,
                        "origin": "raganything",
                    },
                )
            )

        # Merge into caller's KG
        from src.minerals.knowledge_graph import CausalKnowledgeGraph
        bridge_kg = CausalKnowledgeGraph()
        for entity in new_kg_entities.values():
            bridge_kg.add_entity(entity)
        for rel in new_kg_rels:
            # Ensure both endpoints exist
            for eid in (rel.source_id, rel.target_id):
                if eid not in new_kg_entities:
                    from src.minerals.knowledge_graph import EntityType
                    bridge_kg.add_entity(Entity(id=eid, entity_type=EntityType.COMMODITY))
            bridge_kg.add_relationship(rel)

        before = kg.num_relationships
        kg.merge(bridge_kg)
        added = kg.num_relationships - before

        logger.info(
            f"bridge_to_causal_kg: {len(new_kg_entities)} entities, "
            f"{len(new_kg_rels)} candidate rels → +{added} new, "
            f"{len(new_kg_rels) - added} reinforced"
        )
        return len(new_kg_rels)


# ---------------------------------------------------------------------------
# Availability check + factory
# ---------------------------------------------------------------------------

def raganything_available() -> bool:
    return RAGANYTHING_AVAILABLE


def get_raganything_retriever(
    working_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    retrieve_mode: str = "mix",
) -> Optional[RAGAnythingRetriever]:
    """
    Return a ``RAGAnythingRetriever`` if raganything is installed and the index exists.
    Returns ``None`` otherwise (caller falls back to another backend).
    """
    if not RAGANYTHING_AVAILABLE:
        return None
    wd = Path(working_dir or DEFAULT_WORKING_DIR)
    graphml = wd / "graph_chunk_entity_relation.graphml"
    if not graphml.exists():
        return None
    try:
        return RAGAnythingRetriever(
            working_dir=str(wd),
            openai_api_key=api_key,
            retrieve_mode=retrieve_mode,
        )
    except Exception as exc:
        logger.debug(f"RAGAnythingRetriever init failed: {exc}")
        return None
