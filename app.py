"""
Critical Minerals Causal Engine - Full inference engine UI.

Integrates: LLM query, RAG document search, causal DAG/identifiability,
scenario run, RAG validation, synthetic control, and POMDP (sensor maintenance).
"""

# Use non-interactive backend so matplotlib works in Gradio/server (no display)
import matplotlib
matplotlib.use("Agg")

import gradio as gr
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Shared RAGPipeline singleton — persists episodic memory within a session.
# Memory is loaded from / saved to data/memory/ across restarts automatically.
# ---------------------------------------------------------------------------
_MEMORY_DIR = PROJECT_ROOT / "data" / "memory"
_pipeline = None  # lazy-init on first use


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.minerals.rag_pipeline import RAGPipeline
        _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        _pipeline = RAGPipeline(backend="simple", memory_dir=str(_MEMORY_DIR))
    return _pipeline


# ----- Subprocess helpers -----

def _run(cmd: list, timeout: int = 120, capture_stdout_stderr: bool = True) -> tuple[str, str, int]:
    result = subprocess.run(
        cmd,
        capture_output=capture_stdout_stderr,
        text=True,
        timeout=timeout,
        cwd=str(PROJECT_ROOT),
    )
    return (result.stdout or "", result.stderr or "", result.returncode)


# ----- Tab: Query Model -----

def query_model(natural_language_query: str) -> str:
    try:
        out, err, code = _run(
            [sys.executable, "-m", "scripts.llm_query", natural_language_query],
            timeout=60,
        )
        if code != 0:
            return f"❌ Error:\n\n{err}\n\nOutput:\n{out}"
        return f"```\n{out}\n```"
    except subprocess.TimeoutExpired:
        return "❌ Query timed out (>60s). Try a simpler question."
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ----- Tab: RAG - Search Documents -----

DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
UPLOAD_SUBDIR = DOCUMENTS_DIR / "uploaded"  # Your uploaded .txt/.md go here (any scenario, incl. non-mineral)


def save_uploaded_documents(files: list | None) -> str:
    """Save uploaded .txt/.md files to data/documents/uploaded/. Returns status message."""
    if not files:
        return "No files selected. Choose one or more .txt or .md files."
    UPLOAD_SUBDIR.mkdir(parents=True, exist_ok=True)
    allowed = {".txt", ".md"}
    saved = []
    import shutil
    for f in files:
        if f is None:
            continue
        path = (f.get("name") if isinstance(f, dict) else f) or f
        path = Path(str(path))
        if not path.suffix or path.suffix.lower() not in allowed:
            continue
        dest = UPLOAD_SUBDIR / path.name
        try:
            shutil.copy2(str(path), str(dest))
            saved.append(path.name)
        except Exception as e:
            return f"❌ Error saving {path.name}: {e}"
    if not saved:
        return "No .txt or .md files to save. Only .txt and .md are indexed."
    return f"✅ Saved {len(saved)} file(s) to `data/documents/uploaded/`: {', '.join(saved)}\n\nClick **Rebuild search index** to include them in search."


def reindex_rag() -> str:
    """Rebuild the RAG document index after adding/uploading files."""
    script = PROJECT_ROOT / "scripts" / "index_rag_documents.py"
    if not script.exists():
        return f"❌ Index script not found: {script}"
    try:
        out, err, code = _run(
            [sys.executable, str(script)],
            timeout=120,
        )
        if code != 0:
            return f"❌ Index failed:\n\n{err}\n{out}"
        return f"✅ Index rebuilt.\n\n{out or 'Done.'}"
    except subprocess.TimeoutExpired:
        return "❌ Reindex timed out."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def build_hipporag_index() -> str:
    """Build HippoRAG graph index from data/documents. Requires hipporag + OPENAI or vLLM."""
    try:
        from src.minerals.hipporag_retrieval import HippoRAGRetriever, hipporag_available
        if not hipporag_available():
            return "❌ HippoRAG not installed. Run: python3 -m pip install hipporag  or  pip install -e \".[hipporag]\""
        docs_dir = PROJECT_ROOT / "data" / "documents"
        save_dir = docs_dir / "hipporag_index"
        retriever = HippoRAGRetriever(documents_dir=str(docs_dir), save_dir=str(save_dir))
        return retriever.index()
    except Exception as e:
        return f"❌ HippoRAG index failed: {e}"


def rag_search(query: str, top_k: int = 5, use_kg_context: bool = False, use_classic_search_only: bool = False) -> str:
    """Retrieve document chunks via RAGPipeline (memory-boosted) and return formatted markdown."""
    try:
        pipeline = _get_pipeline()
        q = query.strip() or "graphite supply trade"
        backend_override = "simple" if use_classic_search_only else None
        if backend_override:
            # Bypass pipeline for explicit classic-only request
            from src.minerals.rag_retrieval import SimpleRAGRetriever
            docs_dir = PROJECT_ROOT / "data" / "documents"
            retriever = SimpleRAGRetriever(str(docs_dir), str(docs_dir / "index.json"))
            if not retriever.chunks:
                return "⚠️ No documents indexed. Run `python scripts/index_rag_documents.py` first."
            chunks = retriever.retrieve(query=q, top_k=max(1, min(20, top_k)))
        else:
            chunks = pipeline.retrieve(query=q, top_k=max(1, min(20, top_k)))
            if not chunks:
                return "⚠️ No documents indexed. Run `python scripts/index_rag_documents.py` first."

        lines = []
        if use_kg_context:
            kg_ctx = get_kg_context_for_rag()
            if kg_ctx:
                lines.append(kg_ctx)
                lines.append("---\n")

        backend_label = f"({pipeline.backend_name})" if not backend_override else "(classic)"
        lines.append(f"**Retrieved {len(chunks)} chunks** {backend_label}\n")
        for i, c in enumerate(chunks, 1):
            meta = c.get("metadata", {}) if isinstance(c.get("metadata"), dict) else {}
            src = meta.get("source_file", c.get("source", "?"))
            sim = c.get("similarity", 0.0)
            text = (c.get("text") or "")[:1500]
            if len(c.get("text") or "") > 1500:
                text += "..."
            lines.append(f"### {i}. `{src}` (score: {sim:.3f})\n{text}\n")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ RAG error: {e}"


def rag_ask(query: str, top_k: int = 5) -> tuple[str, str]:
    """Full Q&A with memory: retrieve → few-shot inject → LLM answer → store episode.

    Returns (answer_markdown, episode_id) for subsequent feedback.
    """
    if not query.strip():
        return ("Enter a question first.", "")
    try:
        from src.llm.chat import is_chat_available
        if not is_chat_available():
            return ("❌ No LLM backend configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your `.env`.", "")

        pipeline = _get_pipeline()
        result = pipeline.ask(query.strip(), top_k=max(1, min(20, top_k)), use_memory=True)

        answer = result.get("answer", "(no answer)")
        episode_id = result.get("episode_id", "")
        sources = result.get("sources", [])

        lines = [f"### Answer\n{answer}\n"]
        if sources:
            lines.append(f"**Sources ({len(sources)}):**")
            for s in sources[:5]:
                meta = s.get("metadata", {}) if isinstance(s.get("metadata"), dict) else {}
                src = meta.get("source_file", s.get("source", "?"))
                sim = s.get("similarity", 0.0)
                lines.append(f"- `{src}` (score: {sim:.3f})")
        lines.append(f"\n*Episode ID: `{episode_id}` — use thumbs to rate this answer.*")
        return ("\n".join(lines), episode_id)
    except Exception as e:
        return (f"❌ Ask error: {e}", "")


def rag_feedback(episode_id: str, rating: float) -> str:
    """Store user rating for the last episode."""
    if not episode_id:
        return "No episode to rate — run Ask first."
    try:
        pipeline = _get_pipeline()
        pipeline.feedback(episode_id, rating=rating)
        label = "👍 positive" if rating > 0 else "👎 negative"
        return f"Feedback recorded ({label}) for episode `{episode_id}`."
    except Exception as e:
        return f"❌ Feedback error: {e}"


def run_rag_eval(n_questions: int = 10, top_k: int = 5) -> str:
    """Generate synthetic questions, evaluate retrieval + answer quality, and trigger self-learning."""
    try:
        from src.llm.chat import is_chat_available
        if not is_chat_available():
            return "❌ No LLM configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        from src.minerals.rag_eval import RAGEvaluator
        pipeline = _get_pipeline()
        ev = RAGEvaluator(pipeline)

        n = max(3, int(n_questions))
        questions = ev.generate_questions(n_chunks=n, questions_per_chunk=1)
        if not questions:
            return "⚠️ No questions generated — check that documents are indexed."

        ret_report = ev.evaluate_retrieval(questions, top_k=int(top_k))
        ans_n = min(5, len(questions))
        ans_report = ev.evaluate_answers(questions[:ans_n], top_k=int(top_k))
        learn_stats = ev.learn(ret_report, ans_report)

        lines = [
            "### RAG Evaluation",
            f"**Questions generated:** {len(questions)}",
            "",
            "**Retrieval quality:**",
            f"- Hit@{int(top_k)}: **{ret_report.get('hit_at_k', 0)*100:.1f}%**",
            f"- MRR@{int(top_k)}: **{ret_report.get('mrr', 0):.3f}**",
            "",
            f"**Answer quality** (n={ans_n}):",
            f"- Avg faithfulness: **{ans_report.get('faithfulness_mean', ans_report.get('avg_faithfulness', 0)):.2f}**",
            f"- Avg relevance: **{ans_report.get('relevance_mean', ans_report.get('avg_relevance', 0)):.2f}**",
            "",
            "**Self-learning:**",
            f"- Episodes stored as few-shot examples: **{learn_stats.get('stored', 0)}**",
            f"- Knowledge gaps logged: **{learn_stats.get('gaps', 0)}**",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Eval error: {e}"


def rag_memory_stats() -> str:
    """Return human-readable memory statistics."""
    try:
        pipeline = _get_pipeline()
        s = pipeline.stats()
        mem = s.get("memory") or {}
        if not mem:
            return "Memory not initialised."
        lines = [
            f"**Memory** (`{mem.get('memory_dir', '')}`):",
            f"- Episodes stored: **{mem.get('n_episodes', 0)}**",
            f"- Avg quality score: **{mem.get('avg_quality', 0):.2f}**",
            f"- Boosted chunks: **{mem.get('n_boosted_chunks', 0)}**",
            f"- Knowledge gaps: **{mem.get('n_gaps', 0)}**",
            f"- Backend: **{s.get('backend', '?')}**",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Stats error: {e}"


# ----- Knowledge Graph (shared helpers for KG tab, RAG, Causal) -----

_kg = None  # singleton — persists enrichment within session
_KG_SAVE_PATH = PROJECT_ROOT / "data" / "knowledge_graph.json"


def _get_kg():
    """Return the KG singleton.  Loads from disk if a saved copy exists; otherwise builds fresh."""
    global _kg
    if _kg is None:
        from src.minerals.knowledge_graph import build_critical_minerals_kg, CausalKnowledgeGraph
        if _KG_SAVE_PATH.exists():
            try:
                _kg = CausalKnowledgeGraph.load(str(_KG_SAVE_PATH))
            except Exception:
                _kg = build_critical_minerals_kg()
        else:
            _kg = build_critical_minerals_kg()
    return _kg


_CRITICAL_MINERALS = [
    "graphite", "lithium", "cobalt", "copper", "nickel",
    "antimony", "beryllium", "cesium", "gallium", "germanium",
    "indium", "niobium", "platinum", "tantalum", "tellurium",
    "titanium", "tungsten", "vanadium", "yttrium", "rare-earths",
]


def kg_batch_enrich(top_k: int = 3) -> str:
    """Enrich the KG with supply chain knowledge for every critical mineral in the corpus."""
    try:
        from src.minerals.kg_extractor import KGExtractor
        from src.llm.chat import is_chat_available
        if not is_chat_available():
            return "❌ No LLM configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY."
        pipeline = _get_pipeline()
        kg = _get_kg()
        extractor = KGExtractor(pipeline)
        before_rels = kg.num_relationships
        before_ents = kg.num_entities
        _KG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        per_mineral: list[str] = []
        for mineral in _CRITICAL_MINERALS:
            query = f"{mineral} supply chain production trade export restrictions"
            try:
                n = extractor.enrich(kg, query, top_k=max(1, int(top_k)))
                per_mineral.append(f"- {mineral}: +{n} triples")
            except Exception as e:
                per_mineral.append(f"- {mineral}: ⚠️ {e}")
        kg.save(str(_KG_SAVE_PATH))
        added_rels = kg.num_relationships - before_rels
        added_ents = kg.num_entities - before_ents
        lines = [
            f"**Batch enrichment complete** ({len(_CRITICAL_MINERALS)} minerals)",
            f"- New relationships: **+{added_rels}** (total: {kg.num_relationships})",
            f"- New entities: **+{added_ents}** (total: {kg.num_entities})",
            f"- Saved to `{_KG_SAVE_PATH.name}`",
            "",
            "**Per-mineral:**",
        ] + per_mineral
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Batch enrich failed: {e}"


def kg_enrich_from_corpus(query: str, top_k: int = 5) -> str:
    """Retrieve corpus chunks for *query*, extract triples, merge into the live KG, and save."""
    if not query.strip():
        return "Enter a query to enrich the KG with."
    try:
        from src.minerals.kg_extractor import KGExtractor
        pipeline = _get_pipeline()
        kg = _get_kg()
        extractor = KGExtractor(pipeline)
        before_rels = kg.num_relationships
        before_ents = kg.num_entities
        _KG_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        n = extractor.enrich(kg, query.strip(), top_k=max(1, min(20, top_k)),
                             save_path=str(_KG_SAVE_PATH))
        added_rels = kg.num_relationships - before_rels
        added_ents = kg.num_entities - before_ents
        lines = [
            f"**KG enriched** from query: `{query.strip()}`",
            f"- Triples extracted: **{n}**",
            f"- New relationships: **+{added_rels}** (total: {kg.num_relationships})",
            f"- New entities: **+{added_ents}** (total: {kg.num_entities})",
            f"- Saved to `{_KG_SAVE_PATH.relative_to(PROJECT_ROOT)}`",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Enrich failed: {e}"


def get_kg_summary() -> str:
    """Return human-readable KG summary (entities, relationship counts)."""
    try:
        kg = _get_kg()
        return kg.summary()
    except Exception as e:
        return f"❌ Failed to build KG: {e}"


def kg_rebuild() -> tuple[str, object]:
    """Reset the KG singleton and rebuild from disk / base, return (summary, shock_choices)."""
    global _kg
    _kg = None  # clear singleton so _get_kg() rebuilds
    try:
        kg = _get_kg()
        return kg.summary(), gr.update(choices=kg.get_shock_origin_candidates())
    except Exception as e:
        return f"❌ {e}", gr.update(choices=[])


def get_kg_shock_sources() -> list[str]:
    """Return entity IDs suitable as shock origins (have outgoing CAUSES)."""
    try:
        kg = _get_kg()
        return kg.get_shock_origin_candidates()
    except Exception:
        return []


def run_kg_shock_propagation(origin_id: str) -> str:
    """Propagate shock from origin_id and return formatted trace."""
    if not (origin_id or origin_id.strip()):
        return "Select or enter a shock origin (e.g. china_export_controls)."
    try:
        kg = _get_kg()
        trace = kg.propagate_shock(origin_id.strip(), initial_magnitude=1.0, decay=0.5, max_depth=5)
        lines = [f"**Shock origin:** `{trace.origin}`", f"**Affected entities ({len(trace.affected)}):**", ""]
        for eid, mag in sorted(trace.affected.items(), key=lambda x: -x[1]):
            path = trace.paths.get(eid, [])
            path_str = " → ".join(path) if path else eid
            lines.append(f"- `{eid}` (impact {mag:.3f}): path `{path_str}`")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Shock propagation failed: {e}"


def run_kg_identifiability() -> str:
    """Run identifiability analysis using the KG-derived causal DAG."""
    try:
        kg = _get_kg()
        dag = kg.to_causal_dag()
        queries = [
            ("china_export_controls", "graphite"),
            ("graphite", "ev_batteries"),
            ("china_export_controls", "ev_batteries"),
        ]
        lines = [
            "**Identifiability (KG-derived DAG)**",
            "",
            "**Do-calculus (Pearl):** Rule 1 — insertion/deletion of observations. Rule 2 — action/observation exchange. Rule 3 — insertion/deletion of actions.",
            "",
            f"Nodes: {len(dag.graph.nodes())}, Edges: {len(dag.graph.edges())}",
            "",
        ]
        for treatment, outcome in queries:
            if treatment not in dag.graph or outcome not in dag.graph:
                lines.append(f"- P({outcome}|do({treatment})): (skip — node not in DAG)")
                continue
            result = dag.is_identifiable(treatment, outcome)
            yes_no = "✅ YES" if result.identifiable else "❌ NO"
            lines.append(f"- **P({outcome}|do({treatment}))**: {yes_no}")
            if result.identifiable:
                lines.append(f"  Formula: {result.formula}")
                if result.strategy:
                    lines.append(f"  Strategy: {result.strategy.value}")
                if result.adjustment_set:
                    lines.append(f"  Adjustment set: {result.adjustment_set}")
                if result.derivation_steps:
                    lines.append("  *Do-calculus derivation:*")
                    for step in result.derivation_steps:
                        if step.strip():
                            lines.append(f"    {step}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ KG identifiability failed: {e}"


def get_kg_dag_edges() -> str:
    """Return text listing of edges in the KG-derived causal DAG."""
    try:
        kg = _get_kg()
        dag = kg.to_causal_dag()
        edges = list(dag.graph.edges())
        if not edges:
            return "(No causal edges in KG-derived DAG.)"
        return "Causal edges (cause → effect):\n" + "\n".join(f"  {u} → {v}" for u, v in sorted(edges))
    except Exception as e:
        return f"❌ {e}"


def get_kg_dag_image(simplified=True):
    """Render KG-derived causal DAG to PNG. simplified=True caps nodes for legibility."""
    try:
        use_simplified = simplified if isinstance(simplified, bool) else (simplified != "Full graph")
        import matplotlib.pyplot as plt
        plt.close("all")  # clear any previous figures
        path = PROJECT_ROOT / "kg_causal_dag.png"
        kg = _get_kg()
        from scripts.run_knowledge_graph import visualize_kg_dag
        max_nodes = 42 if use_simplified else None
        visualize_kg_dag(kg, str(path), max_nodes=max_nodes)
        plt.close("all")
        return str(path)
    except Exception as e:
        # Return path to a simple error placeholder so user sees feedback
        err_path = PROJECT_ROOT / "kg_dag_error.png"
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Visualization failed:\n{str(e)}", ha="center", va="center",
                    fontsize=12, wrap=True)
            ax.axis("off")
            plt.savefig(str(err_path), bbox_inches="tight", dpi=150)
            plt.close("all")
            return str(err_path)
        except Exception:
            return None


def get_kg_dag_interactive_html(simplified=True):
    """Return HTML for an interactive DAG viewer with smooth zoom and pan (vis-network)."""
    import json
    try:
        use_simplified = simplified if isinstance(simplified, bool) else (simplified != "Full graph")
        kg = _get_kg()
        from scripts.run_knowledge_graph import get_kg_dag_interactive_data
        data = get_kg_dag_interactive_data(kg, max_nodes=42 if use_simplified else None)
        if not data or not data.get("nodes"):
            return "<p>No graph data to display. Build the KG first.</p>"
        # Escape JSON for embedding in HTML
        data_js = json.dumps(data).replace("</", "<\\/")
        return _INTERACTIVE_DAG_HTML_TEMPLATE.format(data_js=data_js)
    except Exception as e:
        return f"<p>Interactive viewer failed: {e}</p>"


_INTERACTIVE_DAG_HTML_TEMPLATE = """
<div id="dag-container" style="width:100%; height:520px; background:#fafafa; border-radius:8px;"></div>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<script>
(function() {
  var data = {data_js};
  var nodes = new vis.DataSet(data.nodes.map(function(n) {
    return {{ id: n.id, label: n.label, title: n.id, x: n.x, y: n.y, color: n.color, fixed: true }};
  }));
  var edges = new vis.DataSet(data.edges.map(function(e) {{ return {{ from: e.from, to: e.to, arrows: "to" }}; }}));
  var container = document.getElementById("dag-container");
  var net = new vis.Network(container, {{ nodes: nodes, edges: edges }}, {{
    nodes: {{ shape: "box", font: {{ size: 14 }}, margin: 10, borderWidth: 2 }},
    edges: {{ width: 1.5, smooth: {{ type: "cubicBezier", roundness: 0.2 }}}},
    physics: false,
    interaction: {{ zoomView: true, dragView: true, hover: true, tooltipDelay: 100 }},
    layout: {{ randomSeed: 1 }}
  }});
  net.fit({{ animation: {{ duration: 300, easingFunction: "easeInOutQuad" }}}});
})();
</script>
<p style="margin-top:8px;color:#666;font-size:13px;">Scroll to zoom &bull; Drag background to pan &bull; Hover nodes for full name</p>
"""


def get_kg_context_for_rag() -> str:
    """Short KG context string for RAG augmentation (entities + key causal relations)."""
    try:
        kg = _get_kg()
        summary = kg.summary()
        dag = kg.to_causal_dag()
        all_edges = list(dag.graph.edges())
        edges = sorted(all_edges)[:30]
        head = "**Knowledge Graph context**\n" + summary + "\n\n**Key causal relations:**\n"
        suffix = "\n..." if len(all_edges) > 30 else ""
        return head + "\n".join(f"{u} → {v}" for u, v in edges) + suffix
    except Exception:
        return ""


# ----- Tab: Query Model (unified chain) -----

_FALLBACK_CAUSAL_PAIRS = [
    ("china_export_controls", "graphite"),
    ("graphite", "ev_batteries"),
    ("china_export_controls", "ev_batteries"),
    ("china", "graphite"),
]


def _causal_candidates_for_question(question: str, dag) -> list[tuple[str, str]]:
    """Derive relevant (treatment, outcome) pairs by matching question tokens to DAG node IDs.

    Each DAG node ID like ``china_export_controls`` is split into tokens and checked against
    the question words.  All matched-node pairs are returned (capped at 40 to bound runtime).
    Falls back to the hardcoded graphite/China pairs if nothing matches.
    """
    q_tokens = set(question.lower().replace("?", "").replace(",", "").replace("'", "").split())
    nodes = list(dag.graph.nodes())
    matched: list[str] = []
    for node in nodes:
        node_tokens = set(node.lower().replace("_", " ").replace("-", " ").split())
        if node_tokens & q_tokens:
            matched.append(node)

    if not matched:
        return _FALLBACK_CAUSAL_PAIRS

    pairs = [(t, o) for t in matched for o in matched if t != o]
    return pairs[:40]


def unified_query(question: str, top_k: int = 5) -> str:
    """Chain: RAG retrieve → KG causal ID → simulation do() + backdoor ATE → LLM synthesis."""
    if not question.strip():
        return "Enter a question."

    from src.llm.chat import chat_completion, is_chat_available

    sections: list[str] = []
    causal_numeric_context = ""   # numeric estimates for LLM prompt

    # ── 1. RAG retrieve ──────────────────────────────────────────────────────
    try:
        pipeline = _get_pipeline()
        chunks = pipeline.retrieve(question.strip(), top_k=max(1, min(20, top_k)))
        rag_context = "\n\n".join(
            f"[{i+1}] {(c.get('metadata') or {}).get('source_file', '?')}: "
            f"{(c.get('text') or '')[:600]}"
            for i, c in enumerate(chunks)
        )
        sections.append(
            f"### 1. Retrieved {len(chunks)} documents\n"
            + "\n".join(
                f"- [{i+1}] `{(c.get('metadata') or {}).get('source_file', '?')}`"
                for i, c in enumerate(chunks)
            )
        )
    except Exception as e:
        rag_context = ""
        sections.append(f"### 1. RAG retrieval\n⚠️ {e}")

    # ── 2. Causal identification + numeric estimation ─────────────────────────
    try:
        from src.minerals.causal_inference import GraphiteSupplyChainDAG
        from src.minerals.causal_engine import CausalInferenceEngine

        # Use KG-derived DAG for identifiability, graphite SCM for estimation
        kg = _get_kg()
        kg_dag = kg.to_causal_dag()
        candidates = _causal_candidates_for_question(question.strip(), kg_dag)

        # Also always check graphite SCM pairs relevant to the question
        scm_dag = GraphiteSupplyChainDAG()
        _SCM_PAIRS = [
            ("ExportPolicy", "Price"),
            ("ExportPolicy", "TradeValue"),
            ("Demand", "Price"),
            ("GlobalDemand", "Price"),
        ]
        q_lower = question.lower()
        scm_candidates = [
            (t, o) for t, o in _SCM_PAIRS
            if any(kw in q_lower for kw in (t.lower(), o.lower(),
                   "export", "price", "demand", "trade", "restrict", "impact"))
        ] or [("ExportPolicy", "Price")]  # always include main pair

        id_lines: list[str] = []
        numeric_lines: list[str] = []

        # KG-DAG identifiability
        for t, o in candidates:
            if t in kg_dag.graph and o in kg_dag.graph:
                result = kg_dag.is_identifiable(t, o)
                if result.identifiable:
                    strategy = result.strategy.value if result.strategy else "adjustment"
                    id_lines.append(f"- P({o}|do({t})): ✅ via {strategy} — `{result.formula}`")

        # Graphite SCM: simulation-based do() estimate using default scenario
        try:
            import yaml
            from src.minerals.schema import ScenarioConfig
            _default_scenario = PROJECT_ROOT / "scenarios" / "graphite_baseline_2000_2011.yaml"
            if _default_scenario.exists():
                cfg = ScenarioConfig(**yaml.safe_load(_default_scenario.read_text()))
                engine = CausalInferenceEngine(scm_dag, cfg=cfg)
                for t, o in scm_candidates:
                    try:
                        do_result = engine.do(t, 0.4)  # do(treatment=40%)
                        delta = do_result.effect_on_outcome.get(
                            {"Price": "P", "TradeValue": "TradeValue", "Demand": "D"}.get(o, o), None
                        )
                        if delta is not None:
                            numeric_lines.append(
                                f"- **do({t}=0.4)** → Δ{o} = **{delta:+.3f}** "
                                f"(sim-based, 2000–2011 baseline)"
                            )
                    except Exception:
                        pass
        except Exception:
            pass

        # Graphite SCM: backdoor ATE using cross-scenario data (shock vs baseline)
        # Run a shocked version to get variation in ExportPolicy for ATE estimation
        try:
            from src.minerals.simulate import run_scenario
            import yaml
            from src.minerals.schema import ScenarioConfig
            from src.minerals.schema import ShockConfig
            if _default_scenario.exists():
                cfg_base = ScenarioConfig(**yaml.safe_load(_default_scenario.read_text()))
                cfg_shock = cfg_base.model_copy(deep=True)
                cfg_shock.shocks = list(cfg_shock.shocks) + [
                    ShockConfig(type="export_restriction", start_year=cfg_base.time.start_year,
                                end_year=cfg_base.time.end_year, magnitude=0.4)
                ]
                df_base, _ = run_scenario(cfg_base)
                df_shock, _ = run_scenario(cfg_shock)
                # Pool baseline (ExportPolicy=0) + shock (ExportPolicy=0.4) runs
                df_base["ExportPolicy"] = 0.0
                df_shock["ExportPolicy"] = 0.4
                import pandas as pd
                df_pooled = pd.concat([df_base, df_shock], ignore_index=True)
                df_pooled = df_pooled.rename(columns={"P": "Price", "D": "Demand", "Q": "TradeValue"})

                engine2 = CausalInferenceEngine(scm_dag)
                for t, o in [("ExportPolicy", "Price"), ("Demand", "Price")]:
                    if t in df_pooled.columns and o in df_pooled.columns:
                        try:
                            est = engine2.backdoor_estimate(df_pooled, treatment=t, outcome=o)
                            numeric_lines.append(
                                f"- **Backdoor ATE** {t}→{o}: **{est.ate:+.3f}** "
                                f"95% CI [{est.ate_ci[0]:+.3f}, {est.ate_ci[1]:+.3f}]"
                            )
                        except Exception:
                            pass
        except Exception:
            pass

        matched_nodes = sorted({n for pair in candidates for n in pair if n in kg_dag.graph})
        node_note = (
            f"*(matched KG nodes: {', '.join(f'`{n}`' for n in matched_nodes[:8])})*"
            if matched_nodes else ""
        )
        causal_context = "\n".join(id_lines) if id_lines else "(no identifiable pairs in KG DAG for this question)"
        numeric_context = "\n".join(numeric_lines) if numeric_lines else ""
        causal_numeric_context = numeric_context

        sec2 = f"### 2. Causal identification + estimation\n{node_note}\n{causal_context}"
        if numeric_lines:
            sec2 += f"\n\n**Numeric estimates (graphite SCM):**\n{numeric_context}"
        sections.append(sec2)

    except Exception as e:
        causal_context = ""
        sections.append(f"### 2. Causal identification\n⚠️ {e}")

    # ── 3. LLM synthesis ─────────────────────────────────────────────────────
    if not is_chat_available():
        sections.append("### 3. Synthesis\n❌ No LLM configured — set ANTHROPIC_API_KEY or OPENAI_API_KEY.")
        return "\n\n".join(sections)

    prompt = (
        f"You are a critical minerals supply chain analyst.\n\n"
        f"QUESTION: {question.strip()}\n\n"
        f"RETRIEVED DOCUMENTS:\n{rag_context or '(none)'}\n\n"
        f"CAUSAL STRUCTURE:\n{causal_context or '(none)'}\n\n"
        + (f"NUMERIC CAUSAL ESTIMATES:\n{causal_numeric_context}\n\n" if causal_numeric_context else "")
        + "Write a concise, structured answer that:\n"
        "1. Directly answers the question using the documents (cite [1], [2], …)\n"
        "2. Explains the causal mechanism where relevant\n"
        "3. Quantifies the expected impact using the numeric estimates above when available\n"
        "4. Flags key uncertainties and model assumptions"
    )
    try:
        answer = chat_completion([{"role": "user", "content": prompt}], max_tokens=900)
        sections.append(f"### 3. Synthesis\n{answer}")
    except Exception as e:
        sections.append(f"### 3. Synthesis\n❌ LLM error: {e}")

    return "\n\n".join(sections)


# ----- Report export -----

def export_report(content: str) -> str | None:
    """Save *content* to a timestamped file in runs/ and return the path for download."""
    if not content or not content.strip():
        return None
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "runs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{ts}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


# ----- Tab: Causal Analysis + DAG -----

def show_causal_analysis(scenario_name: str = "") -> str:
    """
    Run identifiability analysis against the SELECTED scenario's shocks.

    - If a scenario is selected: extracts its shocks, maps them to treatment
      nodes, and runs identifiability for those specific (treatment, outcome)
      pairs plus the standard ones.
    - If no scenario: falls back to the standard graphite queries.
    - Always uses the live GraphiteSupplyChainDAG (which includes the
      TradeValue→Price and Demand→Price observed edges we added).
    """
    import yaml
    from src.minerals.causal_inference import GraphiteSupplyChainDAG
    from src.minerals.do_calculus import (
        id_algorithm, rule_1_statement, rule_2_statement, rule_3_statement,
    )

    dag = GraphiteSupplyChainDAG()
    lines = [
        "## Causal Identifiability Analysis (Pearl do-calculus)\n",
        "**Do-calculus rules:**",
        f"- {rule_1_statement()}",
        f"- {rule_2_statement()}",
        f"- {rule_3_statement()}",
        "",
        f"**DAG:** {len(dag.graph.nodes())} nodes "
        f"({len(dag.observed_vars)} observed, {len(dag.unobserved_vars)} unobserved), "
        f"{len(dag.graph.edges())} edges",
        "",
    ]

    # Build queries from selected scenario shocks + always-include standard queries
    queries: list[tuple[str, str]] = []
    scenario_label = "default graphite DAG"

    if scenario_name:
        path = PROJECT_ROOT / "scenarios" / scenario_name
        if path.exists():
            try:
                cfg_raw = yaml.safe_load(path.read_text())
                shocks = cfg_raw.get("shocks", [])
                scenario_label = scenario_name
                for s in shocks:
                    node = _SHOCK_TYPE_TO_CAUSAL.get(s.get("type", ""), ("ExportPolicy",))[0]
                    for outcome in ("Price", "TradeValue"):
                        if (node, outcome) not in queries and node != outcome:
                            queries.append((node, outcome))
            except Exception:
                pass

    # Always include the core graphite queries
    for pair in [("ExportPolicy", "Price"), ("ExportPolicy", "TradeValue"),
                 ("Demand", "Price"), ("GlobalDemand", "Price")]:
        if pair not in queries:
            queries.append(pair)

    lines.append(f"*Scenario: `{scenario_label}`*\n")

    for treatment, outcome in queries:
        if treatment not in dag.graph or outcome not in dag.graph:
            lines.append(f"- P({outcome}|do({treatment})): (skip — node not in DAG)\n")
            continue

        result = dag.is_identifiable(treatment, outcome)
        status = "✅ Identifiable" if result.identifiable else "❌ Not identifiable"
        lines.append(f"### P({outcome} | do({treatment}))")
        lines.append(f"**{status}**")
        if result.identifiable:
            lines.append(f"- Strategy: `{result.strategy.value if result.strategy else 'N/A'}`")
            if result.adjustment_set:
                lines.append(f"- Adjustment set Z: {sorted(result.adjustment_set)}")
            lines.append(f"- Estimand: `{result.formula}`")
            if result.assumptions:
                lines.append("- Assumptions: " + "; ".join(result.assumptions))
            # ID algorithm cross-check
            id_res = id_algorithm(dag, treatment, outcome)
            if id_res["derivation_steps"]:
                deriv = "\n".join(s for s in id_res["derivation_steps"] if s.strip())
                lines.append(f"\n<details><summary>ID algorithm derivation</summary>\n\n```\n{deriv}\n```\n\n</details>")
        else:
            lines.append(f"- Reason: {result.formula}")
        lines.append("")

    # Parameter identification strategies
    lines.append("---\n### Parameter identification strategies")
    for pid in dag.get_parameter_identifications():
        lines.append(
            f"- **{pid.parameter}** (`{pid.description}`): "
            f"`{pid.estimand}` via {pid.strategy.value}"
        )

    return "\n".join(lines)


def get_dag_image_path():
    """Return the most up-to-date DAG image: KG-derived if it exists, else default."""
    for p in ("kg_causal_dag.png", "graphite_causal_dag.png"):
        path = PROJECT_ROOT / p
        if path.exists():
            return str(path)
    return None


def generate_dag_image(scenario_name: str = "") -> str | None:
    """
    Regenerate the DAG image.

    - If the KG has been built/enriched, generates the KG-derived DAG
      (which reflects LLM-extracted causal edges from the corpus).
    - Falls back to the static GraphiteSupplyChainDAG.

    The KG-derived DAG will be DIFFERENT from run to run as more documents
    are indexed and the KG is enriched — it's live, not hardcoded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Prefer KG-derived DAG (reflects enrichment)
    try:
        kg = _get_kg()
        if kg is not None:
            dag = kg.to_causal_dag()
            if len(dag.graph.edges()) > 0:
                path = PROJECT_ROOT / "kg_causal_dag.png"
                dag.visualize(str(path))
                plt.close("all")
                return str(path)
    except Exception:
        pass

    # Fallback: static GraphiteSupplyChainDAG (always has the correct structure)
    try:
        from src.minerals.causal_inference import GraphiteSupplyChainDAG
        path = PROJECT_ROOT / "graphite_causal_dag.png"
        dag = GraphiteSupplyChainDAG()
        dag.visualize(str(path))
        plt.close("all")
        return str(path)
    except Exception:
        return None


# Keep old name as alias for any callers
def generate_default_dag_image():
    return generate_dag_image()


# ----- Tab: Run Scenario -----

def list_scenarios() -> list[str]:
    scenarios_dir = PROJECT_ROOT / "scenarios"
    if not scenarios_dir.exists():
        return []
    return sorted(
        f.name for f in scenarios_dir.iterdir()
        if f.suffix.lower() in (".yaml", ".yml")
    )


def _parse_run_dir_from_stdout(stdout: str) -> str:
    """Extract run directory from 'Outputs: <path>' in scenario script output."""
    for line in reversed((stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("Outputs:") and len(line) > 8:
            return line[8:].strip()
    return ""


def run_scenario_tab(scenario_name: str) -> tuple[str, str]:
    """Run scenario and return (markdown_output, run_dir_to_prefill). Includes scenario YAML in output."""
    empty = ("Select a scenario from the dropdown.", "")
    if not scenario_name:
        return empty
    path = PROJECT_ROOT / "scenarios" / scenario_name
    if not path.exists():
        return (f"❌ File not found: {path}", "")
    try:
        # Read scenario YAML to show after run
        yaml_section = ""
        try:
            yaml_text = path.read_text(encoding="utf-8")
            yaml_section = "\n\n---\n**Scenario YAML** (`" + scenario_name + "`):\n\n```yaml\n" + yaml_text.strip() + "\n```\n"
        except Exception:
            yaml_section = "\n\n*(Could not read scenario file.)*"

        out, err, code = _run(
            [sys.executable, "-m", "scripts.run_scenario", "--scenario", str(path)],
            timeout=120,
        )
        if code != 0:
            return (f"❌ Run failed:\n\n{err}\n\n{out}", "")
        run_dir = _parse_run_dir_from_stdout(out)
        body = f"```\n{out}\n```" + yaml_section
        return (body, run_dir)
    except subprocess.TimeoutExpired:
        return ("❌ Scenario run timed out.", "")
    except Exception as e:
        return (f"❌ Error: {str(e)}", "")


# ----- Causal Ask (unified entry point) -----

def causal_ask(question: str, scenario_name: str = "", top_k: int = 5) -> str:
    """
    Natural-language causal question → complete analysis in one shot.

    Combines every tool in the stack:
      1. RAG retrieval (HippoRAG/SimpleRAG) — what the corpus says
      2. KG causal identification — is the effect identifiable? what's the formula?
      3. Simulation do() — numeric effect from graph surgery (needs scenario)
      4. Backdoor ATE + bootstrap CI — observational causal estimate (needs scenario)
      5. Counterfactual — what would have happened without the shock? (needs scenario)
      6. Supply chain cascade — how the shock propagates through trade network
      7. LLM synthesis — grounded answer citing all of the above
    """
    import yaml
    import numpy as np
    import pandas as pd
    from src.llm.chat import chat_completion, is_chat_available

    if not question.strip():
        return "Enter a question."

    out_sections: list[str] = []
    llm_context_parts: list[str] = []

    # ── 1. RAG ───────────────────────────────────────────────────────────────
    rag_context = ""
    try:
        pipeline = _get_pipeline()
        chunks = pipeline.retrieve(question.strip(), top_k=max(1, min(20, top_k)))
        rag_context = "\n\n".join(
            f"[{i+1}] {(c.get('metadata') or {}).get('source_file', '?')}: "
            f"{(c.get('text') or '')[:500]}"
            for i, c in enumerate(chunks)
        )
        src_list = "\n".join(
            f"- [{i+1}] `{(c.get('metadata') or {}).get('source_file', '?')}`"
            for i, c in enumerate(chunks)
        )
        out_sections.append(f"### Sources retrieved ({len(chunks)})\n{src_list}")
        llm_context_parts.append(f"RETRIEVED DOCUMENTS:\n{rag_context}")
    except Exception as e:
        out_sections.append(f"### Sources\n⚠️ RAG error: {e}")

    # ── 2. Causal identification (KG + graphite SCM) ─────────────────────────
    causal_id_text = ""
    try:
        from src.minerals.causal_inference import GraphiteSupplyChainDAG
        from src.minerals.do_calculus import id_algorithm

        scm_dag = GraphiteSupplyChainDAG()
        kg = _get_kg()
        kg_dag = kg.to_causal_dag() if kg else None

        # Dynamic pairs from question tokens against KG DAG
        kg_pairs: list[tuple[str, str]] = []
        if kg_dag:
            kg_pairs = _causal_candidates_for_question(question.strip(), kg_dag)

        # Core SCM pairs always checked
        q_lower = question.lower()
        scm_pairs = [
            (t, o) for t, o in [
                ("ExportPolicy", "Price"), ("ExportPolicy", "TradeValue"),
                ("Demand", "Price"), ("GlobalDemand", "Price"),
            ]
            if any(kw in q_lower for kw in (
                t.lower(), o.lower(), "export", "price", "demand", "trade",
                "restrict", "impact", "effect", "cause"
            ))
        ] or [("ExportPolicy", "Price")]

        id_lines: list[str] = []
        for t, o in scm_pairs:
            if t not in scm_dag.graph or o not in scm_dag.graph:
                continue
            res = scm_dag.is_identifiable(t, o)
            if res.identifiable:
                id_lines.append(
                    f"- **P({o} | do({t}))**: ✅ `{res.strategy.value if res.strategy else 'identified'}` "
                    f"— `{res.formula}`"
                )

        for t, o in kg_pairs[:6]:
            if kg_dag and t in kg_dag.graph and o in kg_dag.graph:
                res = kg_dag.is_identifiable(t, o)
                if res.identifiable:
                    id_lines.append(
                        f"- **P({o} | do({t}))** (KG): ✅ via `{res.strategy.value if res.strategy else 'identified'}`"
                    )

        causal_id_text = "\n".join(id_lines) if id_lines else "(no identifiable pairs found)"
        out_sections.append(f"### Causal identification\n{causal_id_text}")
        llm_context_parts.append(f"CAUSAL STRUCTURE:\n{causal_id_text}")
    except Exception as e:
        out_sections.append(f"### Causal identification\n⚠️ {e}")

    # ── 3–6. Scenario-based analysis (do, ATE, counterfactual, cascade) ──────
    numeric_text = ""
    if scenario_name:
        path = PROJECT_ROOT / "scenarios" / scenario_name
        if path.exists():
            try:
                from src.minerals.schema import ScenarioConfig, ShockConfig
                from src.minerals.simulate import run_scenario
                from src.minerals.causal_engine import CausalInferenceEngine

                cfg = ScenarioConfig(**yaml.safe_load(path.read_text()))
                primary = next((s for s in cfg.shocks), None)
                treatment_node = "ExportPolicy"
                shock_magnitude = 0.0
                if primary:
                    info = _SHOCK_TYPE_TO_CAUSAL.get(primary.type)
                    if info:
                        treatment_node = info[0]
                    shock_magnitude = primary.magnitude

                scm_dag2 = GraphiteSupplyChainDAG()
                engine = CausalInferenceEngine(scm_dag2, cfg=cfg)

                numeric_lines: list[str] = []

                # do() simulation effect
                try:
                    mag = shock_magnitude if shock_magnitude > 0 else 0.4
                    do_res = engine.do(treatment_node, mag)
                    eff = do_res.effect_on_outcome
                    effects_str = "  ".join(
                        f"Δ{v}: **{d:+.3f}**"
                        for v, d in sorted(eff.items()) if abs(d) > 1e-6
                    )
                    numeric_lines.append(
                        f"**Simulation do({treatment_node}={mag:.2f}):** {effects_str}"
                    )
                except Exception as e:
                    numeric_lines.append(f"*(do() error: {e})*")

                # Backdoor ATE (pooled runs)
                try:
                    cfg_base = cfg.model_copy(deep=True)
                    cfg_base.shocks = []
                    df_base, _ = run_scenario(cfg_base)
                    df_shock, _ = run_scenario(cfg)
                    df_base[treatment_node] = 0.0
                    df_shock[treatment_node] = shock_magnitude if shock_magnitude > 0 else 0.4
                    df_pooled = pd.concat([df_base, df_shock], ignore_index=True)
                    df_pooled = df_pooled.rename(columns={"P": "Price", "D": "Demand", "Q": "TradeValue"})
                    engine2 = CausalInferenceEngine(scm_dag2)
                    ate = engine2.backdoor_estimate(df_pooled, treatment_node, "Price")
                    numeric_lines.append(
                        f"**Backdoor ATE** ({treatment_node} → Price): "
                        f"**{ate.ate:+.4f}** (95% CI [{ate.ate_ci[0]:+.4f}, {ate.ate_ci[1]:+.4f}])"
                    )
                except Exception as e:
                    numeric_lines.append(f"*(ATE error: {e})*")

                # Counterfactual
                try:
                    do_overrides: dict[int, dict[str, float]] = {}
                    for s in cfg.shocks:
                        info = _SHOCK_TYPE_TO_CAUSAL.get(s.type)
                        if info:
                            for yr in range(s.start_year, s.end_year + 1):
                                do_overrides.setdefault(yr, {})[info[1]] = info[2]
                    if do_overrides:
                        df_fact, _ = run_scenario(cfg)
                        engine3 = CausalInferenceEngine(scm_dag2, cfg=cfg)
                        cf = engine3.counterfactual(df_fact, do_overrides=do_overrides, cfg=cfg)
                        mean_d = float(cf.effect["P"].mean()) if "P" in cf.effect.columns else float("nan")
                        peak_d = float(cf.effect["P"].abs().max()) if "P" in cf.effect.columns else float("nan")

                        # Short table: shock years only
                        cf_rows = []
                        for s in cfg.shocks:
                            for yr in range(s.start_year, s.end_year + 1):
                                f_row = df_fact[df_fact["year"] == yr]
                                c_row = cf.counterfactual_trajectory
                                c_row = c_row[c_row["year"] == yr] if "year" in c_row.columns else c_row.iloc[[0]]
                                if not f_row.empty and not c_row.empty:
                                    fp = float(f_row["P"].values[0])
                                    cp = float(c_row["P"].values[0])
                                    cf_rows.append(f"| {yr} | {fp:.3f} | {cp:.3f} | {cp-fp:+.3f} |")

                        cf_table = ""
                        if cf_rows:
                            cf_table = (
                                "\n| Year | Factual | No-shock CF | Δ |\n"
                                "|------|---------|-------------|---|\n"
                                + "\n".join(cf_rows)
                            )
                        numeric_lines.append(
                            f"**Counterfactual** (what if no shock?): "
                            f"mean ΔPrice = {mean_d:+.4f}, peak |Δ| = {peak_d:.4f}"
                            + cf_table
                        )
                except Exception as e:
                    numeric_lines.append(f"*(Counterfactual error: {e})*")

                # Supply chain cascade
                try:
                    from src.minerals.supply_chain_network import GlobalSupplyChainNetwork
                    if primary and primary.type in ("export_restriction", "policy_shock") and shock_magnitude > 0:
                        net = GlobalSupplyChainNetwork()
                        if "graphite" in net.networks:
                            affected = net.simulate_shock("graphite", "China", shock_magnitude, cascade=True)
                            if affected:
                                cascade_lines = "\n".join(
                                    f"  {u} → {v}: {orig:,.0f} → {new:,.0f} "
                                    f"({(1-new/orig)*100:.1f}% cut)"
                                    for u, v, orig, new in affected[:10]
                                )
                                numeric_lines.append(
                                    f"**Supply chain cascade** ({shock_magnitude*100:.0f}% cut from China):\n"
                                    + cascade_lines
                                )
                except Exception:
                    pass

                if numeric_lines:
                    numeric_text = "\n\n".join(numeric_lines)
                    out_sections.append(
                        f"### Quantitative estimates (scenario: `{scenario_name}`)\n"
                        + numeric_text
                    )
                    llm_context_parts.append(f"QUANTITATIVE ESTIMATES:\n{numeric_text}")

            except Exception as e:
                out_sections.append(f"### Quantitative estimates\n⚠️ {e}")

    # ── 7. LLM synthesis ─────────────────────────────────────────────────────
    if not is_chat_available():
        out_sections.append(
            "### Answer\n❌ No LLM — set ANTHROPIC_API_KEY or OPENAI_API_KEY.\n\n"
            "*(Analysis sections above are still valid.)*"
        )
    else:
        context_block = "\n\n".join(llm_context_parts)
        prompt = (
            f"You are a critical minerals supply chain analyst.\n\n"
            f"QUESTION: {question.strip()}\n\n"
            f"{context_block}\n\n"
            "Write a concise, structured answer (3–5 paragraphs) that:\n"
            "1. Directly answers the question, citing retrieved documents by [number]\n"
            "2. States the causal mechanism using the identification results\n"
            "3. Quantifies the impact using the numeric estimates (do-effects, ATE, counterfactual) "
            "if available — be specific about magnitudes\n"
            "4. Notes key uncertainties (model assumptions, data gaps)\n"
        )
        try:
            answer = chat_completion([{"role": "user", "content": prompt}], max_tokens=1000)
            out_sections.append(f"### Answer\n{answer}")
        except Exception as e:
            out_sections.append(f"### Answer\n❌ LLM error: {e}")

    return "\n\n---\n\n".join(out_sections)


# ----- Tab: Run Scenario + Causal Analysis -----

# Map ShockConfig type → (DAG treatment node, ShockSignals field to zero for counterfactual)
_SHOCK_TYPE_TO_CAUSAL = {
    "export_restriction":  ("ExportPolicy",  "export_restriction",     0.0),
    "policy_shock":        ("ExportPolicy",  "policy_supply_mult",     1.0),
    "demand_surge":        ("Demand",        "demand_surge",           0.0),
    "macro_demand_shock":  ("GlobalDemand",  "demand_destruction_mult",1.0),
    "capacity_reduction":  ("Capacity",      "capacity_supply_mult",   1.0),
    "capex_shock":         ("Capacity",      "capex_shock",            0.0),
    "stockpile_release":   ("Inventory",     "stockpile_release",      0.0),
}


def run_scenario_causal(scenario_name: str) -> str:
    """
    Run a scenario and apply all three Pearl layers to the results.

    Layer 1 — Association:   Correlation matrix of simulation outputs.
    Layer 2 — Intervention:  do-calculus identification + symbolic estimand,
                             simulation-based do() effect, and backdoor ATE.
    Layer 3 — Counterfactual: "What would prices have been WITHOUT the shock?"
                              Pearl 3-step: abduct noise → action (remove shock)
                              → predict alternate trajectory.
    Supply chain: cascading shock propagation through the trade network.
    """
    import yaml
    import numpy as np
    import pandas as pd

    if not scenario_name:
        return "Select a scenario from the dropdown."

    path = PROJECT_ROOT / "scenarios" / scenario_name
    if not path.exists():
        return f"❌ Scenario not found: {path}"

    try:
        from src.minerals.schema import ScenarioConfig, ShockConfig
        from src.minerals.simulate import run_scenario
        from src.minerals.causal_inference import GraphiteSupplyChainDAG
        from src.minerals.causal_engine import CausalInferenceEngine
        from src.minerals.do_calculus import id_algorithm
        from src.minerals.shocks import ShockSignals

        cfg = ScenarioConfig(**yaml.safe_load(path.read_text()))
        dag = GraphiteSupplyChainDAG()
        years = list(range(cfg.time.start_year, cfg.time.end_year + 1))

        # ── Identify primary shock ──────────────────────────────────────────
        primary = next((s for s in cfg.shocks), None)
        treatment_node = "ExportPolicy"
        cf_field = "export_restriction"
        cf_baseline_val = 0.0
        shock_magnitude = 0.0

        if primary:
            info = _SHOCK_TYPE_TO_CAUSAL.get(primary.type)
            if info:
                treatment_node, cf_field, cf_baseline_val = info
            shock_magnitude = primary.magnitude

        # ── Run baseline and shocked scenarios ─────────────────────────────
        cfg_base = cfg.model_copy(deep=True)
        cfg_base.shocks = []
        df_base, metrics_base = run_scenario(cfg_base)
        df_shock, metrics_shock = run_scenario(cfg)

        sections = [f"## Causal Analysis — `{scenario_name}`\n"]

        # Shock summary
        shock_lines = []
        for s in cfg.shocks:
            shock_lines.append(
                f"- `{s.type}` magnitude={s.magnitude:.2f} "
                f"({s.start_year}–{s.end_year})"
            )
        sections.append(
            "### Scenario shocks\n" + ("\n".join(shock_lines) if shock_lines else "*(none)*")
        )

        # ── LAYER 1 — Association ───────────────────────────────────────────
        try:
            engine = CausalInferenceEngine(dag, cfg=cfg)
            vars_l1 = [c for c in ["P", "D", "Q", "shortage", "tight", "cover"]
                       if c in df_shock.columns]
            corr = engine.correlate(df_shock, variables=vars_l1)

            # Build correlation table
            n = len(vars_l1)
            header = "| | " + " | ".join(vars_l1) + " |"
            sep    = "|---|" + "|".join(["---"] * n) + "|"
            rows = []
            for i in range(n):
                row = f"| **{vars_l1[i]}** |"
                for j in range(n):
                    r_val = float(corr.pearson[i, j])
                    p_val = float(corr.p_values[i, j])
                    star = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    cell = f" {r_val:.2f}{star} " if i != j else " 1.00 "
                    row += cell + "|"
                rows.append(row)

            # Independence test: use shock column (≠ outcome) to avoid duplicate-column bug
            shock_col = next((c for c in df_shock.columns if c.startswith("shock_") and df_shock[c].std() > 1e-10), None)
            ind_lines = ""
            if shock_col and "P" in df_shock.columns:
                ind = engine.test_independence(df_shock, shock_col, "P")
                ind_lines = (
                    f"\n**Independence test** {shock_col} _||_ P: "
                    f"r = {ind.test_statistic:.3f}, p = {ind.p_value:.4f} "
                    f"({'independent' if ind.independent else 'NOT independent'})\n"
                )

            # OLS association (not causal) — rename columns to match DAG node names
            df_renamed = df_shock.rename(columns={"P": "Price", "D": "Demand"})
            assoc_preds = [c for c in ["Demand", "shortage", "tight"] if c in df_renamed.columns]
            assoc = engine.regression_association(df_renamed, outcome="Price", predictors=assoc_preds)
            assoc_lines = "\n".join(
                f"  - {var}: β = {b:+.4f}  (SE={assoc.std_errors.get(var, float('nan')):.4f})"
                for var, b in assoc.coefficients.items()
            )

            sections.append(
                "---\n### Layer 1 — Association: P(Y|X)\n\n"
                "Pairwise Pearson correlations (* p<0.05, ** p<0.01, *** p<0.001):\n\n"
                + header + "\n" + sep + "\n" + "\n".join(rows) + "\n\n"
                + ind_lines
                + f"**OLS regression** Price ~ {assoc_preds} (R² = {assoc.r_squared:.3f}):\n"
                + assoc_lines + "\n\n"
                + "> These are associations only — not causal effects. "
                  "Confounding, reverse causation and collider bias can distort them."
            )
        except Exception as e:
            sections.append(f"---\n### Layer 1 — Association\n⚠️ {e}")

        # ── LAYER 2 — Intervention ──────────────────────────────────────────
        try:
            # 2a. do-calculus identification
            id_result = id_algorithm(dag, treatment_node, "Price")
            id_status = "✅ Identifiable" if id_result["identifiable"] else "❌ Not identifiable"
            deriv = "\n".join(
                f"  {s}" for s in id_result["derivation_steps"] if s.strip()
            )

            # 2b. Simulation-based do() effect
            engine2 = CausalInferenceEngine(dag, cfg=cfg)
            do_result = engine2.do(treatment_node, shock_magnitude if shock_magnitude > 0 else 0.4)
            effect = do_result.effect_on_outcome
            effect_lines = "\n".join(
                f"  - Δ{var}: **{delta:+.4f}**"
                for var, delta in sorted(effect.items())
                if abs(delta) > 1e-6
            )

            # 2c. Backdoor ATE: pool baseline (treatment=0) + shock (treatment=magnitude)
            df_b = df_base.copy(); df_b[treatment_node] = 0.0
            df_s = df_shock.copy(); df_s[treatment_node] = shock_magnitude if shock_magnitude > 0 else 0.4
            df_pooled = pd.concat([df_b, df_s], ignore_index=True)
            df_pooled = df_pooled.rename(columns={"P": "Price", "D": "Demand", "Q": "TradeValue"})
            ate_result = engine2.backdoor_estimate(df_pooled, treatment_node, "Price")
            ate_adj = sorted(ate_result.adjustment_set) if ate_result.adjustment_set else ["∅ (no confounders)"]

            sections.append(
                "---\n### Layer 2 — Intervention: P(Y|do(X))\n\n"
                f"#### do-calculus identification\n"
                f"- **Query:** P(Price | do({treatment_node}))\n"
                f"- **Status:** {id_status}\n"
                f"- **Strategy:** {id_result['strategy']}\n"
                f"- **Estimand:** `{id_result['formula']}`\n"
                f"- **Adjustment set:** {sorted(id_result.get('bidirected_edges') or [])}\n\n"
                f"<details><summary>Full derivation trace</summary>\n\n```\n{deriv}\n```\n\n</details>\n\n"
                f"#### Simulation-based do({treatment_node} = {shock_magnitude:.2f})\n"
                f"*Graph surgery: run structural model with and without shock, compare trajectories.*\n\n"
                + effect_lines + "\n\n"
                f"#### Backdoor ATE (observational, pooled baseline + shock)\n"
                f"- **ATE** ({treatment_node} → Price): **{ate_result.ate:+.4f}**\n"
                f"- **95% CI:** [{ate_result.ate_ci[0]:+.4f}, {ate_result.ate_ci[1]:+.4f}]\n"
                f"- **Adjustment set Z:** {ate_adj}\n"
                f"- **Method:** OLS Y ~ X + Z, bootstrap n=200\n\n"
                f"> The ATE is the average change in Price per unit increase in {treatment_node}, "
                f"adjusting for confounders Z. Because ExportPolicy is a root node (no backdoor paths), "
                f"Z = ∅ and the estimate reduces to a simple regression coefficient."
            )
        except Exception as e:
            sections.append(f"---\n### Layer 2 — Intervention\n⚠️ {e}")

        # ── LAYER 3 — Counterfactual ────────────────────────────────────────
        try:
            engine3 = CausalInferenceEngine(dag, cfg=cfg)

            # Build do_overrides: for each year in shock period, zero out the shock
            do_overrides: dict[int, dict[str, float]] = {}
            for s in cfg.shocks:
                cf_field_s, cf_val_s = _SHOCK_TYPE_TO_CAUSAL.get(
                    s.type, (None, cf_field, cf_baseline_val)
                )[1], _SHOCK_TYPE_TO_CAUSAL.get(
                    s.type, (None, cf_field, cf_baseline_val)
                )[2]
                for yr in range(s.start_year, s.end_year + 1):
                    do_overrides.setdefault(yr, {})[cf_field_s] = cf_val_s

            cf_result = engine3.counterfactual(df_shock, do_overrides=do_overrides, cfg=cfg)

            # Build comparison table (factual vs counterfactual price)
            cf_df = cf_result.counterfactual_trajectory
            f_df = df_shock

            year_col = "year"
            tbl_rows = []
            peak_delta = 0.0
            for i, yr in enumerate(years):
                f_row = f_df[f_df[year_col] == yr]
                c_row = cf_df[cf_df[year_col] == yr] if year_col in cf_df.columns else cf_df.iloc[[i]]
                if f_row.empty or c_row.empty:
                    continue
                f_p = float(f_row["P"].values[0]) if "P" in f_row.columns else float("nan")
                c_p = float(c_row["P"].values[0]) if "P" in c_row.columns else float("nan")
                delta = c_p - f_p
                peak_delta = max(peak_delta, abs(delta))
                in_shock = any(s.start_year <= yr <= s.end_year for s in cfg.shocks)
                marker = " ◀ shock" if in_shock else ""
                tbl_rows.append(f"| {yr} | {f_p:.3f} | {c_p:.3f} | {delta:+.3f}{marker} |")

            mean_d = float(cf_result.effect["P"].mean()) if "P" in cf_result.effect.columns else float("nan")
            abduction_rmse = cf_result.abduction.fit_error

            tbl = (
                "| Year | Factual Price | Counterfactual Price | Δ (CF − Factual) |\n"
                "|------|--------------|---------------------|------------------|\n"
                + "\n".join(tbl_rows)
            )

            sections.append(
                "---\n### Layer 3 — Counterfactual: P(Y_x | X', Y')\n\n"
                '*"What would prices have been if the shock had **NOT** occurred?"\n\n'
                "*Pearl's 3 steps:*\n"
                "1. **Abduct** — infer exogenous noise ε from the factual trajectory\n"
                "2. **Act** — remove the shock from structural equations (do_overrides)\n"
                "3. **Predict** — replay with same ε but modified equations\n\n"
                f"*Abduction RMSE (noise recovery quality): {abduction_rmse:.6f}*\n\n"
                + tbl + "\n\n"
                f"**Mean Δ Price (counterfactual − factual):** {mean_d:+.4f}\n"
                f"**Peak |Δ Price|:** {peak_delta:.4f}\n\n"
                "> A positive Δ means prices would have been **higher** without the shock. "
                "A negative Δ means the shock raised prices (e.g. export restriction → higher price)."
            )
        except Exception as e:
            sections.append(f"---\n### Layer 3 — Counterfactual\n⚠️ {e}")

        # ── Supply Chain Cascade ────────────────────────────────────────────
        try:
            from src.minerals.supply_chain_network import GlobalSupplyChainNetwork
            if primary and primary.type in ("export_restriction", "policy_shock") and shock_magnitude > 0:
                net = GlobalSupplyChainNetwork()
                if "graphite" in net.networks:
                    affected = net.simulate_shock("graphite", "China", shock_magnitude, cascade=True)
                    if affected:
                        aff_lines = "\n".join(
                            f"  - {u} → {v}: {orig:,.0f} → {new:,.0f} "
                            f"({(1 - new/orig)*100:.1f}% reduction)"
                            for u, v, orig, new in affected[:15]
                        )
                        sections.append(
                            "---\n### Supply Chain Cascade\n"
                            f"*Cascading effect of a **{shock_magnitude*100:.0f}% export restriction** "
                            f"from China through the graphite trade network:*\n\n"
                            + aff_lines
                            + ("\n  - *(truncated — 15 of {} edges shown)*".format(len(affected))
                               if len(affected) > 15 else "")
                        )
        except Exception as e:
            sections.append(f"---\n### Supply Chain Cascade\n⚠️ {e}")

        return "\n\n".join(sections)

    except Exception as e:
        import traceback
        return f"❌ Causal analysis failed:\n\n```\n{traceback.format_exc()}\n```"


# ----- Tab: Validate with RAG -----

def validate_with_rag(run_dir: str, year: int | None) -> str:
    if not run_dir or not run_dir.strip():
        return "Enter a run directory (e.g. `runs/graphite_2008_multishock/20260129_132854`)."
    try:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "validate_with_real_rag.py"), "--run-dir", run_dir.strip()]
        if year is not None and year > 0:
            cmd.extend(["--year", str(int(year))])
        out, err, code = _run(cmd, timeout=120)
        if code != 0:
            return f"❌ Validation failed:\n\n{err}\n\n{out}"
        return f"```\n{out}\n```"
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ----- Tab: Synthetic Control -----

def run_synthetic_control() -> str:
    try:
        out, err, code = _run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "test_synthetic_control.py")],
            timeout=60,
        )
        if code != 0:
            return f"❌ Error:\n\n{err}\n\n{out}"
        return f"```\n{out}\n```"
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nEnsure `data/canonical/comtrade_graphite_trade.normalized.csv` exists."


def run_pomdp_build(data_path: str = "", priors_path: str = "") -> str:
    """Run POMDP build and simulation via scripts.build_pomdp. Optional data/priors paths."""
    cmd = [sys.executable, "-m", "scripts.build_pomdp", "--out-dir", "artifacts/pomdp"]
    if (p := (data_path or "").strip()):
        cmd.extend(["--data", p])
    if (q := (priors_path or "").strip()):
        cmd.extend(["--priors", q])
    try:
        out, err, code = _run(cmd, timeout=120)
        if code != 0:
            return f"❌ POMDP build failed:\n\n{err}\n\n{out}"
        return f"```\n{out}\n```\n\nDOT graphs written to `artifacts/pomdp/graphs/`."
    except subprocess.TimeoutExpired:
        return "❌ POMDP build timed out (>120s)."
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nDefault data: `data/sensor_test_data.csv`. Optional priors: JSON with priors_T_alpha, priors_Z_alpha, etc."


def _default_sensor_data_path() -> Path:
    for p in ("data/sensor_test_data.csv", "tests/sensor_test_data.csv"):
        path = PROJECT_ROOT / p
        if path.exists():
            return path
    return PROJECT_ROOT / "data/sensor_test_data.csv"


def run_sensor_causal_estimation(data_path: str = "") -> str:
    """Run DoWhy ATE estimation on sensor data using dag_registry/sensor_reliability.dot."""
    path = (data_path or "").strip() or str(_default_sensor_data_path())
    if not Path(path).exists():
        return f"❌ Data file not found: {path}"
    dag_path = PROJECT_ROOT / "dag_registry" / "sensor_reliability.dot"
    if not dag_path.exists():
        return f"❌ DAG not found: {dag_path}"
    try:
        import pandas as pd
        from src.estimate import estimate_from_dag_path

        df = pd.read_csv(path)
        result = estimate_from_dag_path(
            df=df,
            treatment="CalibrationInterval",
            outcome="Failure",
            controls=["MaterialType", "Temperature", "Drift"],
            dag_path=str(dag_path),
        )
        return (
            f"**Sensor causal estimation (DoWhy)**\n\n"
            f"- **ATE**: {result.ate:.4f}\n"
            f"- **95% CI**: [{result.ate_ci[0]:.4f}, {result.ate_ci[1]:.4f}]\n"
            f"- **Method**: {result.method or 'N/A'}\n\n"
            f"*Same data/DAG used by POMDP and causal estimation.*"
        )
    except Exception as e:
        return f"❌ Estimation error: {str(e)}\n\nEnsure CSV has columns: CalibrationInterval, Failure, MaterialType, Temperature, Drift."


def run_pomdp_and_causal(data_path: str = "", priors_path: str = "") -> str:
    """Run POMDP build and sensor causal estimation (same data) and return combined report."""
    pomdp_out = run_pomdp_build(data_path=data_path, priors_path=priors_path)
    if pomdp_out.startswith("❌"):
        return pomdp_out
    causal_out = run_sensor_causal_estimation(data_path=data_path or str(_default_sensor_data_path()))
    return f"### POMDP\n{pomdp_out}\n\n---\n### Causal estimation (same sensor data)\n{causal_out}"


def run_pomdp_causal_integrated(data_path: str = "") -> str:
    """Run POMDP in-process, extract terminal belief, and integrate with causal ATE.

    The POMDP terminal belief gives P(state) after observing the data.
    We use P(at_risk) = P(degrading) + P(failed) to weight the causal effect:
        adjusted_impact = |ATE| × P(at_risk)
    This answers: "Given our belief about the current system state, how much does
    the calibration interval causally affect failure risk?"
    """
    import numpy as np
    import pandas as pd

    path = (data_path or "").strip() or str(_default_sensor_data_path())
    if not Path(path).exists():
        return f"❌ Data file not found: {path}"

    dag_path = PROJECT_ROOT / "dag_registry" / "sensor_reliability.dot"
    if not dag_path.exists():
        return f"❌ DAG not found: {dag_path}"

    try:
        from src.pomdp.fit import fit_pomdp
        from src.pomdp.simulate import rollout
        from src.pomdp.policies import make_policy_qmdp
        from src.estimate import estimate_from_dag_path

        df = pd.read_csv(path)

        # Sensor CSV has Failure column but no observation column — use Failure as proxy observation
        obs_col = "observation" if "observation" in df.columns else "Failure"
        if obs_col not in df.columns:
            return "❌ CSV missing both 'observation' and 'Failure' columns."

        # Synthesise episode_id if missing (treat each row as its own episode)
        if "episode_id" not in df.columns:
            df = df.copy()
            df["episode_id"] = range(len(df))

        # Fit POMDP from sensor data
        pomdp, meta = fit_pomdp(df, obs_col=obs_col)

        # Simulate 20-step rollout with QMDP policy
        b0 = np.ones(len(pomdp.S)) / len(pomdp.S)
        policy = make_policy_qmdp(pomdp)
        sim = rollout(pomdp, policy, b0, horizon=20)

        terminal_belief = sim["belief_history"][-1]
        belief_dict = dict(zip(pomdp.S, terminal_belief))

        # Causal ATE from DoWhy
        causal_result = estimate_from_dag_path(
            df=df,
            treatment="CalibrationInterval",
            outcome="Failure",
            controls=["MaterialType", "Temperature", "Drift"],
            dag_path=str(dag_path),
        )

        # Belief-weighted impact
        p_failed = belief_dict.get("failed", 0.0)
        p_degrading = belief_dict.get("degrading", 0.0)
        p_at_risk = p_failed + p_degrading
        adjusted_impact = abs(causal_result.ate) * p_at_risk

        belief_lines = "\n".join(
            f"  - P({s}) = **{b:.3f}**" for s, b in sorted(belief_dict.items())
        )
        actions = sim["action_history"]
        most_common_action = max(set(actions), key=actions.count) if actions else "?"

        return (
            "### POMDP + Causal Integration\n\n"
            f"**POMDP fit:** {meta['n_episodes']} episodes, {meta['n_samples']} samples\n\n"
            f"**Terminal belief** (after 20-step simulation):\n{belief_lines}\n\n"
            f"**Dominant policy action:** `{most_common_action}`\n"
            f"**Total discounted reward:** {sim['total_reward']:.2f}\n\n"
            "---\n\n"
            f"**Causal ATE** (CalibrationInterval → Failure): **{causal_result.ate:.4f}**\n"
            f"95% CI: [{causal_result.ate_ci[0]:.4f}, {causal_result.ate_ci[1]:.4f}]\n\n"
            "---\n\n"
            "**Belief-weighted causal impact**\n\n"
            f"P(at_risk) = P(degrading) + P(failed) = **{p_at_risk:.3f}**\n\n"
            f"Adjusted impact = |ATE| × P(at_risk) = **{adjusted_impact:.4f}**\n\n"
            f"*Interpretation: Given the POMDP belief that {p_at_risk*100:.1f}% of sensors are at risk, "
            f"the causal effect of calibration interval translates to an expected failure change "
            f"of {adjusted_impact:.4f} per unit change in interval — weighted by current system state uncertainty.*"
        )
    except Exception as e:
        return f"❌ Integrated analysis failed: {e}"


def run_full_pipeline_mineral(scenario_name: str) -> tuple[str, str]:
    """Run causal analysis, then scenario, then synthetic control. Return (combined_md, run_dir)."""
    sections = []
    run_dir = ""
    if not scenario_name:
        return ("Select a scenario.", "")
    # 1) Causal analysis
    try:
        out, err, code = _run([sys.executable, "-m", "src.minerals.causal_inference"], timeout=30)
        sections.append("### 1. Causal identifiability\n" + (f"```\n{out}\n```" if code == 0 else f"❌ {err}"))
    except Exception as e:
        sections.append(f"### 1. Causal identifiability\n❌ {e}")
    # 2) Run scenario
    path = PROJECT_ROOT / "scenarios" / scenario_name
    if path.exists():
        try:
            out, err, code = _run(
                [sys.executable, "-m", "scripts.run_scenario", "--scenario", str(path)],
                timeout=120,
            )
            run_dir = _parse_run_dir_from_stdout(out)
            sections.append("### 2. Scenario run\n" + (f"```\n{out}\n```" if code == 0 else f"❌ {err}\n{out}"))
        except Exception as e:
            sections.append(f"### 2. Scenario run\n❌ {e}")
    else:
        sections.append(f"### 2. Scenario run\n❌ Not found: {path}")
    # 3) Synthetic control
    try:
        out, err, code = _run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "test_synthetic_control.py")],
            timeout=60,
        )
        sections.append("### 3. Synthetic control\n" + (f"```\n{out}\n```" if code == 0 else f"⚠️ {err or out}"))
    except Exception as e:
        sections.append(f"### 3. Synthetic control\n⚠️ {e}")
    return ("\n\n".join(sections), run_dir)


def run_full_pipeline_sensor(data_path: str = "", priors_path: str = "") -> str:
    """Run POMDP + sensor causal estimation (same data). Single combined report."""
    return run_pomdp_and_causal(data_path=data_path, priors_path=priors_path)


# ----- Three Layers (CausalInferenceEngine) -----

def three_layers_query(
    run_dir: str,
    layer: str,
    treatment: str,
    outcome: str,
    cf_year: str,
    cf_value: str,
) -> str:
    """Run a Pearl Ladder query through CausalInferenceEngine."""
    import pandas as pd
    import numpy as np
    from src.minerals.causal_engine import CausalInferenceEngine
    from src.minerals.causal_inference import GraphiteSupplyChainDAG

    dag = GraphiteSupplyChainDAG()
    engine = CausalInferenceEngine(dag=dag)

    # Load simulation timeseries if available
    data = None
    run_path = Path(run_dir.strip()) if run_dir.strip() else None
    if run_path and (run_path / "timeseries.csv").exists():
        try:
            data = pd.read_csv(run_path / "timeseries.csv")
        except Exception as e:
            return f"❌ Could not load timeseries.csv: {e}"

    treatment = treatment.strip() or "ExportPolicy"
    outcome = outcome.strip() or "Price"

    layer_num = layer.split()[0]  # "1", "2", or "3"

    try:
        if layer_num == "1":
            # Layer 1: Association — correlate observed variables
            if data is None:
                return "⚠️ Layer 1 requires a run directory with timeseries.csv."
            numeric_cols = list(data.select_dtypes(include=[np.number]).columns)[:6]
            if not numeric_cols:
                return "⚠️ No numeric columns in timeseries.csv."
            corr = engine.correlate(data, variables=numeric_cols)
            lines = [f"## Layer 1 — Association: P({outcome}|{treatment})\n"]
            lines.append("**Pairwise Pearson correlations** (observational — not causal):\n")
            lines.append("| | " + " | ".join(corr.variables) + " |")
            lines.append("|" + "---|" * (len(corr.variables) + 1))
            for i, v in enumerate(corr.variables):
                row = f"| **{v}** |"
                for j in range(len(corr.variables)):
                    r = corr.pearson[i, j]
                    p = corr.p_values[i, j]
                    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    row += f" {r:.3f}{star} |"
                lines.append(row)
            lines.append("\n*: p<0.05  **: p<0.01  ***: p<0.001")
            lines.append("\n⚠️ **Correlations are NOT causal.** Use Layer 2 for P(Y|do(X)).")
            if treatment in data.columns and outcome in data.columns:
                ind = engine.test_independence(data, treatment, outcome)
                lines.append(f"\n**Independence test** {treatment} ⊥ {outcome}: "
                              f"r={ind.test_statistic:.3f}, p={ind.p_value:.4f} "
                              f"→ {'independent' if ind.independent else 'associated'}")
            return "\n".join(lines)

        elif layer_num == "2":
            # Layer 2: Intervention — P(outcome | do(treatment))
            id_result = engine.identify(treatment, outcome)
            lines = [f"## Layer 2 — Intervention: P({outcome}|do({treatment}))\n"]
            lines.append(f"**Identifiable:** {'✅ YES' if id_result.identifiable else '❌ NO'}")
            if id_result.strategy:
                lines.append(f"**Strategy:** {id_result.strategy.value}")
            if id_result.adjustment_set:
                lines.append(f"**Adjustment set:** {id_result.adjustment_set}")
            lines.append(f"**Formula:** `{id_result.formula}`")
            if id_result.derivation_steps:
                lines.append("\n**Do-calculus derivation:**")
                for step in id_result.derivation_steps:
                    lines.append(f"- {step}")
            lines.append("\n**Identification assumptions:**")
            for a in id_result.assumptions:
                lines.append(f"- {a}")
            if data is not None and id_result.identifiable and treatment in data.columns and outcome in data.columns:
                adj_cols = [c for c in id_result.adjustment_set if c in data.columns]
                est = engine.backdoor_estimate(data, treatment, outcome, adjustment_set=set(adj_cols))
                lines.append(f"\n**Backdoor ATE estimate:** {est.ate:.4f} "
                              f"(95% CI: [{est.ate_ci[0]:.4f}, {est.ate_ci[1]:.4f}])")
            return "\n".join(lines)

        else:
            # Layer 3: Counterfactual
            if data is None:
                return "⚠️ Layer 3 requires a run directory with timeseries.csv."
            if not cf_year.strip() or not cf_value.strip():
                return "⚠️ Layer 3 requires a counterfactual year and value."
            try:
                year = int(cf_year.strip())
                value = float(cf_value.strip())
            except ValueError:
                return "❌ Counterfactual year must be an integer and value must be a number."

            cfg_path = run_path / "scenario.yaml" if run_path else None
            cfg = None
            if cfg_path and cfg_path.exists():
                from src.minerals.schema import load_scenario
                try:
                    cfg = load_scenario(str(cfg_path))
                except Exception:
                    pass

            if cfg is None:
                return (
                    "⚠️ Layer 3 needs a scenario.yaml in the run directory to run the structural model.\n"
                    "Run a scenario first, then use its run directory here."
                )

            cf_engine = CausalInferenceEngine(dag=dag, cfg=cfg)
            result = cf_engine.counterfactual(
                data=data,
                do_overrides={year: {treatment: value}},
                observed_price_col=outcome if outcome in data.columns else "price",
            )
            lines = [f"## Layer 3 — Counterfactual: What if {treatment}={value} in {year}?\n"]
            lines.append(f"**Abduction fit RMSE:** {result.abduction.fit_error:.4f}")
            lines.append("\n**Effect (counterfactual − factual) per year:**\n")
            lines.append("| Year | Δ " + outcome + " |")
            lines.append("|------|---------|")
            effect_col = outcome.lower() if outcome.lower() in result.effect.columns else result.effect.columns[0]
            for _, row in result.effect.iterrows():
                yr_val = row.get("year", "?")
                delta = row.get(effect_col, row.iloc[0])
                lines.append(f"| {yr_val} | {delta:+.4f} |")
            summary = result.summary
            if summary:
                lines.append("\n**Summary:**")
                for k, v in summary.items():
                    lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
            return "\n".join(lines)

    except Exception:
        import traceback
        return f"❌ Error in Layer {layer_num}:\n```\n{traceback.format_exc()}\n```"


# ----- Build UI -----

demo = gr.Blocks(title="Critical Minerals Causal Engine")

with demo:
    gr.Markdown(
        """
    # 🔬 Critical Minerals Causal Engine

    **Ask questions → Run scenarios → Search documents → Explore causal structure.** All in one app.
    """
    )

    # Shared run directory (Run Scenario / Unified Workflow update this; Validate uses it)
    run_dir_in = gr.Textbox(
        label="Last run directory (auto-filled after Run Scenario; use in Validate tab)",
        placeholder="e.g. runs/graphite_baseline/20260129_132854",
        lines=1,
        visible=True,
    )

    with gr.Tabs():
        # --- Get Started (suggested flow) ---
        with gr.Tab("🏠 Get Started"):
            gr.Markdown(
                """
                ### Suggested flow

                | Step | Tab | What to do |
                |------|-----|------------|
                | **1** | **Ask a Question** | Type a natural-language question (e.g. *"What if China restricts graphite exports by 40%?"*). Click **Reason** for a causal answer, or **Run Simulation** to generate and run a scenario. |
                | **2** | **Run Scenario** | Pick a scenario from the dropdown and click **Run** to simulate supply-chain dynamics. Outputs go to `runs/<name>/<timestamp>/`. |
                | **3** | **Search Documents** | Search or ask questions over your corpus (USGS, CEPII, etc.). Upload files and rebuild index as needed. |
                | **4** | **Knowledge Graph** | Build the KG, visualize the causal DAG, propagate shocks, enrich from documents. |
                | **5** | **Validate** | After running a scenario, validate results against historical data and RAG. |

                ---
                **First time?** Make sure documents are indexed (Search Documents → Rebuild search index) and the KG is built (Knowledge Graph → Build KG).
                """
            )

        # --- Causal Ask (unified single-entry causal tool) ---
        with gr.Tab("🔮 Causal Ask"):
            gr.Markdown(
                "**One question → full causal analysis.**\n\n"
                "Combines RAG retrieval, causal identification (Shpitser-Pearl ID), "
                "simulation-based do-calculus, backdoor ATE with bootstrap CI, "
                "counterfactual inference, and supply-chain cascade — "
                "then synthesises everything into a grounded answer.\n\n"
                "_Optionally select a scenario to ground numeric estimates._"
            )
            with gr.Row():
                causal_ask_q = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What happens to graphite prices if China restricts exports by 40%?",
                    lines=2,
                    scale=4,
                )
                causal_ask_k = gr.Number(label="Top K docs", value=5, minimum=1, maximum=20, step=1, precision=0, scale=1)
            causal_ask_scenario = gr.Dropdown(
                label="Ground with scenario (optional)",
                choices=[""] + list_scenarios(),
                value="",
                allow_custom_value=True,
            )
            causal_ask_btn = gr.Button("🔮 Analyze", variant="primary")
            causal_ask_out = gr.Markdown(label="Analysis")
            causal_ask_btn.click(
                fn=causal_ask,
                inputs=[causal_ask_q, causal_ask_scenario, causal_ask_k],
                outputs=causal_ask_out,
            )

        # --- Ask a Question (primary entry point) ---
        with gr.Tab("💬 Ask a Question"):
            gr.Markdown(
                "Ask questions in natural language.\n\n"
                "**Reason (RAG + Causal)** — retrieves relevant documents, runs causal identification "
                "on the knowledge graph, then synthesises a structured answer. Fully in-process, fast.\n\n"
                "**Run Simulation** — generates a scenario YAML from the question and runs a full supply-chain simulation."
            )
            with gr.Row():
                query_top_k = gr.Number(label="Top K docs", value=5, minimum=1, maximum=20, step=1, precision=0, scale=1)
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="What if China restricts graphite exports by 40% in 2025?",
                lines=3,
            )
            with gr.Row():
                reason_btn = gr.Button("🧠 Reason (RAG + Causal)", variant="primary", size="lg")
                query_btn = gr.Button("🚀 Run Simulation (scenario)", variant="secondary", size="lg")
            query_out = gr.Markdown(label="Results")
            reason_btn.click(fn=unified_query, inputs=[query_input, query_top_k], outputs=query_out)
            query_btn.click(fn=query_model, inputs=query_input, outputs=query_out)
            with gr.Row():
                export_btn = gr.Button("💾 Export report", variant="secondary", size="sm")
                export_file = gr.File(label="Download", visible=False)
            export_btn.click(
                fn=lambda content: (export_report(content), gr.update(visible=True)),
                inputs=[query_out],
                outputs=[export_file, export_file],
            )
            gr.Examples(
                examples=[
                    ["What if China restricts graphite exports by 40% in 2025?"],
                    ["What happens if graphite demand doubles in 2026?"],
                    ["How does China's export policy affect EV battery supply chains?"],
                ],
                inputs=query_input,
            )

        # --- Run Scenario (step 2 in flow) ---
        with gr.Tab("▶️ Run Scenario"):
            gr.Markdown(
                "Pick a scenario and click **Run** (simulation output) or "
                "**Run + Causal Analysis** (simulation + all three Pearl layers).\n\n"
                "**Run** saves outputs to `runs/<name>/<timestamp>/` and auto-fills the Validate tab.\n\n"
                "**Run + Causal Analysis** runs in-process and produces:\n"
                "- **Layer 1** — Association: correlation matrix + OLS regression\n"
                "- **Layer 2** — Intervention: do-calculus ID, simulation-based do() effect, backdoor ATE\n"
                "- **Layer 3** — Counterfactual: Pearl abduct→act→predict ('what if no shock?')\n"
                "- **Supply chain cascade** — shock propagation through the trade network"
            )
            scenario_dropdown = gr.Dropdown(
                choices=list_scenarios(),
                value=list_scenarios()[0] if list_scenarios() else None,
                label="Scenario",
                allow_custom_value=False,
            )
            with gr.Row():
                run_btn = gr.Button("▶️ Run", variant="secondary", size="lg")
                causal_run_btn = gr.Button("🔬 Run + Causal Analysis", variant="primary", size="lg")
            run_out = gr.Markdown(label="Simulation output")
            causal_run_out = gr.Markdown(label="Causal analysis (Pearl 3 layers)")
            run_btn.click(fn=run_scenario_tab, inputs=scenario_dropdown, outputs=[run_out, run_dir_in])
            causal_run_btn.click(fn=run_scenario_causal, inputs=scenario_dropdown, outputs=causal_run_out)

        # --- Search Documents (step 3 in flow) ---
        with gr.Tab("📚 Search Documents"):
            gr.Markdown(
                "**Standalone RAG** — search the corpus or ask questions with LLM synthesis. "
                "**Search** returns raw chunks. **Ask** retrieves, injects few-shot context from past answers, generates an LLM answer, and stores it in memory. "
                "Rate answers with 👍/👎 — good answers become few-shot examples for future queries. "
                "For RAG *integrated* with runs, use the **Validate with RAG** tab."
            )

            # -- Corpus management --
            gr.Markdown("#### Corpus management")
            gr.Markdown("Upload `.txt` or `.md` files, then rebuild the index to include them in search.")
            with gr.Row():
                rag_upload = gr.File(
                    label="Upload .txt or .md files",
                    file_count="multiple",
                    file_types=[".txt", ".md"],
                )
                rag_upload_out = gr.Markdown(label="Status", value="")
            with gr.Row():
                rag_save_btn = gr.Button("💾 Save to corpus", variant="secondary")
                rag_reindex_btn = gr.Button("🔄 Rebuild search index", variant="secondary")
                rag_hipporag_btn = gr.Button("🕸️ Build HippoRAG index", variant="secondary")
            rag_save_btn.click(fn=save_uploaded_documents, inputs=[rag_upload], outputs=rag_upload_out)
            rag_reindex_btn.click(fn=reindex_rag, inputs=None, outputs=rag_upload_out)
            rag_hipporag_btn.click(fn=build_hipporag_index, inputs=None, outputs=rag_upload_out)

            gr.Markdown("---")

            # -- Query controls (shared) --
            with gr.Row():
                rag_query = gr.Textbox(
                    label="Query / Question",
                    placeholder="e.g. What caused the graphite supply shock in 2023?",
                    lines=2,
                    scale=4,
                )
                rag_top_k = gr.Number(label="Top K", value=5, minimum=1, maximum=20, step=1, precision=0, scale=1)
            with gr.Row():
                rag_use_kg = gr.Checkbox(label="Include KG context (search only)", value=False)
                rag_classic_only = gr.Checkbox(label="Classic search only (disable HippoRAG)", value=False)

            with gr.Row():
                rag_btn = gr.Button("🔍 Search (chunks)", variant="secondary")
                rag_ask_btn = gr.Button("💬 Ask (LLM answer + memory)", variant="primary")

            # -- Search output --
            rag_out = gr.Markdown(label="Retrieved chunks")
            rag_btn.click(fn=rag_search, inputs=[rag_query, rag_top_k, rag_use_kg, rag_classic_only], outputs=rag_out)

            # -- Ask output + feedback --
            gr.Markdown("#### Answer")
            rag_answer_out = gr.Markdown(label="Answer", value="")
            _episode_id = gr.State("")  # hidden state holding last episode ID
            rag_ask_btn.click(fn=rag_ask, inputs=[rag_query, rag_top_k], outputs=[rag_answer_out, _episode_id])

            gr.Markdown("**Rate this answer** (affects future few-shot injection):")
            with gr.Row():
                rag_thumb_up = gr.Button("👍 Good answer", variant="secondary", size="sm")
                rag_thumb_down = gr.Button("👎 Bad answer", variant="secondary", size="sm")
            rag_feedback_out = gr.Markdown(value="")
            rag_thumb_up.click(
                fn=lambda eid: rag_feedback(eid, rating=1.0),
                inputs=[_episode_id],
                outputs=rag_feedback_out,
            )
            rag_thumb_down.click(
                fn=lambda eid: rag_feedback(eid, rating=-1.0),
                inputs=[_episode_id],
                outputs=rag_feedback_out,
            )

            # -- Memory stats + eval --
            gr.Markdown("---\n#### Memory & self-evaluation")
            gr.Markdown(
                "**Show memory stats** — current episode count, quality, boosted chunks.\n\n"
                "**Run RAG eval** — generates synthetic questions from the corpus, measures Hit@k / MRR, "
                "grades answer faithfulness, then stores good answers as few-shot examples and logs failures as knowledge gaps."
            )
            with gr.Row():
                rag_stats_btn = gr.Button("📊 Memory stats", variant="secondary", size="sm")
                rag_eval_n = gr.Number(label="# questions", value=10, minimum=3, maximum=50, step=1, precision=0)
                rag_eval_k = gr.Number(label="Top K", value=5, minimum=1, maximum=20, step=1, precision=0)
                rag_eval_btn = gr.Button("🧪 Run RAG eval + learn", variant="primary", size="sm")
            rag_stats_out = gr.Markdown(value="")
            rag_stats_btn.click(fn=rag_memory_stats, inputs=None, outputs=rag_stats_out)
            rag_eval_btn.click(fn=run_rag_eval, inputs=[rag_eval_n, rag_eval_k], outputs=rag_stats_out)

        # --- Causal Analysis & DAG ---
        with gr.Tab("🔬 Causal Analysis & DAG"):
            gr.Markdown(
                "Formal identifiability via **Pearl do-calculus**.\n\n"
                "- **Run Identifiability (scenario)**: reads the selected scenario's shocks, maps "
                "them to DAG treatment nodes, and runs `id_algorithm` + `is_identifiable` for those "
                "specific (treatment, outcome) pairs — results change with the scenario.\n"
                "- **Run Identifiability (KG DAG)**: uses the live Knowledge Graph DAG "
                "(changes as you enrich the KG).\n"
                "- **Refresh DAG image**: regenerates the image from the current KG DAG (if enriched) "
                "or falls back to the static graphite DAG."
            )
            with gr.Row():
                causal_btn = gr.Button("📊 Run Identifiability (scenario)", variant="primary", size="lg")
                causal_kg_btn = gr.Button("📊 Run Identifiability (KG DAG)", variant="secondary", size="lg")
            causal_out = gr.Markdown(label="Results")
            # Pass the scenario dropdown so identifiability queries match the selected scenario
            causal_btn.click(fn=show_causal_analysis, inputs=scenario_dropdown, outputs=causal_out)
            causal_kg_btn.click(fn=run_kg_identifiability, inputs=None, outputs=causal_out)
            gr.Markdown("---\n#### Causal DAG\n*Shows KG-derived DAG if the KG has been enriched, "
                        "otherwise static graphite supply chain DAG.*")
            causal_dag_img = gr.Image(
                value=get_dag_image_path(),
                label="Directed Acyclic Graph",
                show_label=True,
            )
            causal_dag_btn = gr.Button("🖼️ Refresh DAG image", variant="secondary")
            causal_dag_btn.click(fn=generate_dag_image, inputs=scenario_dropdown, outputs=causal_dag_img)

        # --- Validate (step 5 in flow) ---
        with gr.Tab("📋 Validate"):
            gr.Markdown("Validate a simulation **run** against historical data and RAG. Run a scenario first; the path above is auto-filled. Or type a path in the field above.")
            with gr.Row():
                year_in = gr.Number(label="Reference year (optional)", value=2008, precision=0)
            validate_btn = gr.Button("🔍 Run Validation", variant="primary")
            validate_out = gr.Markdown(label="Report")
            validate_btn.click(fn=validate_with_rag, inputs=[run_dir_in, year_in], outputs=validate_out)

        # --- Unified Workflow (full pipeline) ---
        with gr.Tab("🔄 Unified Workflow"):
            gr.Markdown("Run the **full pipeline** for mineral (Causal → Scenario → Synthetic) or sensor (POMDP + Causal). Run directory auto-fills for Validate.")
            domain_radio = gr.Radio(
                choices=["Mineral supply chain", "Sensor reliability"],
                value="Mineral supply chain",
                label="Domain",
            )
            with gr.Row():
                workflow_scenario = gr.Dropdown(
                    choices=list_scenarios(),
                    value=list_scenarios()[0] if list_scenarios() else None,
                    label="Scenario (mineral)",
                    allow_custom_value=False,
                )
                workflow_sensor_data = gr.Textbox(label="Sensor data CSV (sensor)", placeholder="default: data/sensor_test_data.csv", lines=1)
                workflow_sensor_priors = gr.Textbox(label="Priors JSON (sensor, optional)", placeholder="optional", lines=1)
            def run_unified(domain: str, scenario: str, sensor_data: str, sensor_priors: str) -> tuple[str, str]:
                if "Sensor" in (domain or ""):
                    return (run_full_pipeline_sensor(sensor_data or "", sensor_priors or ""), "")
                return run_full_pipeline_mineral(scenario or "")

            workflow_btn = gr.Button("▶️ Run full pipeline", variant="primary", size="lg")
            workflow_out = gr.Markdown(label="Pipeline output")
            workflow_btn.click(
                fn=run_unified,
                inputs=[domain_radio, workflow_scenario, workflow_sensor_data, workflow_sensor_priors],
                outputs=[workflow_out, run_dir_in],
            )

        # --- Synthetic Control ---
        with gr.Tab("📈 Synthetic Control"):
            gr.Markdown("Run synthetic control (tau_K identification) on Comtrade data. Requires `data/canonical/comtrade_graphite_trade.normalized.csv`.")
            sc_btn = gr.Button("📊 Run Synthetic Control", variant="primary")
            sc_out = gr.Markdown(label="Results")
            sc_btn.click(fn=run_synthetic_control, inputs=None, outputs=sc_out)

        # --- POMDP (Sensor Maintenance) ---
        with gr.Tab("🔧 POMDP – Sensor Maintenance"):
            gr.Markdown(
                """
                **POMDP** (Partially Observable Markov Decision Process) for sensor degradation and maintenance.
                Learns transition and emission matrices from sensor data, runs a belief-update simulation, and exports DOT graphs to `artifacts/pomdp/`.
                Uses the same **sensor domain** as **Causal Estimation** (`dag_registry/sensor_reliability.dot`): sensor data can be used both for POMDP policies and for DoWhy ATE estimation.
                """
            )
            with gr.Row():
                pomdp_data_in = gr.Textbox(
                    label="Sensor data CSV (optional)",
                    placeholder="default: data/sensor_test_data.csv",
                    lines=1,
                )
                pomdp_priors_in = gr.Textbox(
                    label="Priors JSON (optional)",
                    placeholder="e.g. configs/pomdp_priors.json",
                    lines=1,
                )
            with gr.Row():
                pomdp_btn = gr.Button("🔧 Build POMDP & Run Simulation", variant="secondary")
                pomdp_causal_btn = gr.Button("🔧 POMDP + Causal (subprocess)", variant="secondary")
                pomdp_integrated_btn = gr.Button("🧠 Integrated: Belief → Causal", variant="primary")
            pomdp_out = gr.Markdown(label="POMDP output")
            pomdp_btn.click(fn=run_pomdp_build, inputs=[pomdp_data_in, pomdp_priors_in], outputs=pomdp_out)
            pomdp_causal_btn.click(fn=run_pomdp_and_causal, inputs=[pomdp_data_in, pomdp_priors_in], outputs=pomdp_out)
            pomdp_integrated_btn.click(fn=run_pomdp_causal_integrated, inputs=[pomdp_data_in], outputs=pomdp_out)
            gr.Markdown(
                "*Integrated: fits POMDP in-process, runs 20-step belief simulation with QMDP policy, "
                "extracts terminal belief, then weights the DoWhy causal ATE by P(at_risk) — "
                "connecting uncertainty quantification to causal effect estimation.*"
            )

        # --- Knowledge Graph ---
        with gr.Tab("🕸️ Knowledge Graph"):
            gr.Markdown(
                "**Critical minerals knowledge graph**: entities, causal relations, shock propagation, and KG-derived causal DAG. Integrated with **Causal Analysis** (use \"Run Identifiability (KG DAG)\" there) and **RAG** (\"Include KG context\" in Just RAG)."
            )
            kg_build_btn = gr.Button("🔄 Build / Refresh KG", variant="primary")
            kg_summary_out = gr.Markdown(label="KG summary", value="Click **Build / Refresh KG** to load.")
            kg_shock_dropdown = gr.Dropdown(
                choices=[],
                label="Shock origin",
                allow_custom_value=True,
                value=None,
            )
            kg_build_btn.click(
                fn=kg_rebuild,
                inputs=None,
                outputs=[kg_summary_out, kg_shock_dropdown],
            )
            gr.Markdown("**Propagate shock** from an origin (e.g. policy or commodity).")
            kg_shock_btn = gr.Button("▶️ Propagate shock", variant="secondary")
            kg_shock_out = gr.Markdown(label="Shock propagation result")
            kg_shock_btn.click(fn=run_kg_shock_propagation, inputs=[kg_shock_dropdown], outputs=kg_shock_out)
            gr.Markdown("**Identifiability** using the KG-derived causal DAG (same as Causal Analysis → \"Run Identifiability (KG DAG)\").")
            kg_id_btn = gr.Button("📊 Run identifiability (KG DAG)", variant="secondary")
            kg_id_out = gr.Markdown(label="Identifiability result")
            kg_id_btn.click(fn=run_kg_identifiability, inputs=None, outputs=kg_id_out)
            gr.Markdown("**Derived causal DAG** (edges extracted from KG CAUSES relations). Use **Simplified** (recommended) to cap nodes. **Interactive** view supports smooth zoom and pan.")
            with gr.Row():
                kg_dag_btn = gr.Button("Show DAG edges (text)", variant="secondary")
                kg_viz_simplified = gr.Radio(
                    choices=[("Simplified (recommended)", True), ("Full graph", False)],
                    value="Simplified (recommended)",
                    label="DAG view",
                )
                kg_viz_btn = gr.Button("📊 Static image", variant="secondary")
                kg_interactive_btn = gr.Button("🔍 Interactive (zoom & pan)", variant="primary")
            kg_dag_out = gr.Textbox(label="KG-derived DAG edges", lines=12, interactive=False)
            kg_dag_img = gr.Image(label="KG-derived causal DAG (static)", type="filepath", height=560)
            kg_dag_html = gr.HTML(label="Interactive DAG")
            kg_dag_btn.click(fn=get_kg_dag_edges, inputs=None, outputs=kg_dag_out)
            kg_viz_btn.click(fn=get_kg_dag_image, inputs=[kg_viz_simplified], outputs=kg_dag_img)
            kg_interactive_btn.click(fn=get_kg_dag_interactive_html, inputs=[kg_viz_simplified], outputs=kg_dag_html)

            gr.Markdown(
                "---\n**Enrich KG from corpus** — retrieve documents, extract triples via LLM, "
                "merge into the live KG with confidence accumulation, and save to disk. "
                "**Single query** for targeted enrichment. "
                "**Batch (all minerals)** runs a query for all 20 critical minerals at once."
            )
            with gr.Row():
                kg_enrich_query = gr.Textbox(
                    label="Enrichment query (single)",
                    placeholder="e.g. China graphite export restrictions 2023",
                    lines=1,
                    scale=4,
                )
                kg_enrich_top_k = gr.Number(
                    label="Top K docs", value=5, minimum=1, maximum=20, step=1, precision=0, scale=1
                )
            with gr.Row():
                kg_enrich_btn = gr.Button("🔬 Enrich from query", variant="primary")
                kg_batch_top_k = gr.Number(
                    label="Top K (batch)", value=3, minimum=1, maximum=10, step=1, precision=0
                )
                kg_batch_btn = gr.Button("🌐 Batch enrich (all 20 minerals)", variant="secondary")
            kg_enrich_out = gr.Markdown(value="")
            kg_enrich_btn.click(
                fn=kg_enrich_from_corpus,
                inputs=[kg_enrich_query, kg_enrich_top_k],
                outputs=kg_enrich_out,
            )
            kg_batch_btn.click(
                fn=kg_batch_enrich,
                inputs=[kg_batch_top_k],
                outputs=kg_enrich_out,
            )

        # --- Three Layers ---
        with gr.Tab("🧬 Three Layers"):
            gr.Markdown(
                """
                ### Pearl's Ladder of Causation — Three Layers
                | Layer | Question | Quantity | Method |
                |-------|----------|----------|--------|
                | **1 — Association** | What if I *see* X? | P(Y|X) | Correlations, regression |
                | **2 — Intervention** | What if I *do* X? | P(Y|do(X)) | Do-calculus, backdoor adjustment |
                | **3 — Counterfactual** | What if X *had been* x? | P(Y_x|X',Y') | Abduction → action → prediction |

                **Layer 1** only needs data. **Layer 2** needs data + a causal DAG. **Layer 3** needs data + DAG + the structural model (a scenario run directory).
                """
            )
            with gr.Row():
                tl_run_dir = gr.Textbox(
                    label="Run directory (optional for L1/L2, required for L3)",
                    placeholder="e.g. runs/graphite_baseline/20260129_132854",
                    lines=1, scale=3,
                )
                tl_layer = gr.Radio(
                    choices=["1 — Association", "2 — Intervention", "3 — Counterfactual"],
                    value="2 — Intervention",
                    label="Layer",
                    scale=2,
                )
            with gr.Row():
                tl_treatment = gr.Textbox(
                    label="Treatment variable",
                    value="ExportPolicy",
                    lines=1, scale=1,
                )
                tl_outcome = gr.Textbox(
                    label="Outcome variable",
                    value="Price",
                    lines=1, scale=1,
                )
                tl_cf_year = gr.Textbox(
                    label="Counterfactual year (L3 only)",
                    placeholder="e.g. 2008",
                    lines=1, scale=1,
                )
                tl_cf_value = gr.Textbox(
                    label="Counterfactual value (L3 only)",
                    placeholder="e.g. 0.8",
                    lines=1, scale=1,
                )
            tl_btn = gr.Button("▶️ Run query", variant="primary")
            tl_out = gr.Markdown(label="Result")
            tl_btn.click(
                fn=three_layers_query,
                inputs=[tl_run_dir, tl_layer, tl_treatment, tl_outcome, tl_cf_year, tl_cf_value],
                outputs=tl_out,
            )
            # Wire shared run_dir_in → tl_run_dir
            run_dir_in.change(fn=lambda x: x, inputs=run_dir_in, outputs=tl_run_dir)

        # --- Causal Discovery ---
        with gr.Tab("🔍 Causal Discovery"):
            gr.Markdown(
                """
                Extract causal edges from documents with LLM + human validation.
                Run in terminal (interactive):
                ```bash
                python -m src.minerals.causal_discovery
                ```
                Output: `dag_registry/discovered_graphite_causal_structure.json`
                """
            )


if __name__ == "__main__":
    print("🚀 Starting Critical Minerals Causal Engine...")
    demo.launch(share=False, server_name="127.0.0.1", theme=gr.themes.Soft())
