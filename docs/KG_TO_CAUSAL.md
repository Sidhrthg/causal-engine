# Knowledge Graph → Causal Model: How It Works and Where LLM / vLLM / HippoRAG Fit

## The pipeline you described

**LLM + Knowledge Graph → [Translate KG to Causal Model] → Use Causal Model for Analysis**

Here’s how that is implemented today and how vLLM and HippoRAG can be used.

---

## 1. Translating Knowledge Graph to Causal Model (already in place)

The “translate KG → causal model” step is **rule-based** in code; no LLM is required for it.

- **Where:** `src/minerals/knowledge_graph.py` → `CausalKnowledgeGraph.to_causal_dag()`  
- **What it does:** Takes the KG and keeps only **CAUSES** relationships. It builds a **CausalDAG** (nodes = entities, directed edges = cause → effect) that the rest of the stack uses for:
  - identifiability (do-calculus),
  - adjustment sets,
  - simulation,
  - and any analysis that needs a causal DAG.

So: **KG (with CAUSES edges) → `kg.to_causal_dag()` → Causal Model (CausalDAG) → Analysis.**  
That translation is deterministic and already wired in the app and scripts.

---

## 2. Where the LLM fits: *building* or *refining* the structure

The LLM is not used to “run” the translation (KG → DAG). It is used to **discover or refine** the causal structure that later gets turned into a DAG (and optionally merged into the KG).

### Path A: Documents → LLM → Causal edges → DAG (and optionally KG)

- **Causal Discovery** (`src/minerals/causal_discovery.py`):
  1. **Retrieve** documents (RAG or HippoRAG).
  2. **LLM** (Claude, or vLLM if wired) extracts causal claims from text → list of (cause, effect, mechanism, evidence).
  3. Optional **human validation** of edges.
  4. **Export** to `dag_registry/discovered_*.json`.

- **Connecting to the KG:**  
  The KG can **import** that discovered DAG via `kg.import_from_dag_registry(path)`. After import, those edges become CAUSES in the KG, and **the same translation** applies: `kg.to_causal_dag()` produces the causal model used for analysis.

So the full flow is:

- **Documents → (HippoRAG or RAG) → LLM (vLLM/Claude) → causal edges → dag_registry JSON**  
- **Then either:**  
  - use that JSON directly as a DAG, or  
  - **KG.import_from_dag_registry(that JSON) → KG → to_causal_dag() → Causal Model → Analysis**

### Path B (future): KG → LLM → refined DAG

You could add a step: **KG (rich graph) → LLM → “which relations are causal for this analysis?” → refined DAG.**  
That would be a second, optional way to “translate” KG to causal model (LLM-assisted). Not implemented yet; the current translation is the rule-based `to_causal_dag()`.

---

## 3. Using vLLM in this pipeline

- **Causal Discovery** uses the **unified chat backend** (`src/llm/chat.py` → `chat_completion()`), so:
  - **LLM_BACKEND=vllm** (with a running vLLM server) uses vLLM for extracting causal edges from documents.
  - **LLM_BACKEND=anthropic**, **openai**, or **hybrid** use Claude, OpenAI, or vLLM-then-Claude.

So: **vLLM** fits in the “Documents → LLM → causal edges” part; the “KG → Causal Model” translation remains `to_causal_dag()`.

---

## 4. Using HippoRAG in this pipeline

- **HippoRAG** is used for **retrieval** over your document corpus (graph-based retrieval instead of plain semantic/keyword search).
- **Causal Discovery** uses `get_retriever(use_hipporag=True)` by default (set `USE_HIPPORAG=0` to force classic RAG). So:
  - **Documents → HippoRAG retrieval (when index exists) → LLM (vLLM or Claude) extracts causal edges → export to dag_registry (and optionally KG).**

So: **HippoRAG** improves *which* text the LLM sees when discovering causal structure; it does not replace the KG → causal model translation.

---

## 5. Summary

| Step | What happens | vLLM / HippoRAG |
|------|----------------|------------------|
| **Translate KG → Causal Model** | `kg.to_causal_dag()` (CAUSES → CausalDAG) | Not involved; rule-based. |
| **Get causal structure from text** | Causal Discovery: retrieve docs → LLM extracts edges → export (and optionally import into KG) | **vLLM**: as LLM backend for extraction. **HippoRAG**: as retriever for docs. |
| **Use causal model for analysis** | Identifiability, simulation, etc. on the CausalDAG | Not involved. |

So: **“Translate Knowledge Graph to Causal Model”** is already implemented (`to_causal_dag()`). **vLLM** and **HippoRAG** help in the **LLM + documents** part that *feeds* the KG or the DAG registry; they don’t replace the KG→causal translation.
