# Dockerfile for Causal Engine FastAPI backend
# Build:  docker build -t causal-engine .
# Run:    docker run -p 8000:8000 \
#             -v $(pwd)/data:/app/data \
#             -v $(pwd)/hipporag_index:/app/hipporag_index \
#             -v $(pwd)/scenarios:/app/scenarios \
#             causal-engine

FROM python:3.12-slim

# System deps needed by igraph, scipy, and other native packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libigraph-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer-cached unless pyproject.toml changes)
COPY pyproject.toml ./
COPY README.md ./
# Create a minimal src stub so pip -e . can resolve the package without the full source
RUN mkdir -p src && touch src/__init__.py
RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache gunicorn && \
    uv pip install --system --no-cache -e ".[rag,ui]"

# HippoRAG: install the package without its transitive deps (notably vllm,
# which needs CUDA and adds ~4 GB), then add only the runtime deps it actually
# uses with the OpenAI/Claude backend. A vllm stub is wired in below so the
# `from vllm import ...` line in hipporag/llm/vllm_offline.py resolves.
# Pin hipporag to 2.0.0a3 — 2.0.0a4 added bedrock_llm.py which imports
# litellm at module load time and pulls in another ~30 transitive deps.
# 2.0.0a3 only imports openai_gpt for its LLM backend, which is the only
# path the production retriever uses.
RUN uv pip install --system --no-cache --no-deps "hipporag==2.0.0a3" && \
    uv pip install --system --no-cache --no-deps "gritlm>=1.0.2" && \
    uv pip install --system --no-cache \
        "openai>=1.58.0" \
        "python_igraph>=0.11.8" \
        "tiktoken>=0.7.0" \
        "tenacity>=8.5.0" \
        "einops>=0.7.0"

# Vendor the vllm stub. Placed at /app/vllm_stub and prepended to PYTHONPATH so
# it shadows any real vllm package — production never runs local inference.
COPY deploy/vllm_stub/ /app/vllm_stub/
ENV PYTHONPATH=/app/vllm_stub

# Copy source code (after deps so code changes don't bust the dep cache)
COPY src/ ./src/
COPY scenarios/ ./scenarios/
# Embed the document index in the image so the RAG pipeline works on first start.
# The Fly.io volume mounts at /app/data/ (starts empty); the entrypoint copies
# data_init/ → data/ on first boot so the index survives across restarts.
COPY data/documents/ ./data_init/documents/
COPY data/canonical/ ./data_init/canonical/
# Pre-rendered KG scenario PNGs (validation + predictive). Served instantly
# at /api/static/kg_scenarios/{validation,predictive}/*.png.
COPY outputs/kg_scenarios/ ./outputs/kg_scenarios/
# Per-commodity year-by-year KG PDFs (one PDF per commodity, served via
# /api/kg/commodity-pdf?commodity=X). Per-year PNGs excluded for size.
COPY ["Knowledge Graphs/", "./Knowledge Graphs/"]
# scripts/run_knowledge_graph.py is imported by /api/kg/render-scenario
COPY scripts/ ./scripts/

# Non-root user for safety
# Create cache dir before switching user so HuggingFace model downloads work
RUN useradd --no-create-home --shell /bin/false causal && \
    mkdir -p /app/.cache/huggingface && \
    chown -R causal:causal /app
USER causal

# Point all HuggingFace / sentence-transformers downloads to /app/.cache
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    MPLCONFIGDIR=/tmp/matplotlib

EXPOSE 8000

# 2 workers is a good default for a 2-core VM; override via WORKERS env var
COPY api.py ./
COPY app.py ./
COPY --chmod=755 entrypoint.sh ./

ENTRYPOINT ["./entrypoint.sh"]
CMD gunicorn api:app \
    --workers ${WORKERS:-2} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
