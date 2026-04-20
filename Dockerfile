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

# Copy source code (after deps so code changes don't bust the dep cache)
COPY src/ ./src/
COPY scenarios/ ./scenarios/
# Embed the document index in the image so the RAG pipeline works on first start.
# The Fly.io volume mounts at /app/data/ (starts empty); the entrypoint copies
# data_init/ → data/ on first boot so the index survives across restarts.
COPY data/documents/ ./data_init/documents/
# data/canonical/ (CEPII CSVs) are large and gitignored — mount via Fly.io volume

# Non-root user for safety
# Create cache dir before switching user so HuggingFace model downloads work
RUN useradd --no-create-home --shell /bin/false causal && \
    mkdir -p /app/.cache/huggingface && \
    chown -R causal:causal /app
USER causal

# Point all HuggingFace / sentence-transformers downloads to /app/.cache
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

EXPOSE 8000

# 2 workers is a good default for a 2-core VM; override via WORKERS env var
COPY api.py ./
COPY app.py ./
COPY configs/ ./configs/
COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD gunicorn api:app \
    --workers ${WORKERS:-2} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
