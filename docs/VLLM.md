# vLLM Integration

The app can use a **local vLLM server** as the LLM backend for:

- **Query Model** (natural language → scenario YAML, then run simulation)
- **Validate with RAG** (LLM analysis of run vs historical data + documents)

vLLM exposes an **OpenAI-compatible** HTTP API, so the same code path works with OpenAI or vLLM by switching `LLM_BACKEND`.

---

## 1. Start a vLLM server

Install vLLM (separate env recommended):

```bash
pip install vllm
```

Serve a model (example: Llama 3 8B on port 8000):

```bash
vllm serve meta-llama/Llama-3-8B-Instruct \
  --dtype auto \
  --port 8000
```

Or without GPU, a smaller model:

```bash
vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto --port 8000
```

By default the server listens on `http://localhost:8000` and exposes the OpenAI API under `/v1` (e.g. `http://localhost:8000/v1/chat/completions`).

---

## 2. Configure the app

Set environment variables **before** starting the Gradio app or running scripts:

```bash
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_MODEL=meta-llama/Llama-3-8B-Instruct
```

- **`LLM_BACKEND`**: `anthropic` (default) | `openai` | `vllm` | `hybrid`
- **`VLLM_BASE_URL`**: Base URL of the vLLM server (default `http://localhost:8000/v1`)
- **`VLLM_MODEL`**: Model name as served (must match the model you passed to `vllm serve`). If the server only has one model, some setups accept a generic name; otherwise use the exact name.
- **`VLLM_API_KEY`**: Optional; vLLM can be run with `--api-key token-xyz` and then set `VLLM_API_KEY=token-xyz`.

Then run the app:

```bash
python app.py
```

Use **Query Model** or **Validate with RAG** as usual; they will call your local vLLM server instead of Anthropic.

---

## 3. Backend summary

| Backend   | Env / config | Use case |
|----------|--------------|----------|
| `anthropic` | `ANTHROPIC_API_KEY`, `LLM_BACKEND=anthropic` (default) | Claude API |
| `openai`    | `OPENAI_API_KEY`, `LLM_BACKEND=openai` | OpenAI API (e.g. GPT-4) |
| `vllm`      | `LLM_BACKEND=vllm`, `VLLM_BASE_URL`, `VLLM_MODEL` | Local vLLM server (OpenAI-compatible) |
| `hybrid`    | `LLM_BACKEND=hybrid`, `ANTHROPIC_API_KEY` (required), optional `VLLM_*` | Try vLLM first; on failure or if vLLM not configured, use Claude |

**Hybrid:** Use `LLM_BACKEND=hybrid` when you want to prefer local vLLM but still have Claude as fallback (e.g. vLLM server down or not running). You must set `ANTHROPIC_API_KEY`; vLLM is optional.

Implementation: `src/llm/chat.py` (`chat_completion`, `is_chat_available`). Scripts that need an LLM use this module so one env switch changes the backend everywhere.

---

## 4. Troubleshooting

### "Failed to infer device type" on macOS

The default `pip install vllm` build expects a **CUDA GPU** (NVIDIA). On a Mac there is no CUDA, so vLLM raises:

```text
RuntimeError: Failed to infer device type
```

**Options on Apple Silicon / macOS:**

1. **Run vLLM elsewhere** (simplest): Run the vLLM server on a Linux machine or cloud instance with an NVIDIA GPU (or use a CPU build on Linux). In your app, set `VLLM_BASE_URL` to that server (e.g. `http://your-server:8000/v1`). No vLLM install needed on your Mac; the causal-engine app only needs the `openai` package to call the HTTP API.

2. **vLLM CPU on Mac (build from source):** vLLM has experimental Apple Silicon CPU support but **no pre-built wheels**. You must use a **separate virtualenv** (to avoid the numpy conflict below) and build from source. Example (requires Xcode Command Line Tools, macOS Sonoma+):
   ```bash
   python3.12 -m venv .venv_vllm && source .venv_vllm/bin/activate
   git clone https://github.com/vllm-project/vllm.git vllm_source && cd vllm_source
   uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
   uv pip install -e .
   vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dtype auto --port 8000
   ```
   Then run the causal-engine app in your **main** venv with `LLM_BACKEND=vllm` and `VLLM_BASE_URL=http://localhost:8000/v1`.

3. **vLLM-Metal (Apple GPU):** For Metal-accelerated inference on Apple Silicon, see the community plugin [vllm-metal](https://github.com/vllm-project/vllm-metal) (MLX backend). Use a separate env to avoid conflicts with causal-engine.

### Numpy conflict (vLLM vs dowhy/cvxpy)

- **causal-engine** (DoWhy, cvxpy) needs **numpy ≥ 2.0**.
- **vLLM 0.6.x** from PyPI declares **numpy &lt; 2.0**.

So in a **single** venv you cannot satisfy both. Recommended:

- **Keep numpy ≥ 2** in your main causal-engine venv. Do **not** install vLLM in that venv.
- **Option A:** Run the vLLM **server** in a **separate venv** (or Docker / another machine) and point the app at it with `VLLM_BASE_URL`. The app only needs the `openai` HTTP client; it does not need the `vllm` package.
- **Option B:** Use **Claude or OpenAI** in the main venv (`LLM_BACKEND=anthropic` or `openai`) and skip local vLLM.

If you already downgraded to numpy &lt; 2 for vLLM, restore numpy 2.x for causal-engine:

```bash
pip install "numpy>=2.0,<2.4"
```

You may see a pip dependency conflict warning about vllm; that’s expected if vLLM is in the same env. Running vLLM in a separate env or remotely avoids the conflict.
