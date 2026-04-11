"""
Unified chat completion for Query Model, Validate with RAG, and other LLM flows.

Supports backends: anthropic (Claude), openai (OpenAI API), vllm (local vLLM server),
and hybrid (try vLLM first, fall back to Claude). Choose via LLM_BACKEND.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

# Load .env from project root so ANTHROPIC_API_KEY etc. are set when this module is imported
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Backend selection from environment
# ---------------------------------------------------------------------------

def _backend() -> str:
    return (os.getenv("LLM_BACKEND") or "anthropic").strip().lower()


def _chat_anthropic(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError:
        raise RuntimeError("anthropic package required for LLM_BACKEND=anthropic. pip install anthropic")
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=key)
    model = model or "claude-sonnet-4-20250514"
    # Anthropic accepts list of role/content; filter to user/assistant only
    anthropic_messages = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")]
    if not any(m["role"] == "user" for m in anthropic_messages):
        raise ValueError("No user message in messages")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=anthropic_messages,
    )
    return response.content[0].text


def _chat_openai_compatible(
    messages: List[dict],
    model: Optional[str],
    max_tokens: int,
    base_url: str,
    api_key: str,
) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package required for vLLM/OpenAI. pip install openai")
    client = OpenAI(base_url=base_url, api_key=api_key or "dummy")
    response = client.chat.completions.create(
        model=model or "default",
        messages=messages,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _chat_vllm(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
) -> str:
    base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1" if not base_url.endswith("/v1") else base_url
    model = model or os.getenv("VLLM_MODEL", "default")
    key = api_key or os.getenv("VLLM_API_KEY", "dummy")
    return _chat_openai_compatible(messages, model, max_tokens, base_url, key)


def _chat_openai(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set for LLM_BACKEND=openai")
    return _chat_openai_compatible(
        messages,
        model or "gpt-4o-mini",
        max_tokens,
        base_url="https://api.openai.com/v1",
        api_key=key,
    )


def _chat_hybrid(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
) -> str:
    """Try vLLM first; on failure or if vLLM not configured, use Claude."""
    if is_chat_available("vllm"):
        try:
            return _chat_vllm(messages, model=model, max_tokens=max_tokens, api_key=api_key)
        except Exception:
            pass
    return _chat_anthropic(messages, model=model, max_tokens=max_tokens, api_key=api_key)


def chat_completion(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 3000,
    api_key: Optional[str] = None,
    backend: Optional[str] = None,
) -> str:
    """
    Single entry point for chat completion across backends.

    Args:
        messages: List of {"role": "user"|"assistant", "content": "..."}.
        model: Override model name (backend-specific).
        max_tokens: Max tokens to generate.
        api_key: Override API key (else from env).
        backend: Override backend (else from LLM_BACKEND: anthropic | openai | vllm | hybrid).

    Returns:
        Assistant reply text.

    Environment:
        LLM_BACKEND: anthropic | openai | vllm | hybrid (default: anthropic)
        hybrid: try vLLM first, fall back to Claude. Needs ANTHROPIC_API_KEY; optional VLLM_* for vLLM.
    """
    backend = (backend or _backend()).strip().lower()
    if backend == "anthropic":
        return _chat_anthropic(messages, model=model, max_tokens=max_tokens, api_key=api_key)
    if backend == "vllm":
        return _chat_vllm(messages, model=model, max_tokens=max_tokens, api_key=api_key)
    if backend == "openai":
        return _chat_openai(messages, model=model, max_tokens=max_tokens, api_key=api_key)
    if backend == "hybrid":
        return _chat_hybrid(messages, model=model, max_tokens=max_tokens, api_key=api_key)
    raise ValueError(f"Unknown LLM_BACKEND={backend}. Use anthropic, openai, vllm, or hybrid.")


def is_chat_available(backend: Optional[str] = None) -> bool:
    """Return True if the selected backend is configured and importable."""
    backend = (backend or _backend()).strip().lower()
    if backend == "anthropic":
        try:
            from anthropic import Anthropic
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            return False
    if backend == "vllm":
        try:
            from openai import OpenAI
            return True  # Best effort; server might still be down
        except ImportError:
            return False
    if backend == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if backend == "hybrid":
        # Hybrid needs at least Claude (fallback)
        try:
            from anthropic import Anthropic
            return bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            return False
    return False
