"""
Minimal vllm stub for the production Docker image.

HippoRAG hard-pins ``vllm==0.6.6.post1`` as a transitive dependency, but the
real vllm package needs a CUDA GPU and balloons the image by ~4 GB. The Causal
Engine production deployment only ever uses HippoRAG with the OpenAI/Claude
backend, so vllm is never actually called at runtime — the package only needs
to satisfy the ``from vllm import SamplingParams, LLM`` import at the top of
``hipporag/llm/vllm_offline.py``.

This stub is added to ``PYTHONPATH`` ahead of site-packages so it shadows any
real vllm. If anything tries to construct a SamplingParams or LLM instance we
raise loudly, since reaching that path means the deployment was misconfigured
to attempt local inference.
"""


class SamplingParams:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "vllm is stubbed in this image; local inference is not supported. "
            "Use the OpenAI/Claude HippoRAG backend (default in production)."
        )


class LLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "vllm is stubbed in this image; local inference is not supported. "
            "Use the OpenAI/Claude HippoRAG backend (default in production)."
        )


__version__ = "0.6.6.stub"
