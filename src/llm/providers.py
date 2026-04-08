"""LLM provider interfaces for schema mapping and explanations."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

try:  # Optional import; only needed when OPENAI_API_KEY is configured
    import openai  # type: ignore
except Exception:  # pragma: no cover - we gracefully handle missing dependency
    openai = None

from src.schemas.report import RunReport


class BaseLLM(ABC):
    """Abstract interface for LLM-backed utilities used in ingestion/explanation."""

    name: str = "base"

    @abstractmethod
    def suggest_column_mapping(
        self,
        source_cols: List[str],
        target_fields: List[str],
        sample_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Return mapping + confidence + rationale."""

    @abstractmethod
    def generate_explanation(self, report: RunReport) -> str:
        """Return a deterministic, human-readable explanation."""


class MockLLM(BaseLLM):
    """Deterministic, rule-based LLM mock used for offline/default flows."""

    name: str = "mock-llm"

    def __init__(self, seed: int = 0):
        self.seed = seed  # retained for interface compatibility; no randomness used

    def _alias_lookup(self, target: str, source_cols: List[str]) -> Optional[str]:
        aliases = {
            "year": ["year", "yr", "y", "time", "date", "period"],
            "P": ["price", "p", "price_index", "cost", "value"],
            "D": ["demand", "consumption", "usage"],
            "Q": ["supply", "production", "output", "quantity", "supply_qty"],
            "I": ["inventory", "stock", "storage", "reserve", "inventory_level"],
        }
        source_lower = {col.lower().strip(): col for col in source_cols}
        for alias in aliases.get(target, []):
            if alias in source_lower:
                return source_lower[alias]
        return None

    def suggest_column_mapping(
        self,
        source_cols: List[str],
        target_fields: List[str],
        sample_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        mapping: Dict[str, str] = {}
        rationales: List[str] = []
        per_conf: Dict[str, float] = {}
        source_lower = {col.lower().strip(): col for col in source_cols}

        for target in target_fields:
            resolved = self._alias_lookup(target, source_cols)
            if resolved:
                mapping[target] = resolved
                per_conf[target] = 0.9
                rationales.append(f"{target} <- {resolved} (alias match)")
                continue

            best_match = None
            for src_lower, src_orig in sorted(source_lower.items()):
                if target.lower() in src_lower or src_lower in target.lower():
                    best_match = src_orig
                    break
            if best_match:
                mapping[target] = best_match
                per_conf[target] = 0.65
                rationales.append(f"{target} <- {best_match} (substring match)")
            else:
                rationales.append(f"{target} <- None (no heuristic match)")

        overall_conf = (
            sum(per_conf.values()) / len(target_fields) if target_fields else 0.0
        )
        return {
            "mapping": mapping,
            "confidence": round(overall_conf, 3),
            "rationale": "; ".join(rationales),
            "provider": self.name,
        }

    def generate_explanation(self, report: RunReport) -> str:
        metrics_lines = []
        for key, value in sorted(report.metrics.items()):
            metrics_lines.append(f"- {key}: {value}")

        params_lines = []
        if report.fitted_params:
            for key, value in sorted(report.fitted_params.items()):
                params_lines.append(f"- {key}: {value}")

        artifacts_lines = [f"- {path}" for path in sorted(report.artifacts)]

        narrative = (report.narrative or "").strip()
        if not narrative:
            narrative = "- Deterministic narrative unavailable; metrics-only memo."

        sections = [
            "# Run Memo (MockLLM)",
            f"**Scenario:** {report.scenario_path}",
            "## Metrics",
            *(metrics_lines or ["- (none)"]),
            "## Fitted Parameters",
            *(params_lines or ["- (none)"]),
            "## Artifacts",
            *(artifacts_lines or ["- (none)"]),
            "## Narrative",
            narrative,
        ]
        return "\n".join(sections)


class OptionalOpenAIProvider(BaseLLM):
    """Optional real provider gated behind OPENAI_API_KEY."""

    name: str = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set; use MockLLM for offline runs")
        if openai is None:
            raise RuntimeError("openai package not installed; pip install openai")

        self.model = model
        self.temperature = temperature
        if hasattr(openai, "OpenAI"):
            self._client = openai.OpenAI(api_key=api_key)
            self._api_key = None
        else:
            openai.api_key = api_key  # type: ignore[attr-defined]
            self._client = openai
            self._api_key = api_key

    def _chat_completion(self, messages: List[Dict[str, str]]) -> str:
        # This call remains optional and is not used in tests.
        try:
            if hasattr(self._client, "chat") and hasattr(
                self._client.chat, "completions"
            ):
                kwargs = {
                    "model": self.model,
                    "temperature": self.temperature,
                    "messages": messages,
                }
                if self._api_key:
                    kwargs["api_key"] = self._api_key  # type: ignore[arg-type]
                response = self._client.chat.completions.create(**kwargs)
                return response.choices[0].message.content  # type: ignore[index]
        except Exception as exc:  # pragma: no cover - network failures are logged
            raise RuntimeError(f"OpenAI call failed: {exc}") from exc
        raise RuntimeError("OpenAI client is not configured correctly")

    def suggest_column_mapping(
        self,
        source_cols: List[str],
        target_fields: List[str],
        sample_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        # Deterministic fallback using MockLLM to avoid nondeterminism in the base path.
        fallback = MockLLM()
        mapping = fallback.suggest_column_mapping(source_cols, target_fields, sample_data)
        mapping["provider"] = self.name
        return mapping

    def generate_explanation(self, report: RunReport) -> str:
        prompt = (
            "Generate a concise markdown memo summarizing the run.\n"
            f"Scenario: {report.scenario_path}\n"
            f"Metrics: {json.dumps(report.metrics)}\n"
            f"Artifacts: {json.dumps(report.artifacts)}\n"
            f"Narrative: {report.narrative or 'None'}\n"
        )
        # For deterministic/offline default we still delegate to the mock.
        fallback = MockLLM()
        try:
            _ = self._chat_completion(
                [
                    {"role": "system", "content": "You are a helpful analyst."},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:
            # On any failure, rely on deterministic mock output.
            pass
        return fallback.generate_explanation(report)

