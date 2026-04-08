"""LLM-assisted (deterministic) column mapping for messy input data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.llm.providers import BaseLLM, MockLLM
from src.schemas.timeseries import TimeSeriesSchema

# Canonical aliases used for heuristic mapping before invoking LLMs
CANONICAL_ALIASES: Dict[str, List[str]] = {
    "year": ["year", "yr", "y", "time", "date", "period"],
    "P": ["price", "p", "price_index", "cost", "value"],
    "D": ["demand", "consumption", "usage"],
    "Q": ["supply", "production", "output", "quantity", "supply_qty"],
    "I": ["inventory", "stock", "storage", "reserve", "inventory_level"],
}


def _target_fields(target_schema: Any) -> List[str]:
    if isinstance(target_schema, Sequence) and not isinstance(target_schema, str):
        return list(target_schema)
    if hasattr(target_schema, "model_fields"):
        return list(target_schema.model_fields.keys())  # type: ignore[attr-defined]
    raise ValueError("target_schema must be a Pydantic model or an iterable of fields")


def _heuristic_mapping(df_cols: List[str], target_fields: List[str]) -> Dict[str, Any]:
    source_lower = {col.lower().strip(): col for col in df_cols}
    mapping: Dict[str, str] = {}
    rationale: List[str] = []
    confidence: Dict[str, float] = {}
    ambiguous: List[str] = []
    missing: List[str] = []

    for target in target_fields:
        aliases = CANONICAL_ALIASES.get(target, [])
        candidates = [source_lower[a] for a in aliases if a in source_lower]

        if len(candidates) == 1:
            mapping[target] = candidates[0]
            confidence[target] = 0.95
            rationale.append(f"{target} <- {candidates[0]} (alias match)")
            continue
        if len(candidates) > 1:
            ambiguous.append(target)
            rationale.append(f"{target} ambiguous aliases: {', '.join(candidates)}")
            continue

        # Substring heuristic
        substr_candidates = [
            orig
            for lower, orig in sorted(source_lower.items())
            if target.lower() in lower or lower in target.lower()
        ]
        if len(substr_candidates) == 1:
            mapping[target] = substr_candidates[0]
            confidence[target] = 0.7
            rationale.append(f"{target} <- {substr_candidates[0]} (substring match)")
        elif len(substr_candidates) > 1:
            ambiguous.append(target)
            rationale.append(f"{target} ambiguous substrings: {', '.join(substr_candidates)}")
        else:
            missing.append(target)
            rationale.append(f"{target} <- None (no heuristic match)")

    overall = sum(confidence.values()) / len(target_fields) if target_fields else 0.0
    return {
        "mapping": mapping,
        "confidence": round(overall, 3),
        "rationale": rationale,
        "ambiguous": ambiguous,
        "missing": missing,
        "provider": "heuristic",
    }


def infer_column_mapping(
    df_cols: List[str],
    target_schema: Any,
    llm: Optional[BaseLLM] = None,
    sample_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Infer column mapping from source columns to target schema using heuristics first,
    then LLM (deterministic mock by default) only when ambiguous or missing.
    """
    targets = _target_fields(target_schema)
    heuristic = _heuristic_mapping(df_cols, targets)

    needs_llm = bool(heuristic["ambiguous"] or heuristic["missing"])
    provider_used = heuristic["provider"]
    mapping = dict(heuristic["mapping"])
    rationale = list(heuristic["rationale"])
    confidence = heuristic["confidence"]

    if needs_llm:
        provider = llm or MockLLM()
        provider_used = provider.name
        llm_result = provider.suggest_column_mapping(df_cols, targets, sample_data)
        # Only override ambiguous/missing targets to keep deterministic heuristics
        for target in heuristic["ambiguous"] + heuristic["missing"]:
            if target in llm_result.get("mapping", {}):
                mapping[target] = llm_result["mapping"][target]
        rationale.append(f"LLM ({provider.name}) consulted for unresolved fields")
        # Blend confidence deterministically: mean of heuristic + llm confidence
        confidence = round(
            (confidence + float(llm_result.get("confidence", 0.0))) / 2, 3
        )

    return {
        "mapping": mapping,
        "confidence": confidence,
        "rationale": "; ".join(rationale),
        "provider": provider_used,
        "ambiguous": heuristic["ambiguous"],
        "missing": heuristic["missing"],
    }


def apply_mapping(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    target_schema: Any = TimeSeriesSchema,
) -> pd.DataFrame:
    """
    Apply a target->source mapping to produce a canonical DataFrame.

    Missing targets are filled with pd.NA to keep the schema explicit.
    """
    targets = _target_fields(target_schema)
    result = pd.DataFrame()
    for target in targets:
        source = mapping.get(target)
        if source and source in df.columns:
            result[target] = df[source]
        else:
            result[target] = pd.NA
    return result

