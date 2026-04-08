#!/usr/bin/env python3
"""Normalize messy timeseries CSV to canonical TimeSeriesSchema format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingest.mapping import apply_mapping, infer_column_mapping
from src.llm.providers import BaseLLM, MockLLM, OptionalOpenAIProvider
from src.schemas.timeseries import TimeSeriesSchema, validate_timeseries_df


def _resolve_llm(provider: str, seed: int) -> BaseLLM:
    provider_lower = provider.lower()
    if provider_lower in {"mock", "default"}:
        return MockLLM(seed=seed)
    if provider_lower == "openai":
        return OptionalOpenAIProvider()
    raise ValueError(f"Unsupported provider: {provider}")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_timeseries(
    input_path: Path,
    output_path: Path,
    seed: int = 0,
    provider: str = "mock",
    llm: Optional[BaseLLM] = None,
) -> None:
    """
    Normalize messy timeseries CSV to canonical format using heuristics + optional LLM.
    """
    df = pd.read_csv(input_path)

    provider_instance = llm or _resolve_llm(provider, seed)
    mapping_result = infer_column_mapping(
        df_cols=list(df.columns),
        target_schema=TimeSeriesSchema,
        llm=provider_instance,
        sample_data=df.head(10) if len(df) > 0 else None,
    )

    mapped_df = apply_mapping(df, mapping_result["mapping"], target_schema=TimeSeriesSchema)

    if "year" in mapped_df.columns:
        mapped_df["year"] = _coerce_numeric(mapped_df["year"]).astype("Int64")
    for col in ["P", "D", "Q", "I"]:
        if col in mapped_df.columns:
            mapped_df[col] = _coerce_numeric(mapped_df[col])

    validate_timeseries_df(mapped_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_df.to_csv(output_path, index=False)

    mapping_path = output_path.with_suffix(".mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(
            {
                **mapping_result,
                "target_schema": "TimeSeriesSchema",
            },
            f,
            indent=2,
        )

    schema_path = output_path.with_suffix(".schema.json")
    with open(schema_path, "w") as f:
        json.dump(
            {
                "schema": "TimeSeriesSchema",
                "json_schema": TimeSeriesSchema.model_json_schema(),
                "row_count": len(mapped_df),
                "columns": list(mapped_df.columns),
            },
            f,
            indent=2,
        )

    print(f"Normalized {len(mapped_df)} rows")
    print(f"Output: {output_path}")
    print(f"Mapping: {mapping_path}")
    print(f"Schema: {schema_path}")


def main() -> int:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="Normalize messy timeseries CSV to canonical format")
    parser.add_argument("--in", dest="input_path", required=True, help="Input CSV file")
    parser.add_argument("--out", dest="output_path", required=True, help="Output CSV file")
    parser.add_argument(
        "--provider",
        dest="provider",
        default="mock",
        choices=["mock", "openai", "default"],
        help="LLM provider (default: mock, offline deterministic)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    normalize_timeseries(input_path, output_path, seed=args.seed, provider=args.provider)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

