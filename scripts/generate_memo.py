#!/usr/bin/env python3
"""Generate a human-readable memo for a run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.llm.explain import explain_run
from src.llm.providers import BaseLLM, MockLLM, OptionalOpenAIProvider
from src.schemas.report import RunReport


def _resolve_llm(provider: str) -> BaseLLM:
    if provider.lower() in {"mock", "default"}:
        return MockLLM()
    if provider.lower() == "openai":
        return OptionalOpenAIProvider()
    raise ValueError(f"Unsupported provider: {provider}")


def _load_run_report(run_dir: Path) -> RunReport:
    report_path = run_dir / "report.json"
    metrics_path = run_dir / "metrics.json"

    if report_path.exists():
        data = json.loads(report_path.read_text())
    else:
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        artifacts = [str(p) for p in sorted(run_dir.glob("*")) if p.is_file()]
        data = {
            "scenario_path": str(run_dir.name),
            "fitted_params": None,
            "metrics": metrics,
            "artifacts": artifacts,
            "narrative": None,
        }

    return RunReport.model_validate(data)


def generate_memo(
    run_dir: Path,
    output_path: Optional[Path] = None,
    provider: str = "mock",
    llm: Optional[BaseLLM] = None,
) -> Path:
    run_dir = Path(run_dir)
    output_path = output_path or (run_dir / "memo.md")

    report = _load_run_report(run_dir)
    llm_instance = llm or _resolve_llm(provider)
    memo_md = explain_run(report, llm=llm_instance)

    output_path.write_text(memo_md)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate memo for a run directory")
    parser.add_argument("--run-dir", required=True, help="Path to run directory")
    parser.add_argument("--out", dest="output_path", required=False, help="Output memo path")
    parser.add_argument(
        "--provider",
        dest="provider",
        default="mock",
        choices=["mock", "openai", "default"],
        help="LLM provider (default: mock, offline deterministic)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_path = Path(args.output_path) if args.output_path else None
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    memo_path = generate_memo(run_dir, output_path=output_path, provider=args.provider)
    print(f"Memo written to {memo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

