"""Explanation helpers built on top of the LLM providers."""

from __future__ import annotations

from typing import List

from src.llm.providers import BaseLLM, MockLLM
from src.schemas.report import RunReport


def _metric_value(report: RunReport, key: str) -> float:
    return float(report.metrics.get(key, 0.0))


def _interpretations(report: RunReport) -> List[str]:
    interpretations: List[str] = []
    if _metric_value(report, "shock_year_shortage") > 0:
        interpretations.append("shock-year disruption persists in the first year.")
    if _metric_value(report, "post_shock_total_shortage") > 0:
        interpretations.append("lingering effects continue after the initial shock.")
    avg_price = _metric_value(report, "avg_price")
    if avg_price < 1:
        interpretations.append("price suppression / oversupply or policy lowers price.")
    if avg_price > 1:
        interpretations.append("tight market with elevated pricing pressure.")
    final_inventory_cover = _metric_value(report, "final_inventory_cover")
    if final_inventory_cover < 0.2:
        interpretations.append("thin inventories leave little buffer.")
    elif final_inventory_cover > 0.2:
        interpretations.append("healthier buffer remains by the end of horizon.")
    return interpretations


def _build_deterministic_narrative(report: RunReport) -> str:
    """Construct a deterministic, metrics-grounded narrative with required fields."""
    r = report
    bullets = [
        f"- Scenario: {r.scenario_path}",
        f"- shock_year_shortage: {_metric_value(r, 'shock_year_shortage')}",
        f"- shock_window_total_shortage: {_metric_value(r, 'shock_window_total_shortage')}",
        f"- post_shock_total_shortage: {_metric_value(r, 'post_shock_total_shortage')}",
        f"- total_shortage: {_metric_value(r, 'total_shortage')}",
        f"- avg_price: {_metric_value(r, 'avg_price')}",
        f"- final_inventory_cover: {_metric_value(r, 'final_inventory_cover')}",
    ]

    interpretations = _interpretations(r)
    if interpretations:
        bullets.extend([f"- Interpretation: {txt}" for txt in interpretations])

    # Ensure 6–10 bullets: add a simple stabilizer bullet if needed.
    if len(bullets) < 6:
        bullets.append("- Market signal: mixed")
    if len(bullets) > 10:
        bullets = bullets[:10]

    caveats = [
        "Caveats:",
        "- Deterministic template narrative; values reflect provided metrics.",
        "- No external LLM calls were made.",
    ]
    return "\n".join(bullets + caveats)


def explain_run(report: RunReport, llm: BaseLLM | None = None) -> str:
    """
    Generate a markdown explanation for a run report.

    Always injects a deterministic narrative to guarantee non-empty output.
    """
    validated_report = RunReport.model_validate(report)
    narrative = _build_deterministic_narrative(validated_report)
    enriched_report = validated_report.model_copy(update={"narrative": narrative})

    provider = llm or MockLLM()
    rendered = provider.generate_explanation(enriched_report)

    if not rendered or not rendered.strip():
        # Fallback to deterministic narrative if provider yields empty output.
        rendered = "\n".join(
            [
                "# Run Memo (deterministic fallback)",
                f"**Scenario:** {enriched_report.scenario_path}",
                "## Narrative",
                narrative,
            ]
        )
    return rendered

