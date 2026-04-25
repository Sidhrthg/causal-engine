"""
Precision/recall evaluation of the EventShockMapper extraction pipeline.

Gold standard: 20 hand-labeled news headlines with ground-truth shock annotations.
Each annotation specifies: commodity, shock_type, direction (positive/negative).

Metrics:
  - Commodity-level recall:    fraction of expected (commodity) hits extracted
  - Type-level precision:      fraction of extracted (commodity, type) pairs that are correct
  - Type-level recall:         fraction of expected (commodity, type) pairs that are extracted
  - Direction accuracy:        among type-correct extractions, fraction with correct sign
  - F1 (commodity):            harmonic mean of commodity precision/recall
  - F1 (type):                 harmonic mean of type precision/recall
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .event_shock_mapper import EventShockMapper


# ── Gold standard dataset ──────────────────────────────────────────────────────
#
# Each entry:  text, expected list of {commodity, type, direction}
# direction:   "positive" (price-raising) or "negative" (price-reducing)
# Notes are not used for scoring — they document the real-world event.

GOLD_STANDARD: List[Dict[str, Any]] = [
    # ── Graphite ──────────────────────────────────────────────────────────────
    {
        "id": "G01",
        "note": "China Oct-2023 graphite export ban",
        "text": (
            "China imposed export controls on natural and synthetic graphite in October 2023, "
            "requiring companies to obtain licences before shipping abroad."
        ),
        "expected": [
            {"commodity": "graphite", "type": "export_restriction", "direction": "positive"},
        ],
    },
    {
        "id": "G02",
        "note": "China 2010-11 graphite export quota",
        "text": (
            "China reduced its graphite export quota by 40% in 2010, restricting global supply "
            "and driving prices sharply higher through 2011."
        ),
        "expected": [
            {"commodity": "graphite", "type": "export_restriction", "direction": "positive"},
        ],
    },
    {
        "id": "G03",
        "note": "EV demand surge for graphite anode 2022",
        "text": (
            "Electric vehicle sales boomed in 2022, triggering a surge in global demand for "
            "graphite anodes used in lithium-ion batteries."
        ),
        "expected": [
            {"commodity": "graphite", "type": "demand_surge", "direction": "positive"},
        ],
    },
    {
        "id": "G04",
        "note": "Graphite price collapse post-EV slowdown 2024",
        "text": (
            "Graphite prices collapsed in 2024 as EV demand growth slowed and Chinese "
            "oversupply caused a crash in anode material prices."
        ),
        "expected": [
            {"commodity": "graphite", "type": "demand_surge", "direction": "negative"},
        ],
    },
    # ── Cobalt ────────────────────────────────────────────────────────────────
    {
        "id": "C01",
        "note": "DRC Mutanda mine shutdown 2019",
        "text": (
            "Glencore announced the closure of its Mutanda copper-cobalt mine in the DRC in 2019, "
            "causing a significant shutdown in global cobalt output."
        ),
        "expected": [
            {"commodity": "cobalt", "type": "capex_shock", "direction": "positive"},
        ],
    },
    {
        "id": "C02",
        "note": "Congo cobalt export licence suspension",
        "text": (
            "The Democratic Republic of Congo suspended cobalt export licences for artisanal miners "
            "following a government review, restricting shipments to China."
        ),
        "expected": [
            {"commodity": "cobalt", "type": "export_restriction", "direction": "positive"},
        ],
    },
    {
        "id": "C03",
        "note": "Cobalt demand crash from LFP adoption 2023",
        "text": (
            "Cobalt prices dropped sharply in 2023 as automakers shifted to lithium iron phosphate "
            "batteries, causing a steep decline in cobalt demand and a price crash."
        ),
        "expected": [
            {"commodity": "cobalt", "type": "demand_surge", "direction": "negative"},
        ],
    },
    {
        "id": "C04",
        "note": "EV battery boom drives cobalt demand spike 2017",
        "text": (
            "A boom in EV investment in 2017 sent cobalt prices spiking as battery manufacturers "
            "raced to secure supply, triggering a demand surge."
        ),
        "expected": [
            {"commodity": "cobalt", "type": "demand_surge", "direction": "positive"},
        ],
    },
    # ── Lithium ───────────────────────────────────────────────────────────────
    {
        "id": "L01",
        "note": "Chile lithium boom 2022",
        "text": (
            "Lithium prices surged more than 400% in 2022, driven by a boom in electric vehicle "
            "production and a spike in demand from battery manufacturers in China and South Korea."
        ),
        "expected": [
            {"commodity": "lithium", "type": "demand_surge", "direction": "positive"},
        ],
    },
    {
        "id": "L02",
        "note": "Chile lithium price crash 2023-24",
        "text": (
            "Chilean lithium exports declined sharply in 2024 as global prices crashed, with "
            "spot prices dropping more than 80% from their 2022 peak amid a supply glut."
        ),
        "expected": [
            {"commodity": "lithium", "type": "demand_surge", "direction": "negative"},
        ],
    },
    {
        "id": "L03",
        "note": "Chile nationalization policy 2023",
        "text": (
            "Chile announced a policy to nationalise its lithium industry in 2023, with the "
            "government requiring state participation in all new lithium contracts."
        ),
        "expected": [
            {"commodity": "lithium", "type": "export_restriction", "direction": "positive"},
        ],
    },
    # ── Nickel ────────────────────────────────────────────────────────────────
    {
        "id": "N01",
        "note": "Indonesia nickel ore export ban 2020",
        "text": (
            "Indonesia banned exports of nickel ore in January 2020, forcing processors to "
            "smelt locally and restricting global nickel supply from the world's top producer."
        ),
        "expected": [
            {"commodity": "nickel", "type": "export_restriction", "direction": "positive"},
        ],
    },
    {
        "id": "N02",
        "note": "Norilsk nickel sanctions Russia 2022",
        "text": (
            "Western sanctions on Russia following the Ukraine invasion disrupted nickel exports "
            "from Norilsk, with shipments suspended and prices spiking on the LME."
        ),
        "expected": [
            {"commodity": "nickel", "type": "export_restriction", "direction": "positive"},
        ],
    },
    {
        "id": "N03",
        "note": "Nickel demand spike from EV batteries 2021",
        "text": (
            "Nickel demand surged in 2021 as EV manufacturers rushed to secure Class 1 nickel "
            "for high-energy battery cathodes, driving a boom in nickel sulfate contracts."
        ),
        "expected": [
            {"commodity": "nickel", "type": "demand_surge", "direction": "positive"},
        ],
    },
    # ── Soybeans ──────────────────────────────────────────────────────────────
    {
        "id": "S01",
        "note": "US-China trade war tariff 2018",
        "text": (
            "China imposed retaliatory tariffs of 25% on US soybean imports in July 2018 in "
            "response to US trade actions, causing a collapse in US agricultural exports to China."
        ),
        "expected": [
            {"commodity": "soybeans", "type": "export_restriction", "direction": "negative"},
        ],
    },
    {
        "id": "S02",
        "note": "Ukraine war food price spike 2022",
        "text": (
            "Russia's invasion of Ukraine in 2022 triggered a global commodity crisis, with "
            "grain and oilseed prices spiking as exports from the Black Sea region were disrupted."
        ),
        "expected": [
            {"commodity": "soybeans", "type": "macro_demand_shock", "direction": "positive"},
        ],
    },
    {
        "id": "S03",
        "note": "Brazil record soybean harvest supply glut 2023",
        "text": (
            "Brazil produced a record soybean harvest in 2023, generating a global supply glut "
            "that caused a decline in soybean prices as inventories built up."
        ),
        "expected": [
            {"commodity": "soybeans", "type": "demand_surge", "direction": "negative"},
        ],
    },
    # ── Negative test cases (should extract 0 relevant shocks) ────────────────
    {
        "id": "X01",
        "note": "Petroleum reserve drawdown — not a mineral",
        "text": (
            "The US released 50 million barrels from the Strategic Petroleum Reserve in 2021, "
            "causing a drawdown in oil stockpiles and temporarily reducing crude prices."
        ),
        "expected": [],  # no mineral commodity → nothing should match
    },
    {
        "id": "X02",
        "note": "Gold market — not in our commodity set",
        "text": (
            "Gold prices surged to record highs as central banks increased reserve purchases "
            "and inflation fears drove a boom in safe-haven demand."
        ),
        "expected": [],  # gold not in commodity set
    },
    {
        "id": "X03",
        "note": "Vague economic uncertainty — no actionable shock",
        "text": (
            "Global economic uncertainty weighed on commodity markets as rising interest rates "
            "slowed growth and investors reduced risk exposure."
        ),
        "expected": [],  # recession/macro but no specific commodity
    },
]


# ── Evaluation helpers ────────────────────────────────────────────────────────

@dataclass
class ExampleResult:
    id: str
    note: str
    expected: List[Dict]
    extracted: List[Dict]
    commodity_tp: int = 0
    commodity_fp: int = 0
    commodity_fn: int = 0
    type_tp: int = 0
    type_fp: int = 0
    type_fn: int = 0
    direction_correct: int = 0
    direction_total: int = 0

    def commodity_precision(self) -> float:
        denom = self.commodity_tp + self.commodity_fp
        return self.commodity_tp / denom if denom else (1.0 if not self.expected else 0.0)

    def commodity_recall(self) -> float:
        denom = self.commodity_tp + self.commodity_fn
        return self.commodity_tp / denom if denom else (1.0 if not self.expected else 0.0)

    def type_precision(self) -> float:
        denom = self.type_tp + self.type_fp
        return self.type_tp / denom if denom else (1.0 if not self.expected else 0.0)

    def type_recall(self) -> float:
        denom = self.type_tp + self.type_fn
        return self.type_tp / denom if denom else (1.0 if not self.expected else 0.0)

    def direction_accuracy(self) -> float:
        return self.direction_correct / self.direction_total if self.direction_total else float("nan")


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _evaluate_example(example: Dict, mapper: EventShockMapper) -> ExampleResult:
    mappings = mapper.text_to_shocks(example["text"])
    extracted = mapper.mappings_to_dict(mappings)
    expected = example["expected"]

    result = ExampleResult(
        id=example["id"],
        note=example["note"],
        expected=expected,
        extracted=extracted,
    )

    # Commodity-level matching: does the extractor find the right commodity at all?
    expected_commodities = {e["commodity"] for e in expected}
    extracted_commodities = {e["commodity"] for e in extracted}

    result.commodity_tp = len(expected_commodities & extracted_commodities)
    result.commodity_fp = len(extracted_commodities - expected_commodities)
    result.commodity_fn = len(expected_commodities - extracted_commodities)

    # Type-level matching: (commodity, type) pair
    expected_pairs = {(e["commodity"], e["type"]) for e in expected}
    extracted_pairs = {(e["commodity"], e["shock"]["type"]) for e in extracted}

    result.type_tp = len(expected_pairs & extracted_pairs)
    result.type_fp = len(extracted_pairs - expected_pairs)
    result.type_fn = len(expected_pairs - extracted_pairs)

    # Direction accuracy: among type-correct extractions, is the sign right?
    for exp in expected:
        for ext in extracted:
            if ext["commodity"] == exp["commodity"] and ext["shock"]["type"] == exp["type"]:
                result.direction_total += 1
                mag = ext["shock"]["magnitude"]
                correct_positive = exp["direction"] == "positive" and mag > 0
                correct_negative = exp["direction"] == "negative" and mag < 0
                if correct_positive or correct_negative:
                    result.direction_correct += 1
                break  # count each expected shock at most once

    return result


@dataclass
class EvalSummary:
    n_examples: int
    mean_commodity_precision: float
    mean_commodity_recall: float
    commodity_f1: float
    mean_type_precision: float
    mean_type_recall: float
    type_f1: float
    direction_accuracy: float
    per_example: List[ExampleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "commodity": {
                "precision": round(self.mean_commodity_precision, 3),
                "recall": round(self.mean_commodity_recall, 3),
                "f1": round(self.commodity_f1, 3),
            },
            "type": {
                "precision": round(self.mean_type_precision, 3),
                "recall": round(self.mean_type_recall, 3),
                "f1": round(self.type_f1, 3),
            },
            "direction_accuracy": round(self.direction_accuracy, 3) if not math.isnan(self.direction_accuracy) else None,
            "per_example": [
                {
                    "id": r.id,
                    "note": r.note,
                    "n_expected": len(r.expected),
                    "n_extracted": len(r.extracted),
                    "commodity_tp": r.commodity_tp,
                    "commodity_fp": r.commodity_fp,
                    "commodity_fn": r.commodity_fn,
                    "type_tp": r.type_tp,
                    "type_fp": r.type_fp,
                    "type_fn": r.type_fn,
                    "direction_correct": r.direction_correct,
                    "direction_total": r.direction_total,
                }
                for r in self.per_example
            ],
        }


def run_extractor_eval() -> EvalSummary:
    """
    Run the full evaluation of EventShockMapper against the gold standard.

    Returns EvalSummary with commodity/type precision, recall, F1, and
    per-example breakdown.
    """
    mapper = EventShockMapper()
    results = [_evaluate_example(ex, mapper) for ex in GOLD_STANDARD]

    # Aggregate across all examples
    cp = [r.commodity_precision() for r in results]
    cr = [r.commodity_recall() for r in results]
    tp_ = [r.type_precision() for r in results]
    tr = [r.type_recall() for r in results]
    dir_correct = sum(r.direction_correct for r in results)
    dir_total = sum(r.direction_total for r in results)

    mean_cp = sum(cp) / len(cp)
    mean_cr = sum(cr) / len(cr)
    mean_tp = sum(tp_) / len(tp_)
    mean_tr = sum(tr) / len(tr)

    return EvalSummary(
        n_examples=len(results),
        mean_commodity_precision=mean_cp,
        mean_commodity_recall=mean_cr,
        commodity_f1=_f1(mean_cp, mean_cr),
        mean_type_precision=mean_tp,
        mean_type_recall=mean_tr,
        type_f1=_f1(mean_tp, mean_tr),
        direction_accuracy=dir_correct / dir_total if dir_total else float("nan"),
        per_example=results,
    )
