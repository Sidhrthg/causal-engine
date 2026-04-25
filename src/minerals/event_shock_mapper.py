"""
Event → ShockConfig mapper.

Bridges the Knowledge Graph event extraction pipeline (Option C) to the
causal model's ShockConfig schema.

Pipeline:
    raw text
      → KGExtractor.extract_from_text()   (LLM triple extraction)
      → EventShockMapper.triples_to_shocks()  (predicate → ShockConfig)
      → run_scenario(ScenarioConfig(shocks=...))

The mapper uses:
  1. Predicate rules  — RelationType → shock_type (deterministic, fast)
  2. KG traversal     — propagate_shock() to find affected commodities
  3. Magnitude heuristics — relationship confidence + predicate strength

This is the missing link between the KG (structured knowledge) and the
causal ODE model (quantitative dynamics).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .knowledge_graph import (
    CausalKnowledgeGraph,
    EntityType,
    RelationType,
    build_critical_minerals_kg,
)
from .schema import ShockConfig


# ── Predicate → shock type mapping ───────────────────────────────────────────
#
# Maps KG RelationType (or raw predicate string) to a ShockConfig type.
# Magnitude scaling: HIGH confidence → full magnitude, MEDIUM → 0.7×, LOW → 0.4×

_REL_TO_SHOCK: Dict[RelationType, str] = {
    RelationType.REGULATES:   "export_restriction",   # policy restricts supply
    RelationType.DISRUPTS:    "export_restriction",   # disruption = supply cut
    RelationType.CAUSES:      "demand_surge",         # generic causal event
    RelationType.PRODUCES:    "demand_surge",         # production change
    RelationType.SUPPLIES:    "export_restriction",   # supply relationship shock
    RelationType.EXPORTS_TO:  "export_restriction",   # export flow change
    RelationType.DEPENDS_ON:  "capex_shock",          # dependency break
    RelationType.MITIGATES:   "stockpile_release",    # mitigation = release
}

# Predicate keywords → shock type (for raw string matching from LLM output)
_KEYWORD_SHOCK: List[Tuple[str, str, float]] = [
    # (keyword_in_predicate, shock_type, base_magnitude)
    ("ban",           "export_restriction", 0.50),
    ("restrict",      "export_restriction", 0.35),
    ("controls",      "export_restriction", 0.35),   # "export controls"
    ("licenc",        "export_restriction", 0.30),   # "licence/license requirement"
    ("quota",         "export_restriction", 0.30),
    ("sanction",      "export_restriction", 0.40),
    ("tariff",        "export_restriction", 0.16),
    ("embargo",       "export_restriction", 0.60),
    ("suspend",       "export_restriction", 0.45),
    ("nationali",     "export_restriction", 0.25),   # "nationalise/nationalize"
    ("surge",         "demand_surge",       0.30),
    ("boom",          "demand_surge",       0.25),
    ("spike",         "demand_surge",       0.30),
    ("soar",          "demand_surge",       0.30),
    ("spiking",       "demand_surge",       0.30),
    ("collapse",      "demand_surge",      -0.40),
    ("crash",         "demand_surge",      -0.35),
    ("drop",          "demand_surge",      -0.25),
    ("decline",       "demand_surge",      -0.20),
    ("slump",         "demand_surge",      -0.25),
    ("glut",          "demand_surge",      -0.20),   # "supply glut"
    ("strike",        "capex_shock",        0.30),
    ("delay",         "capex_shock",        0.20),
    ("shutdown",      "capex_shock",        0.45),
    ("closure",       "capex_shock",        0.40),
    ("close",         "capex_shock",        0.35),   # "Glencore closed"
    ("release",       "stockpile_release",  15.0),
    ("drawdown",      "stockpile_release",  10.0),
    ("inventory",     "stockpile_release",  12.0),
    ("recession",     "macro_demand_shock", -0.35),
    ("gfc",           "macro_demand_shock", -0.40),
    ("crisis",        "macro_demand_shock", -0.30),
    ("invasion",      "macro_demand_shock",  0.25),  # Ukraine war etc.
    ("war",           "macro_demand_shock",  0.20),
]

_CONFIDENCE_SCALE = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}

# Known commodity aliases → canonical name
_COMMODITY_ALIASES = {
    "graphite": "graphite", "natural graphite": "graphite",
    "lithium": "lithium", "li": "lithium",
    "cobalt": "cobalt", "co": "cobalt",
    "nickel": "nickel", "ni": "nickel",
    "copper": "copper", "cu": "copper",
    "soybeans": "soybeans", "soybean": "soybeans", "soy": "soybeans",
    "rare earth": "rare_earths", "rare earths": "rare_earths",
    "rare_earths": "rare_earths",
}


def _resolve_commodity(name: str) -> Optional[str]:
    return _COMMODITY_ALIASES.get(name.lower().strip())


# Pre-compiled word-boundary patterns for commodity matching in free text.
# Short aliases (≤3 chars) require whole-word match to avoid substring noise
# (e.g. "li" in "oil", "co" in "record").
_COMMODITY_PATTERNS: List[Tuple[str, str]] = []  # populated below
def _build_commodity_patterns() -> None:
    for alias, canonical in _COMMODITY_ALIASES.items():
        # Word-boundary match for all aliases (handles "soybean" vs "soybeans")
        pattern = r"\b" + re.escape(alias) + r"\b"
        _COMMODITY_PATTERNS.append((pattern, canonical))

_build_commodity_patterns()


def _find_commodities_in_text(text_lower: str) -> List[str]:
    """Return unique canonical commodity names found (with word-boundary safety)."""
    seen: set = set()
    found = []
    for pattern, canonical in _COMMODITY_PATTERNS:
        if canonical not in seen and re.search(pattern, text_lower):
            seen.add(canonical)
            found.append(canonical)
    return found


def _extract_year(text: str, fallback: int = 2024) -> int:
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    return int(m.group()) if m else fallback


@dataclass
class ExtractedEvent:
    """A single event extracted from text, ready for shock mapping."""
    subject: str                     # e.g. "China"
    predicate: str                   # raw predicate string
    object: str                      # e.g. "graphite exports"
    confidence: float
    evidence: str
    year: Optional[int]
    commodity: Optional[str] = None  # resolved commodity name
    rel_type: Optional[RelationType] = None


@dataclass
class ShockMapping:
    """Result of mapping an ExtractedEvent to a ShockConfig."""
    event: ExtractedEvent
    shock: ShockConfig
    commodity: str
    affected_entities: List[str] = field(default_factory=list)
    propagation_depth: int = 0
    reasoning: str = ""


class EventShockMapper:
    """
    Maps extracted KG events to ShockConfig objects.

    Uses three layers:
    1. Keyword rules on raw predicate strings (fast, deterministic)
    2. RelationType → shock_type mapping
    3. KG propagation to identify affected commodities

    Args:
        kg: Pre-built CausalKnowledgeGraph. If None, uses the default
            critical minerals KG.
    """

    def __init__(self, kg: Optional[CausalKnowledgeGraph] = None):
        self.kg = kg or build_critical_minerals_kg()

    # ── Core mapping ─────────────────────────────────────────────────────────

    def event_to_shock(
        self,
        event: ExtractedEvent,
        default_duration: int = 2,
    ) -> Optional[ShockMapping]:
        """
        Map a single ExtractedEvent to a ShockMapping.

        Returns None if no shock type can be inferred.
        """
        shock_type, magnitude = self._infer_shock(event)
        if shock_type is None:
            return None

        commodity = event.commodity or self._infer_commodity(event)
        if commodity is None:
            return None

        year = event.year or 2024
        conf_scale = _CONFIDENCE_SCALE.get(
            "HIGH" if event.confidence >= 0.7 else
            "MEDIUM" if event.confidence >= 0.4 else "LOW",
            0.7
        )
        scaled_magnitude = magnitude * conf_scale

        shock = ShockConfig(
            type=shock_type,
            start_year=year,
            end_year=year + default_duration - 1,
            magnitude=round(scaled_magnitude, 3),
        )

        # KG propagation: find what this shock affects
        affected, depth = self._propagate(event.subject, commodity)

        reasoning = (
            f"'{event.predicate}' on '{event.object}' → {shock_type} "
            f"(magnitude={scaled_magnitude:.2f}, confidence={event.confidence:.2f})"
        )

        return ShockMapping(
            event=event,
            shock=shock,
            commodity=commodity,
            affected_entities=affected,
            propagation_depth=depth,
            reasoning=reasoning,
        )

    def triples_to_shocks(
        self,
        triples: List[Dict[str, Any]],
        default_duration: int = 2,
    ) -> List[ShockMapping]:
        """
        Convert a list of raw triple dicts (from KGExtractor) to ShockMappings.

        Filters out triples that can't be mapped to a shock type or commodity.
        """
        mappings = []
        for t in triples:
            year_raw = t.get("year")
            year = _extract_year(str(year_raw)) if year_raw else None

            event = ExtractedEvent(
                subject=t.get("subject", ""),
                predicate=t.get("predicate", ""),
                object=t.get("object", ""),
                confidence=float(t.get("confidence", 0.5)),
                evidence=t.get("evidence", ""),
                year=year,
                commodity=_resolve_commodity(t.get("object", ""))
                          or _resolve_commodity(t.get("subject", "")),
            )
            mapping = self.event_to_shock(event, default_duration=default_duration)
            if mapping:
                mappings.append(mapping)

        return mappings

    def text_to_shocks(
        self,
        text: str,
        extractor=None,
        default_duration: int = 2,
    ) -> List[ShockMapping]:
        """
        Full pipeline: raw text → ShockMappings.

        Args:
            text: News article, policy document, or any text.
            extractor: KGExtractor instance. If None, uses rule-based fallback.
            default_duration: Default shock duration in years.

        Returns:
            List of ShockMappings, sorted by confidence descending.
        """
        if extractor is not None:
            triples = extractor.extract_from_text(text)
        else:
            triples = self._rule_based_extract(text)

        mappings = self.triples_to_shocks(triples, default_duration=default_duration)
        return sorted(mappings, key=lambda m: m.event.confidence, reverse=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _infer_shock(
        self, event: ExtractedEvent
    ) -> Tuple[Optional[str], float]:
        """Return (shock_type, base_magnitude) or (None, 0)."""
        pred_lower = event.predicate.lower()

        # Layer 1: keyword rules
        for keyword, shock_type, magnitude in _KEYWORD_SHOCK:
            if keyword in pred_lower:
                return shock_type, magnitude

        # Layer 2: RelationType mapping
        if event.rel_type and event.rel_type in _REL_TO_SHOCK:
            shock_type = _REL_TO_SHOCK[event.rel_type]
            # Default magnitude from confidence
            return shock_type, 0.25

        return None, 0.0

    def _infer_commodity(self, event: ExtractedEvent) -> Optional[str]:
        """Try to resolve commodity from subject, object, or evidence."""
        for text in (event.object, event.subject, event.evidence):
            c = _resolve_commodity(text)
            if c:
                return c
        return None

    def _propagate(
        self, origin: str, commodity: str
    ) -> Tuple[List[str], int]:
        """
        Use KG shock propagation to find affected entities.
        Returns (affected_entity_ids, max_depth_reached).
        """
        try:
            origin_id = self.kg.resolve_id(origin)
            trace = self.kg.propagate_shock(origin_id, initial_magnitude=1.0, decay=0.6, max_depth=3)
            affected = [k for k, v in sorted(trace.affected.items(), key=lambda x: -x[1])]
            return affected[:10], 3
        except Exception:
            return [], 0

    def _rule_based_extract(self, text: str) -> List[Dict[str, Any]]:
        """
        Lightweight rule-based fallback when no LLM extractor is available.

        Pass 1: (country, keyword, commodity) → confidence 0.60
        Pass 2: (keyword, commodity) without country → confidence 0.45
                Only fires when no triples were found in pass 1, to avoid
                duplicating high-confidence country-anchored extractions.
        """
        countries = ["china", "usa", "australia", "chile", "indonesia",
                     "drc", "congo", "russia", "india", "ukraine", "brazil",
                     "indonesia", "argentina", "japan", "south korea"]
        triples = []
        text_lower = text.lower()

        # Specific false-positive overrides: blocklist suffixes that turn a keyword
        # into a different word (e.g. "ban" → "banks", "close" → "closely")
        _KW_BLOCKLIST = {
            "ban": r"\bbank",          # "banks", "banking" ≠ "ban"
            "close": r"\bclosely",
            "war": r"\bwarrant",
        }

        def _keyword_in(kw: str, txt: str) -> bool:
            # Check word-start boundary; keywords may be inflected ("banned", "restricts")
            found = bool(re.search(r"\b" + re.escape(kw), txt))
            if not found:
                return False
            # Apply blocklist to reject false positive suffixes
            blocklist_pattern = _KW_BLOCKLIST.get(kw)
            if blocklist_pattern and re.search(blocklist_pattern, txt):
                # The match might be the blocklisted form; do a finer check:
                # find all word-start matches and ensure at least one is NOT blocklisted
                matches = [m.start() for m in re.finditer(r"\b" + re.escape(kw), txt)]
                for pos in matches:
                    word = re.match(r"\S+", txt[pos:])
                    if word and not re.search(blocklist_pattern, txt[pos:pos + 1 + len(word.group())]):
                        return True
                return False
            return True

        commodities_in_text = _find_commodities_in_text(text_lower)

        # Pass 1 — country-anchored
        for country in countries:
            if country not in text_lower:
                continue
            for keyword, shock_type, magnitude in _KEYWORD_SHOCK:
                if not _keyword_in(keyword, text_lower):
                    continue
                for canonical in commodities_in_text:
                    year = _extract_year(text)
                    triples.append({
                        "subject": country,
                        "predicate": keyword,
                        "object": canonical,
                        "confidence": 0.6,
                        "evidence": text[:200],
                        "year": str(year) if year else None,
                    })

        # Pass 2 — commodity + keyword only (no country required)
        # covered_in_p1: commodities already handled at high confidence in Pass 1
        # covered_pairs: (canonical, shock_type) pairs already emitted — prevents
        #   duplicates while still allowing multiple shock types per commodity.
        covered_in_p1 = {t["object"] for t in triples}
        covered_pairs: set = set()
        for keyword, shock_type, magnitude in _KEYWORD_SHOCK:
            if not _keyword_in(keyword, text_lower):
                continue
            for canonical in commodities_in_text:
                if (canonical, shock_type) in covered_pairs:
                    continue
                # Lower confidence when adding to a commodity already covered by Pass 1
                confidence = 0.30 if canonical in covered_in_p1 else 0.45
                year = _extract_year(text)
                triples.append({
                    "subject": "unknown",
                    "predicate": keyword,
                    "object": canonical,
                    "confidence": confidence,
                    "evidence": text[:200],
                    "year": str(year) if year else None,
                })
                covered_pairs.add((canonical, shock_type))

        # Deduplicate on (subject, predicate, object)
        seen: set = set()
        unique = []
        for t in triples:
            key = (t["subject"], t["predicate"], t["object"])
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique

    # ── Serialization ─────────────────────────────────────────────────────────

    def mappings_to_dict(self, mappings: List[ShockMapping]) -> List[Dict]:
        return [
            {
                "commodity": m.commodity,
                "shock": m.shock.model_dump(),
                "affected_entities": m.affected_entities,
                "reasoning": m.reasoning,
                "evidence": m.event.evidence,
                "confidence": m.event.confidence,
            }
            for m in mappings
        ]
