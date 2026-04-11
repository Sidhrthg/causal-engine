"""
Causal Knowledge Graph for supply chain inference.

A knowledge graph (KG) combines structured, semantic, and domain-specific
knowledge with causal modeling.  Unlike a bare CausalDAG (which stores only
variable names and "causes" edges), a KG carries:

  - **Typed entities** — countries, commodities, policies, companies, …
  - **Typed relationships** — produces, exports_to, causes, substitutes, …
  - **Property annotations** — confidence, mechanism, evidence, temporal scope
  - **Formal schema** — defines which entity/relationship types exist and
    which connections are allowed between them
  - **Temporal tracking** — entities and relationships carry time ranges
    so the graph can be queried at a specific point in time
  - **Hierarchical taxonomy** — IS_A relationships for classification
  - **Data provenance** — every relationship tracks its source and confidence
  - **Query capability** — traverse upstream/downstream, find paths, propagate
    shocks, identify confounders, query by time

The KG is commodity-agnostic and can represent any supply chain.  It bridges
to Pearl's 3-layer engine via ``to_causal_dag()``, which extracts the CAUSES
sub-graph as a ``CausalDAG`` that the ``CausalInferenceEngine`` can consume
for association, intervention, and counterfactual analysis.

Reference: Hogan et al. (2021). Knowledge Graphs. ACM Computing Surveys.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from .causal_inference import CausalDAG


# ============================================================================
# Enums
# ============================================================================


class EntityType(Enum):
    """Types of entities in the supply chain knowledge graph."""

    COMMODITY = "commodity"
    COUNTRY = "country"
    POLICY = "policy"
    COMPANY = "company"
    FACILITY = "facility"
    MARKET = "market"
    TECHNOLOGY = "technology"
    EVENT = "event"
    ECONOMIC_INDICATOR = "economic_indicator"
    INDUSTRY = "industry"
    REGION = "region"
    TRADE_ROUTE = "trade_route"
    RISK_FACTOR = "risk_factor"


class RelationType(Enum):
    """Types of relationships between entities."""

    # Causal
    CAUSES = "causes"

    # Production & trade
    PRODUCES = "produces"
    EXPORTS_TO = "exports_to"
    IMPORTS_FROM = "imports_from"
    SUPPLIES = "supplies"
    PROCESSES = "processes"

    # Dependency & substitution
    DEPENDS_ON = "depends_on"
    SUBSTITUTES = "substitutes"
    COMPETES_WITH = "competes_with"
    INPUTS = "inputs"
    OUTPUTS = "outputs"

    # Governance & disruption
    REGULATES = "regulates"
    DISRUPTS = "disrupts"
    MITIGATES = "mitigates"

    # Structural & hierarchical
    LOCATED_IN = "located_in"
    PART_OF = "part_of"
    IS_A = "is_a"
    CONSUMES = "consumes"
    ENABLES = "enables"

    # Temporal sequencing
    PRECEDED_BY = "preceded_by"
    FOLLOWED_BY = "followed_by"


# ============================================================================
# Schema — allowed connections
# ============================================================================


@dataclass(frozen=True)
class AllowedConnection:
    """An allowed (source_type, relation_type, target_type) triple."""

    source_type: EntityType
    relation_type: RelationType
    target_type: EntityType


class KGSchema:
    """
    Schema definition for the knowledge graph.

    Declares all entity types, relationship types, and which connections
    are valid.  Used for validation when ``strict=True`` in the KG.

    The schema also documents the ontology for consumers of the KG
    (e.g., UI dropdowns, auto-complete, documentation).
    """

    def __init__(self) -> None:
        self.entity_types: Set[EntityType] = set(EntityType)
        self.relation_types: Set[RelationType] = set(RelationType)
        self._allowed: Set[FrozenSet] = set()

    def allow(
        self,
        source_type: EntityType,
        relation_type: RelationType,
        target_type: EntityType,
    ) -> None:
        """Declare an allowed connection triple."""
        key = frozenset([
            ("src", source_type.value),
            ("rel", relation_type.value),
            ("tgt", target_type.value),
        ])
        self._allowed.add(key)

    def is_allowed(
        self,
        source_type: EntityType,
        relation_type: RelationType,
        target_type: EntityType,
    ) -> bool:
        """Check if a connection triple is allowed by the schema."""
        if not self._allowed:
            return True  # no constraints defined = everything allowed
        key = frozenset([
            ("src", source_type.value),
            ("rel", relation_type.value),
            ("tgt", target_type.value),
        ])
        return key in self._allowed

    def allowed_connections(self) -> List[AllowedConnection]:
        """Return all allowed connection triples."""
        results = []
        for key in self._allowed:
            d = dict(key)
            results.append(AllowedConnection(
                source_type=EntityType(d["src"]),
                relation_type=RelationType(d["rel"]),
                target_type=EntityType(d["tgt"]),
            ))
        return results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_types": [t.value for t in sorted(self.entity_types, key=lambda x: x.value)],
            "relation_types": [t.value for t in sorted(self.relation_types, key=lambda x: x.value)],
            "allowed_connections": [
                {"source_type": c.source_type.value,
                 "relation_type": c.relation_type.value,
                 "target_type": c.target_type.value}
                for c in self.allowed_connections()
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> KGSchema:
        schema = cls()
        for c in d.get("allowed_connections", []):
            schema.allow(
                EntityType(c["source_type"]),
                RelationType(c["relation_type"]),
                EntityType(c["target_type"]),
            )
        return schema


def build_supply_chain_schema() -> KGSchema:
    """
    Build the default schema for supply chain knowledge graphs.

    Defines which (source_type, relationship, target_type) triples are
    semantically valid.
    """
    s = KGSchema()
    E = EntityType
    R = RelationType

    # --- Production & trade ---
    s.allow(E.COUNTRY, R.PRODUCES, E.COMMODITY)
    s.allow(E.COMPANY, R.PRODUCES, E.COMMODITY)
    s.allow(E.FACILITY, R.PRODUCES, E.COMMODITY)
    s.allow(E.COUNTRY, R.EXPORTS_TO, E.COUNTRY)
    s.allow(E.COMPANY, R.EXPORTS_TO, E.COUNTRY)
    s.allow(E.COUNTRY, R.IMPORTS_FROM, E.COUNTRY)
    s.allow(E.COMPANY, R.SUPPLIES, E.COMPANY)
    s.allow(E.COUNTRY, R.SUPPLIES, E.COUNTRY)
    s.allow(E.COMPANY, R.SUPPLIES, E.INDUSTRY)
    s.allow(E.COMPANY, R.PROCESSES, E.COMMODITY)
    s.allow(E.FACILITY, R.PROCESSES, E.COMMODITY)

    # --- Dependency ---
    s.allow(E.COUNTRY, R.DEPENDS_ON, E.COUNTRY)
    s.allow(E.INDUSTRY, R.DEPENDS_ON, E.COMMODITY)
    s.allow(E.TECHNOLOGY, R.DEPENDS_ON, E.COMMODITY)
    s.allow(E.COMPANY, R.DEPENDS_ON, E.COMMODITY)

    # --- Substitution & competition ---
    s.allow(E.COMMODITY, R.SUBSTITUTES, E.COMMODITY)
    s.allow(E.TECHNOLOGY, R.SUBSTITUTES, E.TECHNOLOGY)
    s.allow(E.COMPANY, R.COMPETES_WITH, E.COMPANY)
    s.allow(E.COUNTRY, R.COMPETES_WITH, E.COUNTRY)

    # --- Consumption & enabling ---
    s.allow(E.INDUSTRY, R.CONSUMES, E.COMMODITY)
    s.allow(E.TECHNOLOGY, R.CONSUMES, E.COMMODITY)
    s.allow(E.COMMODITY, R.ENABLES, E.TECHNOLOGY)
    s.allow(E.COMMODITY, R.ENABLES, E.INDUSTRY)
    s.allow(E.TECHNOLOGY, R.ENABLES, E.INDUSTRY)

    # --- Inputs / outputs ---
    s.allow(E.COMMODITY, R.INPUTS, E.TECHNOLOGY)
    s.allow(E.COMMODITY, R.INPUTS, E.FACILITY)
    s.allow(E.TECHNOLOGY, R.OUTPUTS, E.COMMODITY)
    s.allow(E.FACILITY, R.OUTPUTS, E.COMMODITY)

    # --- Governance ---
    s.allow(E.POLICY, R.REGULATES, E.COMMODITY)
    s.allow(E.POLICY, R.REGULATES, E.INDUSTRY)
    s.allow(E.POLICY, R.REGULATES, E.COMPANY)
    s.allow(E.POLICY, R.REGULATES, E.COUNTRY)
    s.allow(E.POLICY, R.MITIGATES, E.RISK_FACTOR)
    s.allow(E.POLICY, R.MITIGATES, E.EVENT)

    # --- Disruption ---
    s.allow(E.EVENT, R.DISRUPTS, E.COMMODITY)
    s.allow(E.EVENT, R.DISRUPTS, E.COUNTRY)
    s.allow(E.EVENT, R.DISRUPTS, E.COMPANY)
    s.allow(E.EVENT, R.DISRUPTS, E.FACILITY)
    s.allow(E.EVENT, R.DISRUPTS, E.TRADE_ROUTE)
    s.allow(E.RISK_FACTOR, R.DISRUPTS, E.COMMODITY)

    # --- Causal (broad — any type can cause effects on another) ---
    for src in E:
        for tgt in E:
            s.allow(src, R.CAUSES, tgt)

    # --- Structural / hierarchical ---
    s.allow(E.FACILITY, R.LOCATED_IN, E.COUNTRY)
    s.allow(E.COMPANY, R.LOCATED_IN, E.COUNTRY)
    s.allow(E.COUNTRY, R.PART_OF, E.REGION)
    s.allow(E.FACILITY, R.PART_OF, E.COMPANY)
    s.allow(E.COMMODITY, R.PART_OF, E.COMMODITY)  # sub-commodities

    # --- Taxonomy (IS_A) ---
    for t in E:
        s.allow(t, R.IS_A, t)  # any entity can IS_A another of same type

    # --- Temporal sequencing ---
    s.allow(E.EVENT, R.PRECEDED_BY, E.EVENT)
    s.allow(E.EVENT, R.FOLLOWED_BY, E.EVENT)
    s.allow(E.POLICY, R.PRECEDED_BY, E.POLICY)
    s.allow(E.POLICY, R.FOLLOWED_BY, E.POLICY)

    return s


# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class Entity:
    """A typed node in the knowledge graph with temporal tracking."""

    id: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)

    # Temporal: when this entity is/was active (None = always)
    start_date: Optional[str] = None  # ISO format or year, e.g. "2010" or "2010-01-15"
    end_date: Optional[str] = None

    def active_at(self, year: int) -> bool:
        """Check if entity is active at a given year."""
        if self.start_date is not None:
            start_year = int(str(self.start_date)[:4])
            if year < start_year:
                return False
        if self.end_date is not None:
            end_year = int(str(self.end_date)[:4])
            if year > end_year:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "aliases": self.aliases,
        }
        if self.start_date is not None:
            d["start_date"] = self.start_date
        if self.end_date is not None:
            d["end_date"] = self.end_date
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Entity:
        return cls(
            id=d["id"],
            entity_type=EntityType(d["entity_type"]),
            properties=d.get("properties", {}),
            aliases=d.get("aliases", []),
            start_date=d.get("start_date"),
            end_date=d.get("end_date"),
        )


@dataclass
class Relationship:
    """A typed, directed edge with temporal tracking and provenance."""

    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)

    # Temporal: when this relationship is/was active
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @property
    def confidence(self) -> str:
        return self.properties.get("confidence", "MEDIUM")

    @property
    def mechanism(self) -> str:
        return self.properties.get("mechanism", "")

    @property
    def provenance(self) -> str:
        """Data source for this relationship."""
        return self.properties.get("provenance", self.properties.get("source_document", ""))

    @property
    def evidence(self) -> str:
        return self.properties.get("evidence", "")

    def active_at(self, year: int) -> bool:
        """Check if relationship is active at a given year."""
        if self.start_date is not None:
            start_year = int(str(self.start_date)[:4])
            if year < start_year:
                return False
        if self.end_date is not None:
            end_year = int(str(self.end_date)[:4])
            if year > end_year:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
        }
        if self.start_date is not None:
            d["start_date"] = self.start_date
        if self.end_date is not None:
            d["end_date"] = self.end_date
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Relationship:
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            relation_type=RelationType(d["relation_type"]),
            properties=d.get("properties", {}),
            start_date=d.get("start_date"),
            end_date=d.get("end_date"),
        )


# ============================================================================
# Shock propagation result
# ============================================================================


@dataclass
class ShockTrace:
    """Result of propagating a shock through the knowledge graph."""

    origin: str
    affected: Dict[str, float]  # entity_id -> impact magnitude (0..1)
    paths: Dict[str, List[str]]  # entity_id -> path from origin
    depth: int


# ============================================================================
# CausalKnowledgeGraph
# ============================================================================


class CausalKnowledgeGraph:
    """
    A typed, queryable knowledge graph for causal supply chain reasoning.

    Backed by a ``networkx.MultiDiGraph`` (supports multiple typed edges
    between the same pair of nodes).  Each node stores an ``Entity``; each
    edge stores a ``Relationship``.

    Features:
      - **Schema validation** — optional ``KGSchema`` enforces which
        (source_type, relation_type, target_type) triples are allowed.
      - **Temporal queries** — entities and relationships carry time ranges;
        ``query_at_time()`` returns a snapshot of the graph at a given year.
      - **Hierarchical taxonomy** — ``IS_A`` and ``PART_OF`` relationships
        enable classification and ``get_taxonomy()`` traversal.
      - **Provenance** — each relationship tracks confidence, mechanism,
        evidence, and data source.
      - **Causal DAG extraction** — ``to_causal_dag()`` produces a
        ``CausalDAG`` for Pearl's 3-layer engine.

    Args:
        schema: Optional schema for validation.  If ``None``, no validation
                is performed on ``add_relationship()``.
        strict: If ``True``, raise ``ValueError`` when a relationship
                violates the schema.  If ``False``, log a warning.
    """

    def __init__(
        self,
        schema: Optional[KGSchema] = None,
        strict: bool = False,
    ) -> None:
        self._graph = nx.MultiDiGraph()
        self._entities: Dict[str, Entity] = {}
        self._alias_map: Dict[str, str] = {}  # alias -> canonical id
        self.schema = schema
        self.strict = strict
        self._validation_warnings: List[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relationships(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Entity CRUD
    # ------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """Add an entity (node) to the knowledge graph."""
        self._entities[entity.id] = entity
        self._graph.add_node(entity.id, entity=entity)
        for alias in entity.aliases:
            self._alias_map[alias.lower()] = entity.id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Look up an entity by ID or alias."""
        if entity_id in self._entities:
            return self._entities[entity_id]
        canonical = self._alias_map.get(entity_id.lower())
        if canonical:
            return self._entities.get(canonical)
        return None

    def resolve_id(self, id_or_alias: str) -> str:
        """Resolve an alias to its canonical entity ID."""
        if id_or_alias in self._entities:
            return id_or_alias
        return self._alias_map.get(id_or_alias.lower(), id_or_alias)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Return all entities of a given type."""
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def remove_entity(self, entity_id: str) -> None:
        """Remove an entity and all its relationships."""
        eid = self.resolve_id(entity_id)
        if eid in self._entities:
            # Remove alias mappings
            entity = self._entities[eid]
            for alias in entity.aliases:
                self._alias_map.pop(alias.lower(), None)
            del self._entities[eid]
            self._graph.remove_node(eid)

    # ------------------------------------------------------------------
    # Relationship CRUD
    # ------------------------------------------------------------------

    def add_relationship(self, rel: Relationship) -> None:
        """Add a typed relationship (edge) to the knowledge graph.

        If a schema is set, validates that the (source_type, relation_type,
        target_type) triple is allowed.  Raises ``ValueError`` if
        ``strict=True`` and the triple is disallowed.
        """
        src = self.resolve_id(rel.source_id)
        tgt = self.resolve_id(rel.target_id)
        # Auto-create entity nodes if they don't exist yet
        if src not in self._entities:
            self.add_entity(Entity(id=src, entity_type=EntityType.COMMODITY))
        if tgt not in self._entities:
            self.add_entity(Entity(id=tgt, entity_type=EntityType.COMMODITY))

        # Schema validation
        if self.schema is not None:
            src_type = self._entities[src].entity_type
            tgt_type = self._entities[tgt].entity_type
            if not self.schema.is_allowed(src_type, rel.relation_type, tgt_type):
                msg = (
                    f"Schema violation: {src_type.value} --{rel.relation_type.value}--> "
                    f"{tgt_type.value} ('{src}' -> '{tgt}') is not allowed."
                )
                self._validation_warnings.append(msg)
                if self.strict:
                    raise ValueError(msg)

        self._graph.add_edge(
            src,
            tgt,
            key=rel.relation_type.value,
            relationship=rel,
        )

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Query relationships, optionally filtering by source, target, or type."""
        results: List[Relationship] = []
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if source_id is not None and u != self.resolve_id(source_id):
                continue
            if target_id is not None and v != self.resolve_id(target_id):
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return results

    def get_relationships_by_type(self, relation_type: RelationType) -> List[Relationship]:
        """Return all relationships of a given type."""
        return self.get_relationships(relation_type=relation_type)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both",
    ) -> List[Entity]:
        """
        Get neighboring entities, optionally filtered by relationship type.

        Args:
            entity_id:     Entity to query from.
            relation_type: Filter to this relationship type (None = all).
            direction:     "out" (successors), "in" (predecessors), or "both".
        """
        eid = self.resolve_id(entity_id)
        neighbor_ids: Set[str] = set()

        if direction in ("out", "both"):
            for _, v, data in self._graph.out_edges(eid, data=True):
                rel: Relationship = data["relationship"]
                if relation_type is None or rel.relation_type == relation_type:
                    neighbor_ids.add(v)

        if direction in ("in", "both"):
            for u, _, data in self._graph.in_edges(eid, data=True):
                rel = data["relationship"]
                if relation_type is None or rel.relation_type == relation_type:
                    neighbor_ids.add(u)

        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]

    def get_upstream(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 10,
    ) -> List[Entity]:
        """
        Get all upstream (predecessor) entities reachable via specified
        relationship types.  BFS traversal up to max_depth.
        """
        eid = self.resolve_id(entity_id)
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(eid, 0)])
        upstream: List[Entity] = []

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth:
                continue
            for u, _, data in self._graph.in_edges(current, data=True):
                rel: Relationship = data["relationship"]
                if relation_types and rel.relation_type not in relation_types:
                    continue
                if u not in visited:
                    visited.add(u)
                    upstream.append(self._entities[u])
                    queue.append((u, depth + 1))

        return upstream

    def get_downstream(
        self,
        entity_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 10,
    ) -> List[Entity]:
        """
        Get all downstream (successor) entities reachable via specified
        relationship types.  BFS traversal up to max_depth.
        """
        eid = self.resolve_id(entity_id)
        visited: Set[str] = set()
        queue: deque[Tuple[str, int]] = deque([(eid, 0)])
        downstream: List[Entity] = []

        while queue:
            current, depth = queue.popleft()
            if depth > max_depth:
                continue
            for _, v, data in self._graph.out_edges(current, data=True):
                rel: Relationship = data["relationship"]
                if relation_types and rel.relation_type not in relation_types:
                    continue
                if v not in visited:
                    visited.add(v)
                    downstream.append(self._entities[v])
                    queue.append((v, depth + 1))

        return downstream

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 5,
    ) -> List[List[str]]:
        """
        Find all simple paths from source to target, optionally filtered
        to edges of specified relationship types.
        """
        src = self.resolve_id(source_id)
        tgt = self.resolve_id(target_id)

        if relation_types is None:
            # Use all edges
            try:
                return list(nx.all_simple_paths(self._graph, src, tgt, cutoff=max_depth))
            except nx.NetworkXError:
                return []

        # Build a filtered view (MultiDiGraph edge filter takes u, v, key)
        def edge_filter(u: str, v: str, key: str) -> bool:
            data = self._graph[u][v][key]
            rel: Relationship = data["relationship"]
            return rel.relation_type in relation_types

        view = nx.subgraph_view(self._graph, filter_edge=edge_filter)
        try:
            return list(nx.all_simple_paths(view, src, tgt, cutoff=max_depth))
        except nx.NetworkXError:
            return []

    def subgraph(
        self,
        entity_ids: List[str],
    ) -> CausalKnowledgeGraph:
        """Extract a sub-KG containing only the specified entities."""
        ids = {self.resolve_id(eid) for eid in entity_ids}
        sub = CausalKnowledgeGraph()
        for eid in ids:
            entity = self._entities.get(eid)
            if entity:
                sub.add_entity(entity)
        for u, v, data in self._graph.edges(data=True):
            if u in ids and v in ids:
                sub.add_relationship(data["relationship"])
        return sub

    # ------------------------------------------------------------------
    # Temporal queries
    # ------------------------------------------------------------------

    def query_at_time(self, year: int) -> CausalKnowledgeGraph:
        """
        Return a snapshot of the KG at a specific year.

        Includes only entities active at ``year`` and relationships
        active at ``year`` whose endpoints are both active.
        """
        snapshot = CausalKnowledgeGraph(schema=self.schema, strict=self.strict)
        for entity in self._entities.values():
            if entity.active_at(year):
                snapshot.add_entity(entity)
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if (
                rel.active_at(year)
                and u in snapshot._entities
                and v in snapshot._entities
            ):
                snapshot.add_relationship(rel)
        return snapshot

    def get_relationships_at(
        self,
        year: int,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> List[Relationship]:
        """Return relationships active at ``year``, with optional filters."""
        results: List[Relationship] = []
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if not rel.active_at(year):
                continue
            if source_id is not None and u != self.resolve_id(source_id):
                continue
            if target_id is not None and v != self.resolve_id(target_id):
                continue
            if relation_type is not None and rel.relation_type != relation_type:
                continue
            results.append(rel)
        return results

    # ------------------------------------------------------------------
    # Taxonomy / hierarchy (IS_A, PART_OF)
    # ------------------------------------------------------------------

    def get_taxonomy(
        self,
        entity_id: str,
        relation_type: RelationType = RelationType.IS_A,
        direction: str = "up",
    ) -> List[Entity]:
        """
        Traverse the IS_A (or PART_OF) hierarchy.

        Args:
            entity_id:     Starting entity.
            relation_type: IS_A or PART_OF.
            direction:     "up" = ancestors (what is this a kind of?),
                           "down" = descendants (what kinds of this exist?).
        """
        eid = self.resolve_id(entity_id)
        visited: Set[str] = set()
        result: List[Entity] = []
        queue: deque[str] = deque([eid])

        while queue:
            current = queue.popleft()
            edges = (
                self._graph.in_edges(current, data=True)
                if direction == "down"
                else self._graph.out_edges(current, data=True)
            )
            for edge in edges:
                if direction == "down":
                    other, _, data = edge
                else:
                    _, other, data = edge
                rel: Relationship = data["relationship"]
                if rel.relation_type == relation_type and other not in visited:
                    visited.add(other)
                    result.append(self._entities[other])
                    queue.append(other)

        return result

    def get_instances_of(self, category_id: str) -> List[Entity]:
        """Get all entities that IS_A the given category (children)."""
        return self.get_taxonomy(category_id, RelationType.IS_A, direction="down")

    def get_categories_of(self, entity_id: str) -> List[Entity]:
        """Get all categories this entity IS_A (parents)."""
        return self.get_taxonomy(entity_id, RelationType.IS_A, direction="up")

    # ------------------------------------------------------------------
    # Causal reasoning
    # ------------------------------------------------------------------

    def to_causal_dag(
        self,
        relation_types: Optional[List[RelationType]] = None,
        observed_entity_types: Optional[List[EntityType]] = None,
    ) -> CausalDAG:
        """
        Extract a CausalDAG from the knowledge graph.

        Filters to CAUSES relationships (by default) and maps entity IDs
        to DAG variable names.  The resulting DAG is compatible with
        ``CausalInferenceEngine`` for Pearl's 3-layer analysis.

        Args:
            relation_types:        Which relationship types count as causal
                                   edges.  Default: [CAUSES].
            observed_entity_types: Entity types to mark as observed in the DAG.
                                   Default: [COMMODITY, COUNTRY, MARKET,
                                   ECONOMIC_INDICATOR].  All others are
                                   marked unobserved.
        """
        if relation_types is None:
            relation_types = [RelationType.CAUSES]
        if observed_entity_types is None:
            observed_entity_types = [
                EntityType.COMMODITY,
                EntityType.COUNTRY,
                EntityType.MARKET,
                EntityType.ECONOMIC_INDICATOR,
                EntityType.POLICY,
            ]

        dag = CausalDAG()

        # Collect nodes that participate in causal edges
        causal_rels = []
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if rel.relation_type in relation_types:
                causal_rels.append((u, v, rel))

        # Add nodes
        nodes_in_dag: Set[str] = set()
        for u, v, _ in causal_rels:
            nodes_in_dag.add(u)
            nodes_in_dag.add(v)

        for nid in nodes_in_dag:
            entity = self._entities.get(nid)
            # Entity-level "observed" property overrides the type-based default,
            # allowing latent SCM variables (Supply, Shortage, etc.) to be
            # correctly marked unobserved even if their entity type is in the
            # observed_entity_types list.
            if entity is not None and "observed" in entity.properties:
                observed = bool(entity.properties["observed"])
            else:
                observed = (
                    entity is not None and entity.entity_type in observed_entity_types
                )
            dag.add_node(nid, observed=observed)

        # Add edges
        for u, v, _ in causal_rels:
            dag.add_edge(u, v)

        return dag

    def find_confounders(
        self,
        treatment: str,
        outcome: str,
    ) -> List[Entity]:
        """
        Identify potential confounders of treatment -> outcome.

        A confounder is an entity that has a causal path to both the
        treatment and the outcome (a common cause).  Uses the CAUSES
        sub-graph for the analysis.
        """
        t = self.resolve_id(treatment)
        o = self.resolve_id(outcome)

        # Build causal-only view
        causal_view = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if rel.relation_type == RelationType.CAUSES:
                causal_view.add_edge(u, v)

        if t not in causal_view or o not in causal_view:
            return []

        ancestors_t = nx.ancestors(causal_view, t)
        ancestors_o = nx.ancestors(causal_view, o)
        common = ancestors_t & ancestors_o

        return [self._entities[c] for c in common if c in self._entities]

    def propagate_shock(
        self,
        origin_id: str,
        initial_magnitude: float = 1.0,
        decay: float = 0.5,
        max_depth: int = 5,
        relation_types: Optional[List[RelationType]] = None,
    ) -> ShockTrace:
        """
        Propagate a shock through the knowledge graph using BFS with
        exponential decay.

        At each hop the impact is multiplied by ``decay``.  Useful for
        answering "if China bans graphite exports, what is affected?"

        Args:
            origin_id:      Entity where the shock originates.
            initial_magnitude: Starting shock strength (1.0 = 100%).
            decay:          Per-hop decay factor (0..1).
            max_depth:      Maximum propagation depth.
            relation_types: Which edges to propagate along.
                            Default: [CAUSES, SUPPLIES, EXPORTS_TO, DEPENDS_ON].
        """
        if relation_types is None:
            relation_types = [
                RelationType.CAUSES,
                RelationType.SUPPLIES,
                RelationType.EXPORTS_TO,
                RelationType.DEPENDS_ON,
                RelationType.PRODUCES,
                RelationType.CONSUMES,
                RelationType.ENABLES,
                RelationType.DISRUPTS,
                RelationType.INPUTS,
                RelationType.REGULATES,
            ]

        origin = self.resolve_id(origin_id)
        affected: Dict[str, float] = {}
        paths: Dict[str, List[str]] = {}
        queue: deque[Tuple[str, float, int, List[str]]] = deque(
            [(origin, initial_magnitude, 0, [origin])]
        )
        visited: Set[str] = set()

        while queue:
            current, magnitude, depth, path = queue.popleft()
            if depth > max_depth or magnitude < 1e-6:
                continue

            if current != origin:
                # Keep the highest magnitude path
                if current not in affected or magnitude > affected[current]:
                    affected[current] = magnitude
                    paths[current] = list(path)

            if current in visited:
                continue
            visited.add(current)

            for _, v, data in self._graph.out_edges(current, data=True):
                rel: Relationship = data["relationship"]
                if rel.relation_type in relation_types and v not in visited:
                    # Weight by relationship confidence
                    conf_mult = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4}.get(
                        rel.confidence, 0.7
                    )
                    next_mag = magnitude * decay * conf_mult
                    queue.append((v, next_mag, depth + 1, path + [v]))

        return ShockTrace(
            origin=origin,
            affected=affected,
            paths=paths,
            depth=max_depth,
        )

    def get_shock_origin_candidates(self) -> List[str]:
        """
        Return entity IDs that have at least one outgoing CAUSES edge.
        Useful for UI dropdowns (e.g. "Propagate shock from...").
        """
        origins: Set[str] = set()
        for u, _, data in self._graph.out_edges(data=True):
            rel: Relationship = data["relationship"]
            if rel.relation_type == RelationType.CAUSES:
                origins.add(u)
        return sorted(origins)

    # ------------------------------------------------------------------
    # Trust & provenance scoring
    # ------------------------------------------------------------------

    def trust_score(self, entity_id: str) -> float:
        """
        Compute a trust score (0..1) for an entity based on the provenance
        and confidence of its incoming relationships.

        Entities with more HIGH-confidence, well-sourced relationships
        score higher.  Entities with no relationships return 0.5 (neutral).
        """
        eid = self.resolve_id(entity_id)
        if eid not in self._entities:
            return 0.0

        rels_in = list(self._graph.in_edges(eid, data=True))
        rels_out = list(self._graph.out_edges(eid, data=True))
        all_rels = rels_in + rels_out
        if not all_rels:
            return 0.5

        conf_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
        scores: List[float] = []
        for edge in all_rels:
            rel: Relationship = edge[2]["relationship"]
            conf = conf_map.get(rel.confidence, 0.5)
            has_provenance = 1.0 if rel.provenance else 0.7
            has_mechanism = 1.0 if rel.mechanism else 0.8
            scores.append(conf * has_provenance * has_mechanism)

        return sum(scores) / len(scores)

    def validate_integrity(self) -> List[str]:
        """
        Validate knowledge graph integrity (inspired by TrustKG SHACL validation).

        Returns a list of issues found.  An empty list means the graph is valid.

        Checks:
          - Dangling edges (edges referencing non-existent entities)
          - Isolated entities (no relationships)
          - Circular causal chains (cycles in CAUSES sub-graph)
          - Missing required properties (CAUSES without mechanism)
          - Confidence consistency (unknown confidence values)
        """
        issues: List[str] = []

        # 1. Dangling edges
        for u, v, data in self._graph.edges(data=True):
            if u not in self._entities:
                issues.append(f"Dangling edge source: '{u}' not in entities")
            if v not in self._entities:
                issues.append(f"Dangling edge target: '{v}' not in entities")

        # 2. Isolated entities (warning, not error)
        for eid in self._entities:
            if self._graph.degree(eid) == 0:
                issues.append(f"Isolated entity: '{eid}' has no relationships")

        # 3. Cycles in CAUSES sub-graph
        causal_graph = nx.DiGraph()
        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            if rel.relation_type == RelationType.CAUSES:
                causal_graph.add_edge(u, v)
        try:
            cycles = list(nx.simple_cycles(causal_graph))
            for cycle in cycles:
                issues.append(f"Causal cycle detected: {' → '.join(cycle)}")
        except nx.NetworkXError:
            pass

        # 4. CAUSES edges should have a mechanism
        for u, v, data in self._graph.edges(data=True):
            rel = data["relationship"]
            if rel.relation_type == RelationType.CAUSES and not rel.mechanism:
                issues.append(
                    f"CAUSES edge '{u}' → '{v}' missing mechanism"
                )

        # 5. Confidence values should be valid
        valid_conf = {"HIGH", "MEDIUM", "LOW"}
        for u, v, data in self._graph.edges(data=True):
            rel = data["relationship"]
            if rel.confidence not in valid_conf:
                issues.append(
                    f"Edge '{u}' → '{v}' has unknown confidence: '{rel.confidence}'"
                )

        return issues

    def provenance_report(self) -> Dict[str, Any]:
        """
        Generate a provenance report showing data source coverage.

        Returns a dict with counts of relationships by provenance source,
        confidence distribution, and coverage gaps.
        """
        by_source: Dict[str, int] = {}
        by_confidence: Dict[str, int] = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
        no_provenance: List[str] = []
        no_mechanism: List[str] = []

        for u, v, data in self._graph.edges(data=True):
            rel: Relationship = data["relationship"]
            # Source tracking
            src = rel.provenance or "unattributed"
            by_source[src] = by_source.get(src, 0) + 1

            # Confidence
            if rel.confidence in by_confidence:
                by_confidence[rel.confidence] += 1
            else:
                by_confidence["UNKNOWN"] += 1

            # Coverage gaps
            if not rel.provenance:
                no_provenance.append(f"{u} → {v} ({rel.relation_type.value})")
            if rel.relation_type == RelationType.CAUSES and not rel.mechanism:
                no_mechanism.append(f"{u} → {v}")

        return {
            "total_relationships": self.num_relationships,
            "by_source": dict(sorted(by_source.items(), key=lambda x: -x[1])),
            "confidence_distribution": by_confidence,
            "unattributed_count": len(no_provenance),
            "causes_without_mechanism": no_mechanism,
        }

    # ------------------------------------------------------------------
    # Merge — KGs building off one another
    # ------------------------------------------------------------------

    def merge(
        self,
        other: "CausalKnowledgeGraph",
        confidence_increment: float = 0.1,
    ) -> "CausalKnowledgeGraph":
        """
        Merge *other* into this graph in-place and return self.

        Rules
        -----
        **Entities**: if an entity from *other* already exists (same ID or
        shared alias), its aliases and properties are merged into the
        existing entity.  Unknown entities are added as-is.

        **Relationships**: if an edge with the same (source, target,
        relation_type) already exists, its confidence is reinforced by
        *confidence_increment* (capped at 1.0) and its provenance list
        is extended with the new source.  Unknown edges are added with
        an initial ``confidence_score`` of 0.5 when not already set.

        This means each new document or KG that supports an existing
        causal link makes it stronger, while novel links start at
        medium confidence and accumulate evidence over time.

        Args:
            other: The KG to merge in.
            confidence_increment: How much to boost the numeric
                ``confidence_score`` of a relationship each time it is
                confirmed by an additional source.

        Returns:
            self (for chaining).
        """
        # --- Entities ---
        for entity in other._entities.values():
            canonical = self.resolve_id(entity.id)
            if canonical in self._entities:
                # Merge aliases
                existing = self._entities[canonical]
                new_aliases = [a for a in entity.aliases if a not in existing.aliases]
                existing.aliases.extend(new_aliases)
                for alias in new_aliases:
                    self._alias_map[alias.lower()] = canonical
                # Merge properties (other wins on conflicts so provenance grows)
                for k, v in entity.properties.items():
                    if k not in existing.properties:
                        existing.properties[k] = v
            else:
                self.add_entity(entity)

        # --- Relationships ---
        for _, _, edge_data in other._graph.edges(data=True):
            rel: Relationship = edge_data["relationship"]
            src = self.resolve_id(rel.source_id)
            tgt = self.resolve_id(rel.target_id)
            key = rel.relation_type.value

            if self._graph.has_edge(src, tgt, key=key):
                # Reinforce existing edge
                existing_rel: Relationship = self._graph[src][tgt][key]["relationship"]
                old_score = float(existing_rel.properties.get("confidence_score", 0.5))
                existing_rel.properties["confidence_score"] = min(
                    1.0, old_score + confidence_increment
                )
                # Refresh string label from numeric score
                score = existing_rel.properties["confidence_score"]
                existing_rel.properties["confidence"] = (
                    "HIGH" if score >= 0.7 else ("MEDIUM" if score >= 0.4 else "LOW")
                )
                # Merge provenance sources
                new_src = rel.provenance or rel.properties.get("source_document", "")
                if new_src:
                    sources = existing_rel.properties.get("sources", [])
                    if new_src not in sources:
                        sources.append(new_src)
                    existing_rel.properties["sources"] = sources
                # Accumulate evidence text
                if rel.evidence:
                    existing_evidence = existing_rel.properties.get("evidence", "")
                    if rel.evidence not in existing_evidence:
                        existing_rel.properties["evidence"] = (
                            existing_evidence + "\n" + rel.evidence
                        ).strip()
            else:
                # New edge: set initial confidence_score if absent
                if "confidence_score" not in rel.properties:
                    conf_str = rel.properties.get("confidence", "MEDIUM")
                    rel.properties["confidence_score"] = (
                        0.8 if conf_str == "HIGH" else (0.5 if conf_str == "MEDIUM" else 0.3)
                    )
                # Initialise sources list
                src_doc = rel.provenance or rel.properties.get("source_document", "")
                if src_doc:
                    rel.properties.setdefault("sources", [src_doc])
                rel.source_id = src
                rel.target_id = tgt
                self.add_relationship(rel)

        return self

    def save(self, path: str) -> None:
        """Save this KG to a JSON file (convenience wrapper around to_json)."""
        self.to_json(path=path)

    @classmethod
    def load(cls, path: str) -> "CausalKnowledgeGraph":
        """Load a KG from a JSON file (convenience wrapper around from_json)."""
        return cls.from_json(path)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the knowledge graph to a dict."""
        entities = [e.to_dict() for e in self._entities.values()]
        relationships = []
        for _, _, data in self._graph.edges(data=True):
            relationships.append(data["relationship"].to_dict())
        result: Dict[str, Any] = {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "entity_types": sorted({e.entity_type.value for e in self._entities.values()}),
                "relation_types": sorted(
                    {data["relationship"].relation_type.value for _, _, data in self._graph.edges(data=True)}
                ),
            },
        }
        if self.schema is not None:
            result["schema"] = self.schema.to_dict()
        return result

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialize to JSON string, optionally writing to file."""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, default=str)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json_str)
        return json_str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CausalKnowledgeGraph:
        """Deserialize from a dict."""
        schema = None
        if "schema" in data:
            schema = KGSchema.from_dict(data["schema"])
        kg = cls(schema=schema)
        for e_data in data.get("entities", []):
            kg.add_entity(Entity.from_dict(e_data))
        for r_data in data.get("relationships", []):
            kg.add_relationship(Relationship.from_dict(r_data))
        return kg

    @classmethod
    def from_json(cls, json_str_or_path: str) -> CausalKnowledgeGraph:
        """Deserialize from a JSON string or file path."""
        # Try as file path first (only if it looks like a plausible path)
        if len(json_str_or_path) < 4096 and not json_str_or_path.lstrip().startswith("{"):
            p = Path(json_str_or_path)
            try:
                if p.exists():
                    json_str_or_path = p.read_text()
            except OSError:
                pass
        data = json.loads(json_str_or_path)
        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Import from existing structures
    # ------------------------------------------------------------------

    def import_from_dag_registry(self, registry_path: str) -> int:
        """
        Import causal edges from a discovered DAG registry JSON file
        (the format produced by ``CausalDiscoveryAgent.export_to_dag()``).

        Returns the number of edges imported.
        """
        p = Path(registry_path)
        if not p.exists():
            raise FileNotFoundError(f"Registry file not found: {registry_path}")

        with open(p) as f:
            data = json.load(f)

        count = 0
        for node_name in data.get("nodes", []):
            if node_name not in self._entities:
                self.add_entity(Entity(
                    id=node_name,
                    entity_type=EntityType.ECONOMIC_INDICATOR,
                    properties={"source": "dag_registry"},
                ))

        for edge in data.get("edges", []):
            self.add_relationship(Relationship(
                source_id=edge["from"],
                target_id=edge["to"],
                relation_type=RelationType.CAUSES,
                properties={
                    "mechanism": edge.get("mechanism", ""),
                    "confidence": edge.get("confidence", "MEDIUM"),
                    "evidence": edge.get("evidence", ""),
                    "source_document": edge.get("source", ""),
                    "validated": edge.get("validated", False),
                },
            ))
            count += 1

        return count

    def import_from_supply_chain_network(
        self,
        network: Any,
        mineral: str,
    ) -> int:
        """
        Import country-level trade flows from a ``GlobalSupplyChainNetwork``
        as EXPORTS_TO relationships.

        Returns the number of edges imported.
        """
        G = network.networks.get(mineral)
        if G is None:
            return 0

        # Ensure commodity entity exists
        if mineral not in self._entities:
            self.add_entity(Entity(
                id=mineral,
                entity_type=EntityType.COMMODITY,
            ))

        count = 0
        for u, v, data in G.edges(data=True):
            # Add country entities
            for country in (u, v):
                if country not in self._entities:
                    self.add_entity(Entity(
                        id=country,
                        entity_type=EntityType.COUNTRY,
                    ))

            self.add_relationship(Relationship(
                source_id=u,
                target_id=v,
                relation_type=RelationType.EXPORTS_TO,
                properties={
                    "commodity": mineral,
                    "trade_value": data.get("weight", 0.0),
                    "year": data.get("year"),
                    "quantity": data.get("quantity"),
                },
            ))
            count += 1

        return count

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the knowledge graph."""
        type_counts = {}
        for e in self._entities.values():
            type_counts[e.entity_type.value] = type_counts.get(e.entity_type.value, 0) + 1

        rel_counts = {}
        for _, _, data in self._graph.edges(data=True):
            rt = data["relationship"].relation_type.value
            rel_counts[rt] = rel_counts.get(rt, 0) + 1

        lines = [
            f"CausalKnowledgeGraph: {self.num_entities} entities, {self.num_relationships} relationships",
            "",
            "Entity types:",
        ]
        for t, c in sorted(type_counts.items()):
            lines.append(f"  {t}: {c}")
        lines.append("")
        lines.append("Relationship types:")
        for t, c in sorted(rel_counts.items()):
            lines.append(f"  {t}: {c}")

        return "\n".join(lines)


# ============================================================================
# Pre-built templates
# ============================================================================


def build_critical_minerals_kg(
    with_schema: bool = False,
) -> CausalKnowledgeGraph:
    """
    Build a starter knowledge graph for critical minerals supply chains.

    Includes major commodities, producing countries, key policies,
    downstream industries/technologies, events, risk factors, taxonomy
    (IS_A hierarchies), temporal annotations, and known causal relationships.

    Args:
        with_schema: If True, attach the supply chain schema and enable
                     validation on future add_relationship() calls.
    """
    schema = build_supply_chain_schema() if with_schema else None
    kg = CausalKnowledgeGraph(schema=schema)

    # ------- Commodities -------
    commodities = [
        ("graphite", {"criticality": "high", "uses": ["batteries", "steel", "lubricants"]}),
        ("lithium", {"criticality": "high", "uses": ["batteries", "ceramics", "pharmaceuticals"]}),
        ("cobalt", {"criticality": "high", "uses": ["batteries", "superalloys", "catalysts"]}),
        ("rare_earths", {"criticality": "high", "uses": ["magnets", "electronics", "defense"]}),
        ("copper", {"criticality": "medium", "uses": ["wiring", "electronics", "construction"]}),
        ("nickel", {"criticality": "medium", "uses": ["batteries", "stainless_steel", "alloys"]}),
        ("manganese", {"criticality": "medium", "uses": ["steel", "batteries", "chemicals"]}),
        ("tungsten", {"criticality": "high", "uses": ["cutting_tools", "defense", "electronics"]}),
        ("titanium", {"criticality": "medium", "uses": ["aerospace", "medical", "defense"]}),
        ("gallium", {"criticality": "high", "uses": ["semiconductors", "LEDs", "solar"]}),
        ("germanium", {"criticality": "high", "uses": ["fiber_optics", "infrared", "semiconductors"]}),
        ("antimony", {"criticality": "high", "uses": ["flame_retardants", "batteries", "defense"]}),
        ("vanadium", {"criticality": "medium", "uses": ["steel", "batteries", "aerospace"]}),
        ("indium", {"criticality": "high", "uses": ["displays", "solar", "semiconductors"]}),
        ("tellurium", {"criticality": "high", "uses": ["solar", "thermoelectrics", "metallurgy"]}),
        ("platinum", {"criticality": "high", "uses": ["catalysts", "electronics", "jewelry"]}),
        ("tantalum", {"criticality": "high", "uses": ["capacitors", "medical", "aerospace"]}),
        ("niobium", {"criticality": "high", "uses": ["steel", "superconductors", "aerospace"]}),
        ("beryllium", {"criticality": "high", "uses": ["aerospace", "defense", "electronics"]}),
        ("cesium", {"criticality": "high", "uses": ["drilling", "atomic_clocks", "research"]}),
    ]
    for cid, props in commodities:
        kg.add_entity(Entity(id=cid, entity_type=EntityType.COMMODITY, properties=props))

    # ------- Countries -------
    countries = [
        ("china", {"region": "asia", "role": "producer"}),
        ("usa", {"region": "north_america", "role": "consumer"}),
        ("australia", {"region": "oceania", "role": "producer"}),
        ("canada", {"region": "north_america", "role": "producer"}),
        ("chile", {"region": "south_america", "role": "producer"}),
        ("drc", {"region": "africa", "role": "producer", "aliases": ["congo"]}),
        ("south_africa", {"region": "africa", "role": "producer"}),
        ("russia", {"region": "europe_asia", "role": "producer"}),
        ("brazil", {"region": "south_america", "role": "producer"}),
        ("india", {"region": "asia", "role": "producer_consumer"}),
        ("japan", {"region": "asia", "role": "consumer_processor"}),
        ("south_korea", {"region": "asia", "role": "consumer_processor"}),
        ("germany", {"region": "europe", "role": "consumer"}),
        ("indonesia", {"region": "asia", "role": "producer"}),
        ("philippines", {"region": "asia", "role": "producer"}),
        ("mozambique", {"region": "africa", "role": "producer"}),
        ("mexico", {"region": "north_america", "role": "producer"}),
        ("madagascar", {"region": "africa", "role": "producer"}),
    ]
    for cid, props in countries:
        aliases = props.pop("aliases", [])
        kg.add_entity(Entity(
            id=cid, entity_type=EntityType.COUNTRY,
            properties=props, aliases=aliases,
        ))

    # ------- Industries / Technologies -------
    industries = [
        ("ev_batteries", EntityType.INDUSTRY, {"sector": "automotive"}),
        ("grid_storage", EntityType.INDUSTRY, {"sector": "energy"}),
        ("semiconductors", EntityType.INDUSTRY, {"sector": "electronics"}),
        ("steel_production", EntityType.INDUSTRY, {"sector": "heavy_industry"}),
        ("aerospace", EntityType.INDUSTRY, {"sector": "defense"}),
        ("solar_panels", EntityType.TECHNOLOGY, {"sector": "energy"}),
        ("wind_turbines", EntityType.TECHNOLOGY, {"sector": "energy"}),
        ("5g_infrastructure", EntityType.TECHNOLOGY, {"sector": "telecom"}),
        ("defense_systems", EntityType.INDUSTRY, {"sector": "defense"}),
    ]
    for iid, itype, props in industries:
        kg.add_entity(Entity(id=iid, entity_type=itype, properties=props))

    # ------- Key Policies / Events -------
    policies = [
        ("china_export_controls", EntityType.POLICY, {
            "country": "china",
            "description": "Export restrictions on critical minerals",
            "years": "2010-present",
        }),
        ("us_inflation_reduction_act", EntityType.POLICY, {
            "country": "usa",
            "description": "Subsidies for domestic critical mineral processing",
            "year": 2022,
        }),
        ("eu_critical_raw_materials_act", EntityType.POLICY, {
            "country": "eu",
            "description": "Targets for domestic sourcing and recycling",
            "year": 2023,
        }),
        ("drc_mining_code_reform", EntityType.POLICY, {
            "country": "drc",
            "description": "Increased royalties and state ownership requirements",
            "year": 2018,
        }),
        ("indonesia_nickel_ore_ban", EntityType.POLICY, {
            "country": "indonesia",
            "description": "Ban on unprocessed nickel ore exports",
            "year": 2020,
        }),
    ]
    for pid, ptype, props in policies:
        kg.add_entity(Entity(id=pid, entity_type=ptype, properties=props))

    # ------- Economic Indicators -------
    indicators = [
        ("global_ev_demand", EntityType.ECONOMIC_INDICATOR, {"unit": "vehicles/year"}),
        ("global_gdp_growth", EntityType.ECONOMIC_INDICATOR, {"unit": "percent"}),
        ("battery_cost_per_kwh", EntityType.ECONOMIC_INDICATOR, {"unit": "USD/kWh"}),
    ]
    for iid, itype, props in indicators:
        kg.add_entity(Entity(id=iid, entity_type=itype, properties=props))

    # ------- PRODUCES relationships -------
    production = [
        ("china", "graphite", {"share": 0.65, "type": "natural_and_synthetic"}),
        ("china", "rare_earths", {"share": 0.60}),
        ("china", "gallium", {"share": 0.80}),
        ("china", "germanium", {"share": 0.60}),
        ("china", "tungsten", {"share": 0.80}),
        ("china", "antimony", {"share": 0.55}),
        ("china", "indium", {"share": 0.55}),
        ("china", "vanadium", {"share": 0.55}),
        ("australia", "lithium", {"share": 0.50}),
        ("chile", "lithium", {"share": 0.25}),
        ("chile", "copper", {"share": 0.27}),
        ("drc", "cobalt", {"share": 0.70}),
        ("south_africa", "platinum", {"share": 0.70}),
        ("south_africa", "manganese", {"share": 0.30}),
        ("brazil", "niobium", {"share": 0.90}),
        ("mozambique", "graphite", {"share": 0.10}),
        ("madagascar", "graphite", {"share": 0.06}),
        ("indonesia", "nickel", {"share": 0.37}),
        ("russia", "nickel", {"share": 0.10}),
        ("russia", "titanium", {"share": 0.20}),
        ("usa", "beryllium", {"share": 0.65}),
        ("canada", "cesium", {"share": 0.15}),
    ]
    for country, commodity, props in production:
        kg.add_relationship(Relationship(
            source_id=country, target_id=commodity,
            relation_type=RelationType.PRODUCES,
            properties={**props, "confidence": "HIGH"},
        ))

    # ------- CONSUMES relationships -------
    consumption = [
        ("ev_batteries", "lithium"), ("ev_batteries", "cobalt"),
        ("ev_batteries", "nickel"), ("ev_batteries", "graphite"),
        ("ev_batteries", "manganese"),
        ("grid_storage", "lithium"), ("grid_storage", "vanadium"),
        ("semiconductors", "gallium"), ("semiconductors", "germanium"),
        ("semiconductors", "indium"),
        ("steel_production", "manganese"), ("steel_production", "niobium"),
        ("steel_production", "vanadium"), ("steel_production", "tungsten"),
        ("solar_panels", "tellurium"), ("solar_panels", "indium"),
        ("solar_panels", "gallium"),
        ("wind_turbines", "rare_earths"), ("wind_turbines", "copper"),
        ("aerospace", "titanium"), ("aerospace", "beryllium"),
        ("aerospace", "niobium"),
        ("defense_systems", "tungsten"), ("defense_systems", "antimony"),
        ("defense_systems", "beryllium"), ("defense_systems", "rare_earths"),
    ]
    for industry, commodity in consumption:
        kg.add_relationship(Relationship(
            source_id=industry, target_id=commodity,
            relation_type=RelationType.CONSUMES,
            properties={"confidence": "HIGH"},
        ))

    # ------- SUBSTITUTES relationships -------
    substitutions = [
        ("lithium", "nickel", {"context": "battery_cathode", "partial": True}),
        ("cobalt", "nickel", {"context": "battery_cathode", "partial": True}),
        ("graphite", "silicon", {"context": "battery_anode", "partial": True}),
    ]
    for src, tgt, props in substitutions:
        # Add silicon if not present
        if tgt not in kg._entities:
            kg.add_entity(Entity(id=tgt, entity_type=EntityType.COMMODITY))
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.SUBSTITUTES,
            properties={**props, "confidence": "MEDIUM"},
        ))

    # ------- REGULATES relationships -------
    regulations = [
        ("china_export_controls", "graphite"),
        ("china_export_controls", "rare_earths"),
        ("china_export_controls", "gallium"),
        ("china_export_controls", "germanium"),
        ("indonesia_nickel_ore_ban", "nickel"),
        ("drc_mining_code_reform", "cobalt"),
    ]
    for policy, commodity in regulations:
        kg.add_relationship(Relationship(
            source_id=policy, target_id=commodity,
            relation_type=RelationType.REGULATES,
            properties={"confidence": "HIGH"},
        ))

    # ------- DEPENDS_ON relationships -------
    dependencies = [
        ("usa", "china", {"commodities": ["graphite", "rare_earths", "gallium", "germanium"]}),
        ("usa", "drc", {"commodities": ["cobalt"]}),
        ("usa", "chile", {"commodities": ["lithium", "copper"]}),
        ("japan", "china", {"commodities": ["rare_earths", "graphite"]}),
        ("south_korea", "china", {"commodities": ["graphite", "rare_earths"]}),
        ("germany", "china", {"commodities": ["rare_earths", "gallium"]}),
    ]
    for consumer, supplier, props in dependencies:
        kg.add_relationship(Relationship(
            source_id=consumer, target_id=supplier,
            relation_type=RelationType.DEPENDS_ON,
            properties={**props, "confidence": "HIGH"},
        ))

    # ------- CAUSES relationships (the causal layer) -------
    causal = [
        ("china_export_controls", "graphite", {
            "mechanism": "export restrictions reduce global supply",
            "confidence": "HIGH",
        }),
        ("china_export_controls", "rare_earths", {
            "mechanism": "export quotas constrain availability",
            "confidence": "HIGH",
        }),
        ("china_export_controls", "gallium", {
            "mechanism": "licensing requirements limit exports",
            "confidence": "HIGH",
        }),
        ("china_export_controls", "germanium", {
            "mechanism": "licensing requirements limit exports",
            "confidence": "HIGH",
        }),
        ("global_ev_demand", "lithium", {
            "mechanism": "EV growth drives battery material demand",
            "confidence": "HIGH",
        }),
        ("global_ev_demand", "cobalt", {
            "mechanism": "EV growth drives battery material demand",
            "confidence": "HIGH",
        }),
        ("global_ev_demand", "graphite", {
            "mechanism": "EV growth drives anode material demand",
            "confidence": "HIGH",
        }),
        ("global_ev_demand", "nickel", {
            "mechanism": "EV growth drives cathode material demand",
            "confidence": "HIGH",
        }),
        ("global_gdp_growth", "copper", {
            "mechanism": "economic growth increases construction and electrification demand",
            "confidence": "MEDIUM",
        }),
        ("indonesia_nickel_ore_ban", "nickel", {
            "mechanism": "ore export ban reduces raw material availability for non-Indonesian smelters",
            "confidence": "HIGH",
        }),
        ("drc_mining_code_reform", "cobalt", {
            "mechanism": "higher royalties increase production costs and reduce investment",
            "confidence": "MEDIUM",
        }),
        ("us_inflation_reduction_act", "lithium", {
            "mechanism": "subsidies incentivize domestic mining and processing",
            "confidence": "HIGH",
        }),
    ]
    for src, tgt, props in causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # ------- ENABLES relationships -------
    enables = [
        ("lithium", "ev_batteries", {"role": "cathode_material"}),
        ("graphite", "ev_batteries", {"role": "anode_material"}),
        ("cobalt", "ev_batteries", {"role": "cathode_stabilizer"}),
        ("rare_earths", "wind_turbines", {"role": "permanent_magnets"}),
        ("gallium", "semiconductors", {"role": "substrate_material"}),
        ("copper", "wind_turbines", {"role": "wiring_and_generators"}),
    ]
    for src, tgt, props in enables:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.ENABLES,
            properties={**props, "confidence": "HIGH"},
        ))

    # ------- Regions (for PART_OF hierarchy) -------
    regions = [
        ("asia_pacific", {"label": "Asia-Pacific"}),
        ("americas", {"label": "Americas"}),
        ("europe", {"label": "Europe"}),
        ("africa", {"label": "Africa"}),
        ("middle_east", {"label": "Middle East"}),
    ]
    for rid, props in regions:
        kg.add_entity(Entity(id=rid, entity_type=EntityType.REGION, properties=props))

    # Country -> Region (PART_OF)
    country_regions = [
        ("china", "asia_pacific"), ("japan", "asia_pacific"),
        ("south_korea", "asia_pacific"), ("india", "asia_pacific"),
        ("indonesia", "asia_pacific"), ("philippines", "asia_pacific"),
        ("australia", "asia_pacific"),
        ("usa", "americas"), ("canada", "americas"),
        ("chile", "americas"), ("brazil", "americas"), ("mexico", "americas"),
        ("germany", "europe"), ("russia", "europe"),
        ("drc", "africa"), ("south_africa", "africa"),
        ("mozambique", "africa"), ("madagascar", "africa"),
    ]
    for country, region in country_regions:
        kg.add_relationship(Relationship(
            source_id=country, target_id=region,
            relation_type=RelationType.PART_OF,
        ))

    # ------- Taxonomy (IS_A) for commodities -------
    # Category entities
    kg.add_entity(Entity(
        id="battery_material", entity_type=EntityType.COMMODITY,
        properties={"is_category": True},
    ))
    kg.add_entity(Entity(
        id="strategic_mineral", entity_type=EntityType.COMMODITY,
        properties={"is_category": True},
    ))
    kg.add_entity(Entity(
        id="semiconductor_material", entity_type=EntityType.COMMODITY,
        properties={"is_category": True},
    ))

    # IS_A relationships
    for mineral in ["lithium", "cobalt", "nickel", "graphite", "manganese"]:
        kg.add_relationship(Relationship(
            mineral, "battery_material", RelationType.IS_A,
        ))
    for mineral in ["tungsten", "antimony", "beryllium", "rare_earths", "titanium"]:
        kg.add_relationship(Relationship(
            mineral, "strategic_mineral", RelationType.IS_A,
        ))
    for mineral in ["gallium", "germanium", "indium"]:
        kg.add_relationship(Relationship(
            mineral, "semiconductor_material", RelationType.IS_A,
        ))

    # ------- Historical events (with temporal data) -------
    events = [
        ("china_rare_earth_crisis_2010", EntityType.EVENT, {
            "description": "China cut rare earth export quotas by 40%, prices spiked 10x",
            "impact": "severe",
            "provenance": "USGS Mineral Commodity Summaries 2011",
        }, "2010", "2012"),
        ("covid_supply_disruption_2020", EntityType.EVENT, {
            "description": "Global supply chain disruptions due to COVID-19 pandemic",
            "impact": "severe",
            "provenance": "IEA Critical Minerals Report 2021",
        }, "2020", "2022"),
        ("russia_ukraine_conflict_2022", EntityType.EVENT, {
            "description": "Sanctions and supply disruptions from Russia-Ukraine conflict",
            "impact": "high",
            "provenance": "IEA Energy Security Report 2022",
        }, "2022", None),
        ("us_china_trade_war_2018", EntityType.EVENT, {
            "description": "Escalating tariffs and trade restrictions between US and China",
            "impact": "high",
            "provenance": "USITC reports",
        }, "2018", None),
        ("chile_lithium_nationalization_2023", EntityType.EVENT, {
            "description": "Chile moves to nationalize lithium production",
            "impact": "medium",
            "provenance": "Reuters 2023",
        }, "2023", None),
    ]
    for eid, etype, props, start, end in events:
        kg.add_entity(Entity(
            id=eid, entity_type=etype, properties=props,
            start_date=start, end_date=end,
        ))

    # Events disrupt commodities/countries
    event_disruptions = [
        ("china_rare_earth_crisis_2010", "rare_earths", {
            "mechanism": "export quota cuts reduced global supply by 40%",
            "confidence": "HIGH",
            "provenance": "USGS",
        }, "2010", "2012"),
        ("covid_supply_disruption_2020", "copper", {
            "mechanism": "mine closures and logistics disruptions",
            "confidence": "HIGH",
        }, "2020", "2021"),
        ("covid_supply_disruption_2020", "cobalt", {
            "mechanism": "DRC mine closures and shipping delays",
            "confidence": "HIGH",
        }, "2020", "2021"),
        ("russia_ukraine_conflict_2022", "nickel", {
            "mechanism": "sanctions on Russian nickel exports",
            "confidence": "HIGH",
        }, "2022", None),
        ("russia_ukraine_conflict_2022", "titanium", {
            "mechanism": "Russia supplies 20% of global titanium; sanctions disrupt",
            "confidence": "HIGH",
        }, "2022", None),
        ("us_china_trade_war_2018", "rare_earths", {
            "mechanism": "threat of Chinese rare earth export restrictions as leverage",
            "confidence": "MEDIUM",
        }, "2018", None),
        ("chile_lithium_nationalization_2023", "lithium", {
            "mechanism": "nationalization creates investment uncertainty",
            "confidence": "MEDIUM",
        }, "2023", None),
    ]
    for event, commodity, props, start, end in event_disruptions:
        kg.add_relationship(Relationship(
            source_id=event, target_id=commodity,
            relation_type=RelationType.DISRUPTS,
            properties=props,
            start_date=start, end_date=end,
        ))

    # Event temporal sequencing
    kg.add_relationship(Relationship(
        "china_rare_earth_crisis_2010", "us_china_trade_war_2018",
        RelationType.PRECEDED_BY,
        properties={"note": "2010 crisis foreshadowed later trade conflicts"},
    ))

    # ------- Risk factors -------
    risks = [
        ("geographic_concentration_risk", EntityType.RISK_FACTOR, {
            "description": "Over-reliance on a single country for supply",
            "severity": "high",
        }),
        ("geopolitical_risk", EntityType.RISK_FACTOR, {
            "description": "Political instability or conflict affecting supply",
            "severity": "high",
        }),
        ("environmental_regulation_risk", EntityType.RISK_FACTOR, {
            "description": "Stricter environmental rules increasing costs or limiting output",
            "severity": "medium",
        }),
        ("demand_volatility_risk", EntityType.RISK_FACTOR, {
            "description": "Rapid demand changes from technology transitions",
            "severity": "medium",
        }),
    ]
    for rid, rtype, props in risks:
        kg.add_entity(Entity(id=rid, entity_type=rtype, properties=props))

    # Risks disrupt commodities
    kg.add_relationship(Relationship(
        "geographic_concentration_risk", "rare_earths", RelationType.DISRUPTS,
        properties={"mechanism": "60%+ from China creates single-point-of-failure", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "geographic_concentration_risk", "cobalt", RelationType.DISRUPTS,
        properties={"mechanism": "70% from DRC, politically unstable", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "geopolitical_risk", "graphite", RelationType.DISRUPTS,
        properties={"mechanism": "China dominance + export control risk", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "demand_volatility_risk", "lithium", RelationType.DISRUPTS,
        properties={"mechanism": "EV adoption pace uncertainty creates boom/bust cycles", "confidence": "MEDIUM"},
    ))

    # Policies mitigate risks
    kg.add_relationship(Relationship(
        "us_inflation_reduction_act", "geographic_concentration_risk", RelationType.MITIGATES,
        properties={"mechanism": "incentivizes domestic/allied supply diversification", "confidence": "MEDIUM"},
        start_date="2022",
    ))
    kg.add_relationship(Relationship(
        "eu_critical_raw_materials_act", "geographic_concentration_risk", RelationType.MITIGATES,
        properties={"mechanism": "sets domestic sourcing targets", "confidence": "MEDIUM"},
        start_date="2023",
    ))

    # ------- EXPORTS_TO relationships (key trade corridors) -------
    trade_flows = [
        ("china", "usa", {"commodities": ["graphite", "rare_earths", "gallium", "germanium"],
                          "confidence": "HIGH"}),
        ("china", "japan", {"commodities": ["graphite", "rare_earths"],
                            "confidence": "HIGH"}),
        ("china", "south_korea", {"commodities": ["graphite", "rare_earths"],
                                  "confidence": "HIGH"}),
        ("china", "germany", {"commodities": ["rare_earths", "gallium"],
                              "confidence": "HIGH"}),
        ("china", "india", {"commodities": ["graphite", "rare_earths"],
                            "confidence": "MEDIUM"}),
        ("australia", "china", {"commodities": ["lithium"],
                                "confidence": "HIGH"}),
        ("australia", "japan", {"commodities": ["lithium"],
                                "confidence": "HIGH"}),
        ("australia", "south_korea", {"commodities": ["lithium"],
                                      "confidence": "HIGH"}),
        ("chile", "china", {"commodities": ["lithium", "copper"],
                            "confidence": "HIGH"}),
        ("chile", "usa", {"commodities": ["copper", "lithium"],
                          "confidence": "HIGH"}),
        ("drc", "china", {"commodities": ["cobalt"],
                          "confidence": "HIGH"}),
        ("drc", "south_korea", {"commodities": ["cobalt"],
                                "confidence": "MEDIUM"}),
        ("south_africa", "china", {"commodities": ["manganese", "platinum"],
                                   "confidence": "HIGH"}),
        ("south_africa", "usa", {"commodities": ["platinum", "manganese"],
                                 "confidence": "HIGH"}),
        ("south_africa", "germany", {"commodities": ["platinum"],
                                     "confidence": "MEDIUM"}),
        ("brazil", "usa", {"commodities": ["niobium"],
                           "confidence": "HIGH"}),
        ("brazil", "china", {"commodities": ["niobium"],
                             "confidence": "HIGH"}),
        ("indonesia", "china", {"commodities": ["nickel"],
                                "confidence": "HIGH"}),
        ("indonesia", "japan", {"commodities": ["nickel"],
                                "confidence": "MEDIUM"}),
        ("russia", "china", {"commodities": ["nickel", "titanium"],
                             "confidence": "MEDIUM"}),
        ("mozambique", "china", {"commodities": ["graphite"],
                                 "confidence": "HIGH"}),
        ("madagascar", "usa", {"commodities": ["graphite"],
                               "confidence": "MEDIUM"}),
    ]
    for exporter, importer, props in trade_flows:
        kg.add_relationship(Relationship(
            source_id=exporter, target_id=importer,
            relation_type=RelationType.EXPORTS_TO,
            properties=props,
        ))

    # ------- Cascading CAUSES edges (supply chain causal chains) -------
    # Commodity supply → industry impact
    commodity_industry_causal = [
        ("graphite", "ev_batteries", {
            "mechanism": "graphite supply disruption increases anode costs for battery manufacturing",
            "confidence": "HIGH",
        }),
        ("lithium", "ev_batteries", {
            "mechanism": "lithium supply constraints raise cathode material costs",
            "confidence": "HIGH",
        }),
        ("cobalt", "ev_batteries", {
            "mechanism": "cobalt scarcity increases battery cathode costs and slows EV production",
            "confidence": "HIGH",
        }),
        ("nickel", "ev_batteries", {
            "mechanism": "nickel supply disruption affects high-energy-density battery production",
            "confidence": "HIGH",
        }),
        ("manganese", "ev_batteries", {
            "mechanism": "manganese supply disruption affects LMO/NMC cathode production",
            "confidence": "MEDIUM",
        }),
        ("lithium", "grid_storage", {
            "mechanism": "lithium cost increases raise grid-scale battery prices",
            "confidence": "HIGH",
        }),
        ("vanadium", "grid_storage", {
            "mechanism": "vanadium supply disruption impacts redox flow battery deployment",
            "confidence": "MEDIUM",
        }),
        ("gallium", "semiconductors", {
            "mechanism": "gallium restriction disrupts GaAs/GaN chip production",
            "confidence": "HIGH",
        }),
        ("germanium", "semiconductors", {
            "mechanism": "germanium restriction impacts fiber optic and IR sensor production",
            "confidence": "HIGH",
        }),
        ("indium", "semiconductors", {
            "mechanism": "indium scarcity affects ITO display and solar cell production",
            "confidence": "MEDIUM",
        }),
        ("manganese", "steel_production", {
            "mechanism": "manganese is essential for steel deoxidation and alloying",
            "confidence": "HIGH",
        }),
        ("niobium", "steel_production", {
            "mechanism": "niobium supply disruption impacts high-strength low-alloy steel production",
            "confidence": "HIGH",
        }),
        ("tungsten", "steel_production", {
            "mechanism": "tungsten supply affects tool steel and high-speed steel production",
            "confidence": "MEDIUM",
        }),
        ("rare_earths", "wind_turbines", {
            "mechanism": "rare earth supply disruption impacts permanent magnet availability for turbines",
            "confidence": "HIGH",
        }),
        ("copper", "wind_turbines", {
            "mechanism": "copper cost increases raise wind turbine generator and wiring costs",
            "confidence": "HIGH",
        }),
        ("tellurium", "solar_panels", {
            "mechanism": "tellurium scarcity limits CdTe thin-film solar panel production",
            "confidence": "HIGH",
        }),
        ("indium", "solar_panels", {
            "mechanism": "indium supply disruption affects CIGS solar cell production",
            "confidence": "MEDIUM",
        }),
        ("titanium", "aerospace", {
            "mechanism": "titanium supply disruption affects airframe and engine manufacturing",
            "confidence": "HIGH",
        }),
        ("beryllium", "aerospace", {
            "mechanism": "beryllium scarcity impacts satellite and aerospace component production",
            "confidence": "HIGH",
        }),
        ("niobium", "aerospace", {
            "mechanism": "niobium supply affects superalloy jet engine component production",
            "confidence": "MEDIUM",
        }),
        ("tungsten", "defense_systems", {
            "mechanism": "tungsten supply disruption impacts armor-piercing ammunition and tooling",
            "confidence": "HIGH",
        }),
        ("antimony", "defense_systems", {
            "mechanism": "antimony scarcity affects ammunition primers and flame retardant supply",
            "confidence": "HIGH",
        }),
        ("rare_earths", "defense_systems", {
            "mechanism": "rare earth restrictions impact precision-guided munitions and radar systems",
            "confidence": "HIGH",
        }),
        ("gallium", "5g_infrastructure", {
            "mechanism": "gallium restriction disrupts GaN power amplifier production for 5G base stations",
            "confidence": "HIGH",
        }),
        ("germanium", "5g_infrastructure", {
            "mechanism": "germanium supply affects fiber optic infrastructure for 5G backhaul",
            "confidence": "MEDIUM",
        }),
    ]
    for src, tgt, props in commodity_industry_causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # Event → industry cascading effects
    event_industry_causal = [
        ("china_rare_earth_crisis_2010", "wind_turbines", {
            "mechanism": "rare earth price spike 10x increased permanent magnet costs for turbines",
            "confidence": "HIGH",
        }),
        ("china_rare_earth_crisis_2010", "defense_systems", {
            "mechanism": "rare earth supply cuts threatened precision-guided munitions production",
            "confidence": "HIGH",
        }),
        ("covid_supply_disruption_2020", "ev_batteries", {
            "mechanism": "mine closures and logistics disruptions delayed battery material deliveries",
            "confidence": "HIGH",
        }),
        ("covid_supply_disruption_2020", "semiconductors", {
            "mechanism": "pandemic disrupted semiconductor material supply chains globally",
            "confidence": "HIGH",
        }),
        ("russia_ukraine_conflict_2022", "aerospace", {
            "mechanism": "titanium supply disruption from Russia affected Boeing and Airbus production",
            "confidence": "HIGH",
        }),
        ("us_china_trade_war_2018", "semiconductors", {
            "mechanism": "trade restrictions disrupted semiconductor material supply chains",
            "confidence": "MEDIUM",
        }),
        ("us_china_trade_war_2018", "ev_batteries", {
            "mechanism": "tariffs on Chinese graphite increased US battery manufacturing costs",
            "confidence": "MEDIUM",
        }),
    ]
    for src, tgt, props in event_industry_causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # Policy → industry causal effects
    policy_industry_causal = [
        ("china_export_controls", "ev_batteries", {
            "mechanism": "graphite/rare earth export restrictions raise global battery material costs",
            "confidence": "HIGH",
        }),
        ("china_export_controls", "semiconductors", {
            "mechanism": "gallium/germanium export controls disrupt chip production outside China",
            "confidence": "HIGH",
        }),
        ("china_export_controls", "defense_systems", {
            "mechanism": "rare earth/antimony export controls threaten defense supply chains",
            "confidence": "HIGH",
        }),
        ("indonesia_nickel_ore_ban", "ev_batteries", {
            "mechanism": "ore export ban forces battery-grade nickel processing shift",
            "confidence": "MEDIUM",
        }),
        ("us_inflation_reduction_act", "ev_batteries", {
            "mechanism": "domestic sourcing requirements reshape battery supply chain geography",
            "confidence": "HIGH",
        }),
        ("drc_mining_code_reform", "ev_batteries", {
            "mechanism": "increased cobalt royalties raise battery cathode costs",
            "confidence": "MEDIUM",
        }),
    ]
    for src, tgt, props in policy_industry_causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # Country dependency → economic indicator causal links
    macro_causal = [
        ("global_ev_demand", "ev_batteries", {
            "mechanism": "EV demand growth drives battery manufacturing expansion",
            "confidence": "HIGH",
        }),
        ("global_ev_demand", "grid_storage", {
            "mechanism": "EV battery cost reductions enable grid storage deployment",
            "confidence": "MEDIUM",
        }),
        ("global_gdp_growth", "steel_production", {
            "mechanism": "GDP growth drives construction and infrastructure steel demand",
            "confidence": "HIGH",
        }),
        ("battery_cost_per_kwh", "ev_batteries", {
            "mechanism": "battery cost determines EV price competitiveness and adoption rate",
            "confidence": "HIGH",
        }),
        ("battery_cost_per_kwh", "grid_storage", {
            "mechanism": "battery cost determines grid storage economic viability",
            "confidence": "HIGH",
        }),
    ]
    for src, tgt, props in macro_causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # ------- Temporal annotations on policies -------
    # Update existing policy entities with start_dates
    for pid, start in [
        ("china_export_controls", "2010"),
        ("us_inflation_reduction_act", "2022"),
        ("eu_critical_raw_materials_act", "2023"),
        ("drc_mining_code_reform", "2018"),
        ("indonesia_nickel_ore_ban", "2020"),
    ]:
        entity = kg.get_entity(pid)
        if entity:
            entity.start_date = start

    # ------- Structural model variables (for CausalDAG / CausalInferenceEngine) -------
    # These are the canonical DAG node names used by GraphiteSupplyChainDAG and
    # _NODE_TO_SHOCK. Adding them to the KG allows to_causal_dag() to produce
    # a CausalDAG directly usable by CausalInferenceEngine.
    scm_observed = [
        ("ExportPolicy",  EntityType.POLICY,             {"description": "Export quota / restriction level (0=free, 1=complete ban)"}),
        ("TradeValue",    EntityType.ECONOMIC_INDICATOR, {"description": "Bilateral trade flow value (USD)"}),
        ("Price",         EntityType.ECONOMIC_INDICATOR, {"description": "Market spot price of commodity (USD/tonne)"}),
        ("Demand",        EntityType.ECONOMIC_INDICATOR, {"description": "Domestic/downstream demand (tonnes/year)"}),
        ("GlobalDemand",  EntityType.ECONOMIC_INDICATOR, {"description": "Global macroeconomic demand driver (index)"}),
    ]
    scm_unobserved = [
        ("Supply",    EntityType.ECONOMIC_INDICATOR, {"description": "Effective supply reaching market (tonnes/year)", "observed": False}),
        ("Shortage",  EntityType.ECONOMIC_INDICATOR, {"description": "Supply-demand imbalance (tight > 0 = shortage)", "observed": False}),
        ("Inventory", EntityType.ECONOMIC_INDICATOR, {"description": "Stock / inventory level (tonnes)", "observed": False}),
        ("Capacity",  EntityType.ECONOMIC_INDICATOR, {"description": "Production capacity (tonnes/year)", "observed": False}),
    ]
    for eid, etype, props in scm_observed + scm_unobserved:
        kg.add_entity(Entity(id=eid, entity_type=etype, properties=props))

    # CAUSES edges forming the graphite supply chain structural causal model (SCM)
    # This mirrors GraphiteSupplyChainDAG._build_structure() at the KG level,
    # so that to_causal_dag() produces the full structural DAG.
    scm_causal = [
        ("ExportPolicy", "TradeValue", {
            "mechanism": "Export quotas directly reduce traded volumes",
            "confidence": "HIGH",
        }),
        ("ExportPolicy", "Supply", {
            "mechanism": "Restrictions constrain effective supply reaching importers",
            "confidence": "HIGH",
        }),
        ("GlobalDemand", "Demand", {
            "mechanism": "Macro growth (GDP, EV adoption) drives downstream commodity demand",
            "confidence": "HIGH",
        }),
        ("Capacity", "Supply", {
            "mechanism": "Mining and processing capacity sets upper bound on supply",
            "confidence": "HIGH",
        }),
        ("TradeValue", "Inventory", {
            "mechanism": "Trade flows replenish importer inventories",
            "confidence": "HIGH",
        }),
        ("Inventory", "Supply", {
            "mechanism": "Drawdown of inventory buffers effective supply",
            "confidence": "HIGH",
        }),
        ("Supply", "Shortage", {
            "mechanism": "Lower supply relative to demand increases shortage signal",
            "confidence": "HIGH",
        }),
        ("Demand", "Shortage", {
            "mechanism": "Higher demand relative to supply increases shortage signal",
            "confidence": "HIGH",
        }),
        ("Shortage", "Price", {
            "mechanism": "Market price adjusts upward under supply shortage via arbitrage",
            "confidence": "HIGH",
        }),
        # Policy → broader supply chain via environmental compliance
        ("EnvironmentalRegulation", "Capacity", {
            "mechanism": "Stricter environmental standards raise compliance costs and curtail output",
            "confidence": "HIGH",
        }),
    ]
    # Add EnvironmentalRegulation entity if missing
    if "EnvironmentalRegulation" not in kg._entities:
        kg.add_entity(Entity(
            id="EnvironmentalRegulation",
            entity_type=EntityType.POLICY,
            properties={"description": "Environmental compliance requirements for mining operations"},
        ))
    for src, tgt, props in scm_causal:
        kg.add_relationship(Relationship(
            source_id=src, target_id=tgt,
            relation_type=RelationType.CAUSES,
            properties=props,
        ))

    # Bridge KG policies to SCM variables via CAUSES
    kg.add_relationship(Relationship(
        "china_export_controls", "ExportPolicy", RelationType.CAUSES,
        properties={"mechanism": "China export controls set the ExportPolicy level", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "indonesia_nickel_ore_ban", "ExportPolicy", RelationType.CAUSES,
        properties={"mechanism": "Ore export ban directly restricts trade", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "global_ev_demand", "GlobalDemand", RelationType.CAUSES,
        properties={"mechanism": "EV demand is the primary driver of GlobalDemand index", "confidence": "HIGH"},
    ))
    kg.add_relationship(Relationship(
        "drc_mining_code_reform", "Capacity", RelationType.CAUSES,
        properties={"mechanism": "Higher royalties reduce investment and constrain capacity", "confidence": "MEDIUM"},
    ))

    return kg
