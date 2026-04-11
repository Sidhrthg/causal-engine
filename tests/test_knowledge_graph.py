"""Tests for CausalKnowledgeGraph — entity/relationship CRUD, queries,
schema validation, temporal queries, taxonomy, causal DAG extraction,
shock propagation, serialization, and the pre-built template."""

from __future__ import annotations

import json

import pytest

from src.minerals.causal_inference import CausalDAG, IdentificationStrategy
from src.minerals.knowledge_graph import (
    AllowedConnection,
    CausalKnowledgeGraph,
    Entity,
    EntityType,
    KGSchema,
    Relationship,
    RelationType,
    ShockTrace,
    build_critical_minerals_kg,
    build_supply_chain_schema,
)


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def empty_kg() -> CausalKnowledgeGraph:
    return CausalKnowledgeGraph()


@pytest.fixture
def simple_kg() -> CausalKnowledgeGraph:
    """Small KG: China --produces--> Graphite --enables--> EV_Batteries,
    plus a policy that CAUSES a supply effect."""
    kg = CausalKnowledgeGraph()
    kg.add_entity(Entity("china", EntityType.COUNTRY, {"region": "asia"}, aliases=["PRC"]))
    kg.add_entity(Entity("graphite", EntityType.COMMODITY, {"criticality": "high"}))
    kg.add_entity(Entity("ev_batteries", EntityType.INDUSTRY))
    kg.add_entity(Entity("export_ban", EntityType.POLICY, {"year": 2024}))

    kg.add_relationship(Relationship("china", "graphite", RelationType.PRODUCES))
    kg.add_relationship(Relationship("graphite", "ev_batteries", RelationType.ENABLES))
    kg.add_relationship(Relationship("export_ban", "graphite", RelationType.CAUSES,
                                     properties={"mechanism": "restricts supply", "confidence": "HIGH"}))
    kg.add_relationship(Relationship("export_ban", "graphite", RelationType.REGULATES))
    return kg


@pytest.fixture
def minerals_kg() -> CausalKnowledgeGraph:
    return build_critical_minerals_kg()


@pytest.fixture
def schema() -> KGSchema:
    return build_supply_chain_schema()


# ====================================================================
# Entity CRUD
# ====================================================================


class TestEntityCRUD:
    def test_add_and_get_entity(self, empty_kg):
        e = Entity("china", EntityType.COUNTRY, {"region": "asia"})
        empty_kg.add_entity(e)
        assert empty_kg.get_entity("china") is e
        assert empty_kg.num_entities == 1

    def test_get_entity_by_alias(self, simple_kg):
        entity = simple_kg.get_entity("PRC")
        assert entity is not None
        assert entity.id == "china"

    def test_get_entity_not_found(self, empty_kg):
        assert empty_kg.get_entity("nonexistent") is None

    def test_get_entities_by_type(self, simple_kg):
        countries = simple_kg.get_entities_by_type(EntityType.COUNTRY)
        assert len(countries) == 1
        assert countries[0].id == "china"

    def test_remove_entity(self, simple_kg):
        initial = simple_kg.num_entities
        simple_kg.remove_entity("china")
        assert simple_kg.num_entities == initial - 1
        assert simple_kg.get_entity("china") is None
        assert simple_kg.get_entity("PRC") is None

    def test_resolve_id(self, simple_kg):
        assert simple_kg.resolve_id("china") == "china"
        assert simple_kg.resolve_id("PRC") == "china"
        assert simple_kg.resolve_id("prc") == "china"
        assert simple_kg.resolve_id("unknown") == "unknown"

    def test_entity_to_dict_roundtrip(self):
        e = Entity("test", EntityType.COMMODITY, {"key": "val"}, aliases=["t"])
        d = e.to_dict()
        e2 = Entity.from_dict(d)
        assert e2.id == e.id
        assert e2.entity_type == e.entity_type
        assert e2.properties == e.properties
        assert e2.aliases == e.aliases

    def test_entity_temporal_fields(self):
        e = Entity("policy_x", EntityType.POLICY, start_date="2020", end_date="2025")
        assert e.active_at(2022)
        assert not e.active_at(2019)
        assert not e.active_at(2026)

    def test_entity_temporal_open_ended(self):
        e = Entity("ongoing", EntityType.EVENT, start_date="2022")
        assert e.active_at(2025)
        assert not e.active_at(2021)

    def test_entity_temporal_no_dates(self):
        e = Entity("always", EntityType.COMMODITY)
        assert e.active_at(1900)
        assert e.active_at(2100)

    def test_entity_temporal_roundtrip(self):
        e = Entity("x", EntityType.EVENT, start_date="2020", end_date="2023")
        d = e.to_dict()
        assert d["start_date"] == "2020"
        assert d["end_date"] == "2023"
        e2 = Entity.from_dict(d)
        assert e2.start_date == "2020"
        assert e2.end_date == "2023"


# ====================================================================
# Relationship CRUD
# ====================================================================


class TestRelationshipCRUD:
    def test_add_relationship(self, simple_kg):
        rels = simple_kg.get_relationships(source_id="china")
        assert len(rels) == 1
        assert rels[0].relation_type == RelationType.PRODUCES

    def test_get_relationships_by_type(self, simple_kg):
        causes = simple_kg.get_relationships_by_type(RelationType.CAUSES)
        assert len(causes) == 1
        assert causes[0].source_id == "export_ban"
        assert causes[0].target_id == "graphite"

    def test_get_relationships_by_target(self, simple_kg):
        rels = simple_kg.get_relationships(target_id="graphite")
        assert len(rels) == 3

    def test_multiple_edges_same_pair(self, simple_kg):
        rels = simple_kg.get_relationships(source_id="export_ban", target_id="graphite")
        assert len(rels) == 2
        types = {r.relation_type for r in rels}
        assert RelationType.CAUSES in types
        assert RelationType.REGULATES in types

    def test_auto_create_entities(self, empty_kg):
        empty_kg.add_relationship(Relationship("a", "b", RelationType.CAUSES))
        assert empty_kg.get_entity("a") is not None
        assert empty_kg.get_entity("b") is not None

    def test_relationship_confidence_property(self):
        r = Relationship("a", "b", RelationType.CAUSES, properties={"confidence": "HIGH"})
        assert r.confidence == "HIGH"

    def test_relationship_confidence_default(self):
        r = Relationship("a", "b", RelationType.CAUSES)
        assert r.confidence == "MEDIUM"

    def test_relationship_provenance(self):
        r = Relationship("a", "b", RelationType.CAUSES,
                         properties={"provenance": "USGS 2023", "evidence": "p.42 states..."})
        assert r.provenance == "USGS 2023"
        assert r.evidence == "p.42 states..."

    def test_relationship_temporal_active_at(self):
        r = Relationship("a", "b", RelationType.CAUSES, start_date="2020", end_date="2023")
        assert r.active_at(2021)
        assert not r.active_at(2019)
        assert not r.active_at(2024)

    def test_relationship_temporal_roundtrip(self):
        r = Relationship("a", "b", RelationType.CAUSES, start_date="2020", end_date="2025")
        d = r.to_dict()
        assert d["start_date"] == "2020"
        r2 = Relationship.from_dict(d)
        assert r2.start_date == "2020"
        assert r2.end_date == "2025"

    def test_relationship_to_dict_roundtrip(self):
        r = Relationship("a", "b", RelationType.CAUSES, {"mechanism": "test", "confidence": "LOW"})
        d = r.to_dict()
        r2 = Relationship.from_dict(d)
        assert r2.source_id == r.source_id
        assert r2.target_id == r.target_id
        assert r2.relation_type == r.relation_type
        assert r2.properties == r.properties


# ====================================================================
# Schema validation
# ====================================================================


class TestSchema:
    def test_build_supply_chain_schema(self, schema):
        assert len(schema.allowed_connections()) > 20
        assert schema.is_allowed(EntityType.COUNTRY, RelationType.PRODUCES, EntityType.COMMODITY)
        assert schema.is_allowed(EntityType.POLICY, RelationType.REGULATES, EntityType.COMMODITY)

    def test_schema_rejects_invalid(self, schema):
        # COMMODITY --PRODUCES--> COUNTRY makes no sense
        assert not schema.is_allowed(EntityType.COMMODITY, RelationType.PRODUCES, EntityType.COUNTRY)

    def test_schema_allows_causes_broadly(self, schema):
        # CAUSES is allowed between any types
        assert schema.is_allowed(EntityType.EVENT, RelationType.CAUSES, EntityType.COMMODITY)
        assert schema.is_allowed(EntityType.POLICY, RelationType.CAUSES, EntityType.ECONOMIC_INDICATOR)

    def test_schema_allows_is_a_same_type(self, schema):
        assert schema.is_allowed(EntityType.COMMODITY, RelationType.IS_A, EntityType.COMMODITY)
        assert schema.is_allowed(EntityType.COUNTRY, RelationType.IS_A, EntityType.COUNTRY)

    def test_strict_mode_raises(self, schema):
        kg = CausalKnowledgeGraph(schema=schema, strict=True)
        kg.add_entity(Entity("graphite", EntityType.COMMODITY))
        kg.add_entity(Entity("china", EntityType.COUNTRY))
        # COMMODITY --PRODUCES--> COUNTRY is not allowed
        with pytest.raises(ValueError, match="Schema violation"):
            kg.add_relationship(Relationship("graphite", "china", RelationType.PRODUCES))

    def test_non_strict_mode_warns(self, schema):
        kg = CausalKnowledgeGraph(schema=schema, strict=False)
        kg.add_entity(Entity("graphite", EntityType.COMMODITY))
        kg.add_entity(Entity("china", EntityType.COUNTRY))
        # Should not raise, but should add a warning
        kg.add_relationship(Relationship("graphite", "china", RelationType.PRODUCES))
        assert len(kg._validation_warnings) == 1

    def test_no_schema_allows_anything(self):
        kg = CausalKnowledgeGraph()  # no schema
        kg.add_entity(Entity("a", EntityType.COMMODITY))
        kg.add_entity(Entity("b", EntityType.COUNTRY))
        kg.add_relationship(Relationship("a", "b", RelationType.PRODUCES))
        assert len(kg._validation_warnings) == 0

    def test_schema_to_dict_roundtrip(self, schema):
        d = schema.to_dict()
        assert "entity_types" in d
        assert "relation_types" in d
        assert "allowed_connections" in d
        schema2 = KGSchema.from_dict(d)
        assert schema2.is_allowed(EntityType.COUNTRY, RelationType.PRODUCES, EntityType.COMMODITY)

    def test_schema_serialized_with_kg(self, schema):
        kg = CausalKnowledgeGraph(schema=schema)
        kg.add_entity(Entity("x", EntityType.COMMODITY))
        d = kg.to_dict()
        assert "schema" in d
        # Roundtrip
        kg2 = CausalKnowledgeGraph.from_dict(d)
        assert kg2.schema is not None

    def test_build_minerals_kg_with_schema(self):
        kg = build_critical_minerals_kg(with_schema=True)
        assert kg.schema is not None
        # Template should be schema-compliant (no warnings)
        assert len(kg._validation_warnings) == 0


# ====================================================================
# Graph queries
# ====================================================================


class TestGraphQueries:
    def test_get_neighbors_out(self, simple_kg):
        neighbors = simple_kg.get_neighbors("china", direction="out")
        ids = {n.id for n in neighbors}
        assert "graphite" in ids

    def test_get_neighbors_in(self, simple_kg):
        neighbors = simple_kg.get_neighbors("graphite", direction="in")
        ids = {n.id for n in neighbors}
        assert "china" in ids
        assert "export_ban" in ids

    def test_get_neighbors_filtered(self, simple_kg):
        neighbors = simple_kg.get_neighbors("graphite", relation_type=RelationType.PRODUCES, direction="in")
        ids = {n.id for n in neighbors}
        assert "china" in ids
        assert "export_ban" not in ids

    def test_get_upstream(self, simple_kg):
        upstream = simple_kg.get_upstream("ev_batteries")
        ids = {e.id for e in upstream}
        assert "graphite" in ids

    def test_get_downstream(self, simple_kg):
        downstream = simple_kg.get_downstream("china")
        ids = {e.id for e in downstream}
        assert "graphite" in ids
        assert "ev_batteries" in ids

    def test_get_downstream_filtered(self, simple_kg):
        downstream = simple_kg.get_downstream("china", relation_types=[RelationType.PRODUCES])
        ids = {e.id for e in downstream}
        assert "graphite" in ids
        assert "ev_batteries" not in ids

    def test_find_paths(self, simple_kg):
        paths = simple_kg.find_paths("china", "ev_batteries")
        assert len(paths) >= 1
        assert paths[0] == ["china", "graphite", "ev_batteries"]

    def test_find_paths_no_path(self, simple_kg):
        paths = simple_kg.find_paths("ev_batteries", "china")
        assert len(paths) == 0

    def test_find_paths_filtered(self, simple_kg):
        paths = simple_kg.find_paths("china", "ev_batteries", relation_types=[RelationType.PRODUCES])
        assert len(paths) == 0

    def test_subgraph(self, simple_kg):
        sub = simple_kg.subgraph(["china", "graphite"])
        assert sub.num_entities == 2
        rels = sub.get_relationships()
        assert len(rels) >= 1


# ====================================================================
# Temporal queries
# ====================================================================


class TestTemporalQueries:
    def test_query_at_time_filters_entities(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("old_policy", EntityType.POLICY, start_date="2010", end_date="2015"))
        kg.add_entity(Entity("new_policy", EntityType.POLICY, start_date="2020"))
        kg.add_entity(Entity("commodity", EntityType.COMMODITY))

        snapshot_2012 = kg.query_at_time(2012)
        ids = {e.id for e in snapshot_2012._entities.values()}
        assert "old_policy" in ids
        assert "commodity" in ids
        assert "new_policy" not in ids

        snapshot_2022 = kg.query_at_time(2022)
        ids = {e.id for e in snapshot_2022._entities.values()}
        assert "new_policy" in ids
        assert "old_policy" not in ids

    def test_query_at_time_filters_relationships(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("a", EntityType.COMMODITY))
        kg.add_entity(Entity("b", EntityType.COMMODITY))
        kg.add_relationship(Relationship("a", "b", RelationType.CAUSES, start_date="2020", end_date="2025"))
        kg.add_relationship(Relationship("a", "b", RelationType.SUBSTITUTES))  # no dates = always

        snapshot_2022 = kg.query_at_time(2022)
        assert snapshot_2022.num_relationships == 2

        snapshot_2030 = kg.query_at_time(2030)
        assert snapshot_2030.num_relationships == 1  # only the always-active one

    def test_get_relationships_at(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("a", EntityType.POLICY, start_date="2020"))
        kg.add_entity(Entity("b", EntityType.COMMODITY))
        kg.add_relationship(Relationship("a", "b", RelationType.REGULATES, start_date="2020", end_date="2025"))

        assert len(kg.get_relationships_at(2022)) == 1
        assert len(kg.get_relationships_at(2019)) == 0
        assert len(kg.get_relationships_at(2026)) == 0

    def test_minerals_kg_temporal_events(self, minerals_kg):
        events = minerals_kg.get_entities_by_type(EntityType.EVENT)
        assert len(events) >= 3
        # Check that events have temporal data
        crisis = minerals_kg.get_entity("china_rare_earth_crisis_2010")
        assert crisis is not None
        assert crisis.start_date == "2010"
        assert crisis.active_at(2011)
        assert not crisis.active_at(2009)

    def test_minerals_kg_temporal_disruptions(self, minerals_kg):
        disruptions = minerals_kg.get_relationships_at(
            2021, relation_type=RelationType.DISRUPTS
        )
        # COVID disruptions active in 2021
        sources = {r.source_id for r in disruptions}
        assert "covid_supply_disruption_2020" in sources

    def test_minerals_kg_policy_start_dates(self, minerals_kg):
        policy = minerals_kg.get_entity("china_export_controls")
        assert policy is not None
        assert policy.start_date == "2010"


# ====================================================================
# Taxonomy (IS_A, PART_OF)
# ====================================================================


class TestTaxonomy:
    def test_is_a_traversal_up(self, minerals_kg):
        categories = minerals_kg.get_categories_of("lithium")
        ids = {e.id for e in categories}
        assert "battery_material" in ids

    def test_is_a_traversal_down(self, minerals_kg):
        instances = minerals_kg.get_instances_of("battery_material")
        ids = {e.id for e in instances}
        assert "lithium" in ids
        assert "cobalt" in ids
        assert "graphite" in ids

    def test_part_of_regions(self, minerals_kg):
        # China should be PART_OF asia_pacific
        regions = minerals_kg.get_taxonomy("china", RelationType.PART_OF, direction="up")
        ids = {e.id for e in regions}
        assert "asia_pacific" in ids

    def test_part_of_countries_in_region(self, minerals_kg):
        # asia_pacific should have children via PART_OF
        countries = minerals_kg.get_taxonomy("asia_pacific", RelationType.PART_OF, direction="down")
        ids = {e.id for e in countries}
        assert "china" in ids
        assert "japan" in ids
        assert "australia" in ids

    def test_strategic_minerals(self, minerals_kg):
        strategic = minerals_kg.get_instances_of("strategic_mineral")
        ids = {e.id for e in strategic}
        assert "tungsten" in ids
        assert "antimony" in ids
        assert "beryllium" in ids

    def test_semiconductor_materials(self, minerals_kg):
        semi = minerals_kg.get_instances_of("semiconductor_material")
        ids = {e.id for e in semi}
        assert "gallium" in ids
        assert "germanium" in ids
        assert "indium" in ids


# ====================================================================
# Causal reasoning
# ====================================================================


class TestCausalReasoning:
    def test_to_causal_dag(self, simple_kg):
        dag = simple_kg.to_causal_dag()
        assert isinstance(dag, CausalDAG)
        assert dag.graph.has_edge("export_ban", "graphite")
        assert dag.graph.number_of_edges() == 1

    def test_to_causal_dag_observability(self, simple_kg):
        dag = simple_kg.to_causal_dag()
        assert "export_ban" in dag.observed_vars
        assert "graphite" in dag.observed_vars

    def test_to_causal_dag_custom_relation_types(self, simple_kg):
        dag = simple_kg.to_causal_dag(relation_types=[RelationType.CAUSES, RelationType.ENABLES])
        assert dag.graph.has_edge("export_ban", "graphite")
        assert dag.graph.has_edge("graphite", "ev_batteries")

    def test_to_causal_dag_identifiability(self, simple_kg):
        dag = simple_kg.to_causal_dag()
        result = dag.is_identifiable("export_ban", "graphite")
        assert result.identifiable

    def test_find_confounders(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("X", EntityType.ECONOMIC_INDICATOR))
        kg.add_entity(Entity("Y", EntityType.ECONOMIC_INDICATOR))
        kg.add_entity(Entity("Z", EntityType.ECONOMIC_INDICATOR))
        kg.add_relationship(Relationship("Z", "X", RelationType.CAUSES))
        kg.add_relationship(Relationship("Z", "Y", RelationType.CAUSES))
        kg.add_relationship(Relationship("X", "Y", RelationType.CAUSES))

        confounders = kg.find_confounders("X", "Y")
        ids = {c.id for c in confounders}
        assert "Z" in ids

    def test_find_confounders_none(self, simple_kg):
        confounders = simple_kg.find_confounders("export_ban", "graphite")
        assert len(confounders) == 0


# ====================================================================
# Shock propagation
# ====================================================================


class TestShockPropagation:
    def test_propagate_shock_basic(self, simple_kg):
        trace = simple_kg.propagate_shock("export_ban", initial_magnitude=1.0, decay=0.5)
        assert isinstance(trace, ShockTrace)
        assert "graphite" in trace.affected
        assert trace.affected["graphite"] > 0

    def test_propagate_shock_decay(self, simple_kg):
        trace = simple_kg.propagate_shock("export_ban", initial_magnitude=1.0, decay=0.5)
        assert trace.affected.get("graphite", 0) > 0

    def test_propagate_shock_custom_relations(self, simple_kg):
        trace = simple_kg.propagate_shock(
            "china",
            relation_types=[RelationType.PRODUCES, RelationType.ENABLES],
        )
        assert "graphite" in trace.affected
        assert "ev_batteries" in trace.affected

    def test_propagate_shock_max_depth(self, simple_kg):
        trace = simple_kg.propagate_shock("export_ban", max_depth=0)
        assert len(trace.affected) == 0

    def test_propagate_shock_paths(self, simple_kg):
        trace = simple_kg.propagate_shock("export_ban")
        if "graphite" in trace.paths:
            assert trace.paths["graphite"] == ["export_ban", "graphite"]

    def test_shock_origin_candidates(self, minerals_kg):
        candidates = minerals_kg.get_shock_origin_candidates()
        assert "china_export_controls" in candidates
        assert "global_ev_demand" in candidates


# ====================================================================
# Serialization
# ====================================================================


class TestSerialization:
    def test_to_dict(self, simple_kg):
        d = simple_kg.to_dict()
        assert "entities" in d
        assert "relationships" in d
        assert "metadata" in d
        assert d["metadata"]["num_entities"] == 4
        assert d["metadata"]["num_relationships"] == 4

    def test_json_roundtrip(self, simple_kg):
        json_str = simple_kg.to_json()
        kg2 = CausalKnowledgeGraph.from_json(json_str)
        assert kg2.num_entities == simple_kg.num_entities
        assert kg2.num_relationships == simple_kg.num_relationships

    def test_json_file_roundtrip(self, simple_kg, tmp_path):
        path = str(tmp_path / "test_kg.json")
        simple_kg.to_json(path)
        kg2 = CausalKnowledgeGraph.from_json(path)
        assert kg2.num_entities == simple_kg.num_entities
        assert kg2.num_relationships == simple_kg.num_relationships
        china = kg2.get_entity("china")
        assert china is not None
        assert china.entity_type == EntityType.COUNTRY

    def test_from_dict(self):
        data = {
            "entities": [
                {"id": "a", "entity_type": "commodity", "properties": {}, "aliases": []},
                {"id": "b", "entity_type": "country", "properties": {}, "aliases": []},
            ],
            "relationships": [
                {"source_id": "a", "target_id": "b", "relation_type": "causes", "properties": {}},
            ],
        }
        kg = CausalKnowledgeGraph.from_dict(data)
        assert kg.num_entities == 2
        assert kg.num_relationships == 1

    def test_temporal_fields_survive_roundtrip(self, tmp_path):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("e", EntityType.EVENT, start_date="2020", end_date="2023"))
        kg.add_entity(Entity("c", EntityType.COMMODITY))
        kg.add_relationship(Relationship("e", "c", RelationType.DISRUPTS, start_date="2020", end_date="2021"))

        path = str(tmp_path / "temporal.json")
        kg.to_json(path)
        kg2 = CausalKnowledgeGraph.from_json(path)

        entity = kg2.get_entity("e")
        assert entity.start_date == "2020"
        assert entity.end_date == "2023"

        rels = kg2.get_relationships_by_type(RelationType.DISRUPTS)
        assert len(rels) == 1
        assert rels[0].start_date == "2020"
        assert rels[0].end_date == "2021"


# ====================================================================
# Import from existing structures
# ====================================================================


class TestImport:
    def test_import_from_dag_registry(self, empty_kg, tmp_path):
        registry = {
            "nodes": ["supply_shock", "price_increase"],
            "edges": [
                {
                    "from": "supply_shock",
                    "to": "price_increase",
                    "mechanism": "reduced supply raises price",
                    "confidence": "HIGH",
                    "source": "test.txt",
                    "evidence": "test evidence",
                    "validated": True,
                }
            ],
            "metadata": {"total_edges": 1, "validated_edges": 1, "sources": ["test.txt"]},
        }
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(registry))

        count = empty_kg.import_from_dag_registry(str(path))
        assert count == 1
        rels = empty_kg.get_relationships_by_type(RelationType.CAUSES)
        assert len(rels) == 1
        assert rels[0].source_id == "supply_shock"
        assert rels[0].properties["validated"] is True


# ====================================================================
# Pre-built critical minerals template
# ====================================================================


class TestCriticalMineralsKG:
    def test_build(self, minerals_kg):
        assert minerals_kg.num_entities > 50
        assert minerals_kg.num_relationships > 150

    def test_has_commodities(self, minerals_kg):
        commodities = minerals_kg.get_entities_by_type(EntityType.COMMODITY)
        ids = {c.id for c in commodities}
        assert "graphite" in ids
        assert "lithium" in ids
        assert "cobalt" in ids
        assert "rare_earths" in ids
        assert "copper" in ids

    def test_has_countries(self, minerals_kg):
        countries = minerals_kg.get_entities_by_type(EntityType.COUNTRY)
        ids = {c.id for c in countries}
        assert "china" in ids
        assert "usa" in ids
        assert "australia" in ids

    def test_has_policies(self, minerals_kg):
        policies = minerals_kg.get_entities_by_type(EntityType.POLICY)
        ids = {p.id for p in policies}
        assert "china_export_controls" in ids
        assert "us_inflation_reduction_act" in ids

    def test_has_industries(self, minerals_kg):
        industries = minerals_kg.get_entities_by_type(EntityType.INDUSTRY)
        ids = {i.id for i in industries}
        assert "ev_batteries" in ids
        assert "semiconductors" in ids

    def test_has_events(self, minerals_kg):
        events = minerals_kg.get_entities_by_type(EntityType.EVENT)
        ids = {e.id for e in events}
        assert "china_rare_earth_crisis_2010" in ids
        assert "covid_supply_disruption_2020" in ids
        assert "russia_ukraine_conflict_2022" in ids

    def test_has_risk_factors(self, minerals_kg):
        risks = minerals_kg.get_entities_by_type(EntityType.RISK_FACTOR)
        ids = {r.id for r in risks}
        assert "geographic_concentration_risk" in ids
        assert "geopolitical_risk" in ids

    def test_has_regions(self, minerals_kg):
        regions = minerals_kg.get_entities_by_type(EntityType.REGION)
        ids = {r.id for r in regions}
        assert "asia_pacific" in ids
        assert "americas" in ids

    def test_china_produces_graphite(self, minerals_kg):
        rels = minerals_kg.get_relationships(
            source_id="china", target_id="graphite", relation_type=RelationType.PRODUCES
        )
        assert len(rels) == 1
        assert rels[0].properties.get("share") == 0.65

    def test_ev_batteries_consumes_lithium(self, minerals_kg):
        rels = minerals_kg.get_relationships(
            source_id="ev_batteries", target_id="lithium", relation_type=RelationType.CONSUMES
        )
        assert len(rels) == 1

    def test_export_controls_cause_graphite_effect(self, minerals_kg):
        rels = minerals_kg.get_relationships(
            source_id="china_export_controls", target_id="graphite", relation_type=RelationType.CAUSES
        )
        assert len(rels) == 1
        assert rels[0].properties["mechanism"] == "export restrictions reduce global supply"

    def test_shock_propagation_china_export_controls(self, minerals_kg):
        trace = minerals_kg.propagate_shock("china_export_controls")
        assert "graphite" in trace.affected
        assert "rare_earths" in trace.affected
        # Shocks should now cascade to downstream industries
        assert "ev_batteries" in trace.affected
        assert "semiconductors" in trace.affected
        assert "defense_systems" in trace.affected
        # And further downstream
        assert "wind_turbines" in trace.affected
        assert len(trace.affected) >= 20

    def test_shock_propagation_paths_make_sense(self, minerals_kg):
        trace = minerals_kg.propagate_shock("china_export_controls")
        # Direct commodity hits should have higher impact than indirect industry hits
        assert trace.affected["graphite"] > trace.affected.get("wind_turbines", 0)
        # Paths should show cascading chains
        if "wind_turbines" in trace.paths:
            assert len(trace.paths["wind_turbines"]) >= 3

    def test_has_trade_flows(self, minerals_kg):
        exports = minerals_kg.get_relationships_by_type(RelationType.EXPORTS_TO)
        assert len(exports) >= 20
        # China exports to USA
        china_usa = [r for r in exports
                     if r.source_id == "china" and r.target_id == "usa"]
        assert len(china_usa) >= 1

    def test_commodity_causes_industry(self, minerals_kg):
        # Commodities should have causal links to the industries that depend on them
        rels = minerals_kg.get_relationships(
            source_id="graphite", target_id="ev_batteries",
            relation_type=RelationType.CAUSES,
        )
        assert len(rels) == 1
        assert "anode" in rels[0].properties["mechanism"].lower()

    def test_to_causal_dag_from_template(self, minerals_kg):
        dag = minerals_kg.to_causal_dag()
        assert isinstance(dag, CausalDAG)
        assert dag.graph.number_of_nodes() > 20
        assert dag.graph.number_of_edges() > 40
        assert dag.graph.has_edge("china_export_controls", "graphite")
        # Should also have commodity → industry causal edges
        assert dag.graph.has_edge("graphite", "ev_batteries")
        assert dag.graph.has_edge("gallium", "semiconductors")

    def test_summary(self, minerals_kg):
        s = minerals_kg.summary()
        assert "CausalKnowledgeGraph" in s
        assert "commodity" in s
        assert "country" in s

    def test_upstream_of_ev_batteries(self, minerals_kg):
        upstream = minerals_kg.get_upstream(
            "ev_batteries",
            relation_types=[RelationType.CONSUMES, RelationType.ENABLES],
        )
        ids = {e.id for e in upstream}
        assert "lithium" in ids
        assert "cobalt" in ids
        assert "graphite" in ids

    def test_downstream_of_china(self, minerals_kg):
        downstream = minerals_kg.get_downstream(
            "china", relation_types=[RelationType.PRODUCES],
        )
        ids = {e.id for e in downstream}
        assert "graphite" in ids
        assert "rare_earths" in ids
        assert "gallium" in ids

    def test_dependency_path_usa_to_china(self, minerals_kg):
        rels = minerals_kg.get_relationships(
            source_id="usa", target_id="china", relation_type=RelationType.DEPENDS_ON
        )
        assert len(rels) == 1
        assert "graphite" in rels[0].properties.get("commodities", [])

    def test_substitution_relationships(self, minerals_kg):
        subs = minerals_kg.get_relationships_by_type(RelationType.SUBSTITUTES)
        assert len(subs) >= 2
        src_ids = {s.source_id for s in subs}
        assert "lithium" in src_ids or "cobalt" in src_ids

    def test_event_disruptions(self, minerals_kg):
        disruptions = minerals_kg.get_relationships_by_type(RelationType.DISRUPTS)
        assert len(disruptions) >= 5
        # COVID should disrupt copper and cobalt
        covid_disruptions = [
            r for r in disruptions if r.source_id == "covid_supply_disruption_2020"
        ]
        targets = {r.target_id for r in covid_disruptions}
        assert "copper" in targets
        assert "cobalt" in targets

    def test_policy_mitigates_risk(self, minerals_kg):
        mitigations = minerals_kg.get_relationships_by_type(RelationType.MITIGATES)
        assert len(mitigations) >= 2
        sources = {r.source_id for r in mitigations}
        assert "us_inflation_reduction_act" in sources

    def test_temporal_sequencing(self, minerals_kg):
        preceded = minerals_kg.get_relationships_by_type(RelationType.PRECEDED_BY)
        assert len(preceded) >= 1


# ====================================================================
# Trust, provenance & validation
# ====================================================================


class TestTrustAndValidation:
    def test_trust_score_well_connected(self, minerals_kg):
        score = minerals_kg.trust_score("graphite")
        assert 0.0 < score <= 1.0
        # Graphite has many HIGH-confidence relationships
        assert score > 0.5

    def test_trust_score_unknown_entity(self, minerals_kg):
        assert minerals_kg.trust_score("nonexistent") == 0.0

    def test_trust_score_isolated(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("lonely", EntityType.COMMODITY))
        assert kg.trust_score("lonely") == 0.5

    def test_validate_integrity_clean(self, minerals_kg):
        issues = minerals_kg.validate_integrity()
        # No causal cycles
        cycle_issues = [i for i in issues if "Causal cycle" in i]
        assert len(cycle_issues) == 0
        # No dangling edges
        dangling = [i for i in issues if "Dangling" in i]
        assert len(dangling) == 0

    def test_validate_integrity_detects_cycle(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("a", EntityType.COMMODITY))
        kg.add_entity(Entity("b", EntityType.COMMODITY))
        kg.add_relationship(Relationship("a", "b", RelationType.CAUSES,
                                         properties={"mechanism": "test"}))
        kg.add_relationship(Relationship("b", "a", RelationType.CAUSES,
                                         properties={"mechanism": "test"}))
        issues = kg.validate_integrity()
        cycle_issues = [i for i in issues if "Causal cycle" in i]
        assert len(cycle_issues) > 0

    def test_validate_integrity_missing_mechanism(self):
        kg = CausalKnowledgeGraph()
        kg.add_entity(Entity("a", EntityType.COMMODITY))
        kg.add_entity(Entity("b", EntityType.COMMODITY))
        kg.add_relationship(Relationship("a", "b", RelationType.CAUSES))
        issues = kg.validate_integrity()
        mechanism_issues = [i for i in issues if "missing mechanism" in i]
        assert len(mechanism_issues) == 1

    def test_provenance_report(self, minerals_kg):
        report = minerals_kg.provenance_report()
        assert report["total_relationships"] == minerals_kg.num_relationships
        assert "confidence_distribution" in report
        assert report["confidence_distribution"]["HIGH"] > 0
        assert "by_source" in report
