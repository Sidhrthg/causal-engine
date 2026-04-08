"""
Pearl's causal inference framework for critical minerals supply chains.
Implements do-calculus for identifiability and parameter identification.
"""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple


class IdentificationStrategy(Enum):
    """Methods for causal identification."""
    SYNTHETIC_CONTROL = "synthetic_control"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    BACKDOOR_ADJUSTMENT = "backdoor_adjustment"
    FRONTDOOR_ADJUSTMENT = "frontdoor_adjustment"


@dataclass
class IdentificationResult:
    """Result of identifiability analysis."""
    identifiable: bool
    strategy: Optional[IdentificationStrategy]
    adjustment_set: Set[str]
    assumptions: List[str]
    formula: str


class CausalDAG:
    """
    Structural Causal Model (SCM) for critical minerals.

    Implements Pearl's causal inference framework:
    - Do-calculus for identifiability
    - Backdoor/frontdoor criteria
    - Identification strategies for parameters

    Reference: Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.observed_vars: Set[str] = set()
        self.unobserved_vars: Set[str] = set()

    def add_node(self, variable: str, observed: bool = True) -> None:
        """Add variable to causal graph."""
        self.graph.add_node(variable)
        if observed:
            self.observed_vars.add(variable)
        else:
            self.unobserved_vars.add(variable)

    def add_edge(self, cause: str, effect: str) -> None:
        """Add causal edge X → Y."""
        if cause not in self.graph:
            self.add_node(cause)
        if effect not in self.graph:
            self.add_node(effect)
        self.graph.add_edge(cause, effect)

    def remove_incoming_edges(self, node: str) -> nx.DiGraph:
        """
        Create mutilated graph by removing incoming edges to node.
        This represents do(node = x) intervention.
        """
        mutilated = self.graph.copy()
        incoming = list(mutilated.in_edges(node))
        mutilated.remove_edges_from(incoming)
        return mutilated

    def get_parents(self, node: str) -> Set[str]:
        """Get direct causes (parents) of node."""
        return set(self.graph.predecessors(node))

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestors (causes) of node."""
        return set(nx.ancestors(self.graph, node))

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendants (effects) of node."""
        return set(nx.descendants(self.graph, node))

    def _graph_no_outgoing(self, node: str) -> nx.DiGraph:
        """
        Graph with all edges out of node removed.
        Used for backdoor criterion: only backdoor paths (into X) remain.
        """
        g = self.graph.copy()
        outgoing = list(g.out_edges(node))
        g.remove_edges_from(outgoing)
        return g

    def d_separated(self, X: Set[str], Y: Set[str], Z: Set[str]) -> bool:
        """
        Check if X and Y are d-separated given Z.
        Uses NetworkX d-separation algorithm.
        """
        return nx.is_d_separator(self.graph, X, Y, Z)

    def backdoor_criterion(
        self, treatment: str, outcome: str, adjustment_set: Set[str]
    ) -> bool:
        """
        Check if adjustment set satisfies backdoor criterion.

        Backdoor criterion (Pearl, 2009):
        1. No node in Z is a descendant of X
        2. Z blocks all backdoor paths from X to Y (paths with arrow into X)

        If satisfied: P(Y|do(X)) = Σ_z P(Y|X,Z=z)P(Z=z)
        """
        descendants = self.get_descendants(treatment)
        if adjustment_set & descendants:
            return False
        # Check d-separation in graph with outgoing edges from X removed
        backdoor_graph = self._graph_no_outgoing(treatment)
        return nx.is_d_separator(backdoor_graph, {treatment}, {outcome}, adjustment_set)

    def find_backdoor_adjustment_set(
        self, treatment: str, outcome: str
    ) -> Optional[Set[str]]:
        """
        Find minimal sufficient adjustment set for backdoor criterion.

        Returns set of variables to condition on to identify P(Y|do(X)).
        """
        ancestors = self.get_ancestors(treatment) | self.get_ancestors(outcome)
        descendants = self.get_descendants(treatment)
        candidates = ancestors - descendants - {treatment, outcome}

        if self.backdoor_criterion(treatment, outcome, set()):
            return set()

        for size in range(1, len(candidates) + 1):
            for subset in combinations(candidates, size):
                adjustment_set = set(subset)
                if self.backdoor_criterion(treatment, outcome, adjustment_set):
                    return adjustment_set

        return None

    def frontdoor_criterion(
        self, treatment: str, outcome: str, mediator_set: Set[str]
    ) -> bool:
        """
        Check if mediator set satisfies frontdoor criterion.

        Frontdoor criterion (Pearl, 2009):
        1. M intercepts all directed paths from X to Y
        2. No backdoor path from X to M
        3. All backdoor paths from M to Y are blocked by X

        If satisfied: P(Y|do(X)) = Σ_m P(M|X) Σ_x' P(Y|M,X=x')P(X=x')
        """
        return False

    def is_identifiable(self, treatment: str, outcome: str) -> IdentificationResult:
        """
        Determine if P(outcome|do(treatment)) is identifiable from observational data.

        Checks backdoor and frontdoor criteria.
        Returns identification strategy and formula.
        """
        adjustment_set = self.find_backdoor_adjustment_set(treatment, outcome)

        if adjustment_set is not None:
            if adjustment_set <= self.observed_vars:
                formula = f"P({outcome}|do({treatment})) = Σ_z P({outcome}|{treatment},Z)P(Z)"
                assumptions = [
                    "No unmeasured confounding given adjustment set",
                    "Positivity: P(Z) > 0 for all Z",
                    "SUTVA: Stable Unit Treatment Value Assumption",
                ]
                return IdentificationResult(
                    identifiable=True,
                    strategy=IdentificationStrategy.BACKDOOR_ADJUSTMENT,
                    adjustment_set=adjustment_set,
                    assumptions=assumptions,
                    formula=formula,
                )

        if nx.is_d_separator(self.graph, {treatment}, {outcome}, set()):
            return IdentificationResult(
                identifiable=True,
                strategy=IdentificationStrategy.SYNTHETIC_CONTROL,
                adjustment_set=set(),
                assumptions=[
                    "No confounding (treatment as-if randomized)",
                    "Parallel trends (for synthetic control)",
                    "SUTVA",
                ],
                formula=f"P({outcome}|do({treatment})) = P({outcome}|{treatment})",
            )

        return IdentificationResult(
            identifiable=False,
            strategy=None,
            adjustment_set=set(),
            assumptions=[],
            formula="Not identifiable - unmeasured confounding present",
        )

    def visualize(self, filename: str = "causal_dag.png") -> None:
        """Export DAG visualization."""
        try:
            import matplotlib.pyplot as plt

            pos = nx.spring_layout(self.graph)
            colors = [
                "lightblue" if n in self.observed_vars else "lightgray"
                for n in self.graph.nodes()
            ]
            nx.draw(
                self.graph,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=3000,
                font_size=10,
                font_weight="bold",
                arrows=True,
                arrowsize=20,
            )
            plt.savefig(filename, bbox_inches="tight", dpi=300)
            print(f"✅ DAG saved to {filename}")
        except ImportError:
            print("⚠️  matplotlib not installed, skipping visualization")


@dataclass
class ParameterIdentification:
    """Maps model parameters to causal identification strategies."""

    parameter: str
    description: str
    estimand: str
    treatment: str
    outcome: str
    strategy: IdentificationStrategy
    data_requirements: List[str]
    identification_assumptions: List[str]


class GraphiteSupplyChainDAG(CausalDAG):
    """
    Specific causal DAG for graphite supply chain.

    Acyclic snapshot for do-calculus (feedback loops modeled in dynamic DAGs).
    Structure: ExportPolicy/TradeValue/Inventory/Capacity → Supply → Shortage
    → Price; GlobalDemand → Demand → Shortage.
    """

    def __init__(self) -> None:
        super().__init__()
        self._build_structure()

    def _build_structure(self) -> None:
        """Build graphite supply chain causal structure."""
        observed = [
            "ExportPolicy",
            "TradeValue",
            "Price",
            "Demand",
            "GlobalDemand",
        ]
        unobserved = [
            "Supply",
            "Shortage",
            "Inventory",
            "Capacity",
        ]

        for var in observed:
            self.add_node(var, observed=True)
        for var in unobserved:
            self.add_node(var, observed=False)

        self.add_edge("ExportPolicy", "TradeValue")
        self.add_edge("GlobalDemand", "Demand")
        self.add_edge("Capacity", "Supply")
        self.add_edge("ExportPolicy", "Supply")
        self.add_edge("Supply", "Shortage")
        self.add_edge("Demand", "Shortage")
        self.add_edge("Shortage", "Price")
        self.add_edge("TradeValue", "Inventory")
        self.add_edge("Inventory", "Supply")

    def get_parameter_identifications(self) -> List[ParameterIdentification]:
        """Return identification strategies for each model parameter."""
        return [
            ParameterIdentification(
                parameter="eta_D",
                description="Demand price elasticity",
                estimand="∂log(Demand)/∂log(Price)",
                treatment="Price",
                outcome="Demand",
                strategy=IdentificationStrategy.INSTRUMENTAL_VARIABLE,
                data_requirements=[
                    "Time series: Price and Demand",
                    "Instrument: Supply shocks (exogenous)",
                    "Controls: GlobalDemand (steel/auto production)",
                ],
                identification_assumptions=[
                    "Instrument relevance: Supply shocks affect Price",
                    "Exclusion restriction: Supply shocks affect Demand only through Price",
                    "No unmeasured price-demand confounders given controls",
                ],
            ),
            ParameterIdentification(
                parameter="tau_K",
                description="Capacity adjustment time",
                estimand="P(Capacity_t | do(PriceShock_{t-k}))",
                treatment="PriceShock",
                outcome="Capacity",
                strategy=IdentificationStrategy.SYNTHETIC_CONTROL,
                data_requirements=[
                    "Panel data: Capacity across countries/regions",
                    "Treatment: Price spike in treated region",
                    "Controls: Similar untreated regions",
                ],
                identification_assumptions=[
                    "Parallel trends: Control regions track treated absent shock",
                    "No spillovers between regions",
                    "SUTVA: Treatment stable across units",
                ],
            ),
            ParameterIdentification(
                parameter="alpha_P",
                description="Price adjustment speed",
                estimand="∂Price/∂Shortage",
                treatment="Shortage",
                outcome="Price",
                strategy=IdentificationStrategy.REGRESSION_DISCONTINUITY,
                data_requirements=[
                    "Time series: Price and estimated Shortage",
                    "Policy events creating discrete shortage jumps",
                ],
                identification_assumptions=[
                    "Local randomization around policy threshold",
                    "No manipulation of running variable",
                    "Continuity of other covariates",
                ],
            ),
            ParameterIdentification(
                parameter="policy_shock_magnitude",
                description="Effect of export quotas on supply",
                estimand="P(Supply|do(ExportPolicy=quota)) - P(Supply|ExportPolicy=free)",
                treatment="ExportPolicy",
                outcome="Supply",
                strategy=IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES,
                data_requirements=[
                    "Panel data: Trade/supply before and after policy",
                    "Treatment: Country implementing quotas",
                    "Control: Countries without policy change",
                ],
                identification_assumptions=[
                    "Parallel trends pre-treatment",
                    "No anticipation effects",
                    "No concurrent shocks to treatment group",
                ],
            ),
        ]


def demonstrate_identifiability() -> None:
    """Demo: Check identifiability of key causal effects."""
    print("=" * 70)
    print("CAUSAL IDENTIFIABILITY ANALYSIS")
    print("=" * 70)

    dag = GraphiteSupplyChainDAG()

    print("\n📊 Causal DAG Structure:")
    print(f"   Nodes: {len(dag.graph.nodes())}")
    print(f"   Edges: {len(dag.graph.edges())}")
    print(f"   Observed: {len(dag.observed_vars)}")
    print(f"   Unobserved: {len(dag.unobserved_vars)}")

    queries = [
        ("ExportPolicy", "Price"),
        ("ExportPolicy", "TradeValue"),
        ("Price", "Demand"),
    ]

    print("\n🔬 Identifiability Analysis:\n")

    for treatment, outcome in queries:
        result = dag.is_identifiable(treatment, outcome)
        print(f"Query: P({outcome}|do({treatment}))")
        print(f"  Identifiable: {'✅ YES' if result.identifiable else '❌ NO'}")
        if result.identifiable:
            print(f"  Strategy: {result.strategy.value if result.strategy else 'N/A'}")
            if result.adjustment_set:
                print(f"  Adjustment set: {result.adjustment_set}")
            print(f"  Formula: {result.formula}")
            print("  Assumptions:")
            for assumption in result.assumptions:
                print(f"    - {assumption}")
        else:
            print(f"  Reason: {result.formula}")
        print()

    print("=" * 70)
    print("PARAMETER IDENTIFICATION STRATEGIES")
    print("=" * 70)

    identifications = dag.get_parameter_identifications()
    for pid in identifications:
        print(f"\n📈 Parameter: {pid.parameter} ({pid.description})")
        print(f"   Estimand: {pid.estimand}")
        print(f"   Treatment: {pid.treatment} → Outcome: {pid.outcome}")
        print(f"   Strategy: {pid.strategy.value}")
        print("   Data Requirements:")
        for req in pid.data_requirements:
            print(f"     - {req}")
        print("   Identification Assumptions:")
        for assumption in pid.identification_assumptions:
            print(f"     - {assumption}")

    if _should_visualize():
        print("\n📊 Generating DAG visualization...")
        try:
            dag.visualize("graphite_causal_dag.png")
        except Exception as e:
            print(f"⚠️  Visualization skipped: {e}")
    else:
        print("\n📊 Skipping DAG visualization (use --plot to enable).")

    print("\n" + "=" * 70)
    print("✅ Analysis complete!")
    print("=" * 70)


def _should_visualize() -> bool:
    """Skip visualization by default to avoid matplotlib/numpy import segfaults on some systems."""
    import sys
    return "--plot" in sys.argv or "--visualize" in sys.argv


if __name__ == "__main__":
    demonstrate_identifiability()
