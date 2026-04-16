"""
Pearl's causal inference framework for critical minerals supply chains.
Implements do-calculus for identifiability and parameter identification.
"""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from . import do_calculus


class IdentificationStrategy(Enum):
    """Methods for causal identification."""
    SYNTHETIC_CONTROL = "synthetic_control"
    INSTRUMENTAL_VARIABLE = "instrumental_variable"
    REGRESSION_DISCONTINUITY = "regression_discontinuity"
    DIFFERENCE_IN_DIFFERENCES = "difference_in_differences"
    BACKDOOR_ADJUSTMENT = "backdoor_adjustment"
    FRONTDOOR_ADJUSTMENT = "frontdoor_adjustment"
    ID_ALGORITHM = "id_algorithm"  # Shpitser-Pearl full ID algorithm


@dataclass
class IdentificationResult:
    """Result of identifiability analysis."""
    identifiable: bool
    strategy: Optional[IdentificationStrategy]
    adjustment_set: Set[str]
    assumptions: List[str]
    formula: str
    derivation_steps: List[str] = field(default_factory=list)


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
        2. No backdoor path from X to M (i.e. X d-separated from M given
           empty set in the graph with outgoing edges from X removed — but we
           actually need: no unblocked backdoor path from X to M, which is
           equivalent to X and M being d-separated given empty set in the
           subgraph where outgoing edges from X are removed)
        3. All backdoor paths from M to Y are blocked by X

        If satisfied: P(Y|do(X)) = Σ_m P(M|X) Σ_x' P(Y|M,X=x')P(X=x')
        """
        if not mediator_set:
            return False

        # All mediators must be observed
        if not mediator_set <= self.observed_vars:
            return False

        # Condition 1: M intercepts all directed paths from X to Y.
        # Remove mediator nodes and check if any directed path remains.
        g_no_m = self.graph.copy()
        g_no_m.remove_nodes_from(mediator_set)
        if nx.has_path(g_no_m, treatment, outcome):
            return False

        # Condition 2: No unblocked backdoor path from X to any M.
        # Equivalent to: in graph with outgoing edges of X removed,
        # X is d-separated from M given the empty set.
        g_no_out_x = self._graph_no_outgoing(treatment)
        for m in mediator_set:
            if not nx.is_d_separator(g_no_out_x, {treatment}, {m}, set()):
                return False

        # Condition 3: All backdoor paths from M to Y are blocked by {X}.
        for m in mediator_set:
            g_no_out_m = self.graph.copy()
            outgoing = list(g_no_out_m.out_edges(m))
            g_no_out_m.remove_edges_from(outgoing)
            if not nx.is_d_separator(g_no_out_m, {m}, {outcome}, {treatment}):
                return False

        return True

    def find_frontdoor_set(
        self, treatment: str, outcome: str
    ) -> Optional[Set[str]]:
        """
        Find a set of mediators satisfying the frontdoor criterion.

        Searches observed nodes that lie on directed paths from treatment
        to outcome.
        """
        # Candidate mediators: observed nodes on directed paths from X to Y
        descendants_x = self.get_descendants(treatment)
        ancestors_y = self.get_ancestors(outcome)
        candidates = (descendants_x & ancestors_y & self.observed_vars) - {treatment, outcome}

        if not candidates:
            return None

        # Try subsets of increasing size
        for size in range(1, len(candidates) + 1):
            for subset in combinations(candidates, size):
                mediator_set = set(subset)
                if self.frontdoor_criterion(treatment, outcome, mediator_set):
                    return mediator_set

        return None

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
                deriv = do_calculus.derivation_steps_for_result(
                    treatment, outcome,
                    IdentificationStrategy.BACKDOOR_ADJUSTMENT.value,
                    adjustment_set, formula,
                )
                return IdentificationResult(
                    identifiable=True,
                    strategy=IdentificationStrategy.BACKDOOR_ADJUSTMENT,
                    adjustment_set=adjustment_set,
                    assumptions=assumptions,
                    formula=formula,
                    derivation_steps=deriv,
                )

        # Try frontdoor criterion
        frontdoor_set = self.find_frontdoor_set(treatment, outcome)
        if frontdoor_set is not None:
            m_str = ", ".join(sorted(frontdoor_set))
            formula = (
                f"P({outcome}|do({treatment})) = "
                f"Σ_{{{m_str}}} P({m_str}|{treatment}) "
                f"Σ_{{{treatment}'}} P({outcome}|{m_str},{treatment}')P({treatment}')"
            )
            deriv = do_calculus.derivation_steps_for_result(
                treatment, outcome,
                IdentificationStrategy.FRONTDOOR_ADJUSTMENT.value,
                frontdoor_set, formula,
            )
            return IdentificationResult(
                identifiable=True,
                strategy=IdentificationStrategy.FRONTDOOR_ADJUSTMENT,
                adjustment_set=frontdoor_set,
                assumptions=[
                    f"Frontdoor criterion satisfied via mediator(s): {{{m_str}}}",
                    f"{m_str} intercept(s) all directed paths from {treatment} to {outcome}",
                    f"No unblocked backdoor path from {treatment} to {m_str}",
                    f"All backdoor paths from {m_str} to {outcome} are blocked by {treatment}",
                ],
                formula=formula,
                derivation_steps=deriv,
            )

        if nx.is_d_separator(self.graph, {treatment}, {outcome}, set()):
            formula_triv = f"P({outcome}|do({treatment})) = P({outcome}|{treatment})"
            deriv = do_calculus.derivation_steps_for_result(
                treatment, outcome, None, set(), formula_triv,
            )
            return IdentificationResult(
                identifiable=True,
                strategy=IdentificationStrategy.SYNTHETIC_CONTROL,
                adjustment_set=set(),
                assumptions=[
                    "No confounding (treatment as-if randomized)",
                    "Parallel trends (for synthetic control)",
                    "SUTVA",
                ],
                formula=formula_triv,
                derivation_steps=deriv if deriv else [],
            )

        # ---------------------------------------------------------------
        # Full ID Algorithm (Shpitser & Pearl 2006) — handles all cases
        # that backdoor and frontdoor cannot, including effects identifiable
        # via C-component decomposition with hidden confounders.
        # ---------------------------------------------------------------
        id_result = do_calculus.id_algorithm(self, treatment, outcome)
        if id_result["identifiable"]:
            formula = id_result["formula"]
            # Build human-readable assumptions from bidirected / C-component info
            bd = id_result.get("bidirected_edges", [])
            cc = id_result.get("c_components", [])
            assumptions = [
                "Causal graph is correct (no missing/extra edges)",
                f"Hidden confounders create bidirected edges: {bd if bd else '∅'}",
                f"Observed C-components of graph: {cc}",
                "Positivity: all conditioning events have positive probability",
                "SUTVA: stable unit treatment value assumption",
            ]
            return IdentificationResult(
                identifiable=True,
                strategy=IdentificationStrategy.ID_ALGORITHM,
                adjustment_set=set(),  # ID algorithm uses C-component structure, not a simple adj set
                assumptions=assumptions,
                formula=formula,
                derivation_steps=id_result["derivation_steps"],
            )

        return IdentificationResult(
            identifiable=False,
            strategy=None,
            adjustment_set=set(),
            assumptions=[],
            formula=id_result.get("formula", "Not identifiable — unmeasured confounding present"),
            derivation_steps=id_result.get("derivation_steps", []),
        )

    def visualize(self, filename: str = "causal_dag.png") -> None:
        """Export DAG visualization."""
        try:
            if self.graph.number_of_nodes() == 0:
                return
            import matplotlib
            matplotlib.use("Agg")  # Headless backend for server/CLI
            import matplotlib.pyplot as plt

            G = self.graph
            n_nodes = G.number_of_nodes()

            # Use hierarchical layout for DAGs to avoid overlap
            try:
                generations = list(nx.topological_generations(G))
                pos = {}
                max_width = max(len(layer) for layer in generations)
                h_spacing = 1.2
                v_spacing = 1.5
                for layer_idx, layer in enumerate(generations):
                    y = (len(generations) - 1 - layer_idx) * v_spacing
                    sorted_layer = sorted(layer)
                    n = len(sorted_layer)
                    spacing = max(0.8, h_spacing * max_width / max(n, 1))
                    for i, node in enumerate(sorted_layer):
                        x = (i - (n - 1) / 2) * spacing
                        pos[node] = (x, y)
            except nx.NetworkXError:
                # Fallback: spring with larger k for more spacing
                pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)

            colors = [
                "lightblue" if n in self.observed_vars else "lightgray"
                for n in G.nodes()
            ]
            fig_w = max(14, n_nodes * 0.8)
            fig_h = max(10, n_nodes * 0.5)
            plt.figure(figsize=(fig_w, fig_h))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=2800,
                font_size=9,
                font_weight="bold",
                arrows=True,
                arrowsize=18,
            )
            plt.tight_layout()
            plt.savefig(filename, bbox_inches="tight", dpi=150)
            plt.close()
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
        """Build graphite supply chain causal structure.

        Observed nodes correspond to variables measurable in trade/price data:
          ExportPolicy  — export quota/restriction severity (0=none, 1=full ban)
          TradeValue    — bilateral trade value USD (UN Comtrade / CEPII)
          Price         — graphite spot price USD/tonne
          Demand        — industrial demand (steel/battery production index)
          GlobalDemand  — macro demand driver (global IP index)

        Unobserved nodes are latent market state variables not directly in data:
          Supply        — physical supply (production - inventory change)
          Shortage      — supply-demand gap (Supply - Demand)
          Inventory     — stockpile level (not publicly reported in real time)
          Capacity      — mining/processing capacity (changes slowly, partially inferred)

        The key observed causal paths are:
          ExportPolicy → TradeValue → Price  (trade channel: restrictions cut trade, raise price)
          ExportPolicy → TradeValue         (direct effect of policy on measured trade)
          GlobalDemand → Demand → Price     (demand channel: demand growth raises price)
          TradeValue   → Price              (reduced-form: lower trade value ↔ higher prices)
          Demand       → Price              (reduced-form: higher demand ↔ higher prices)

        These reduced-form edges make P(Price|do(ExportPolicy)) identifiable from
        observational data via the backdoor criterion (adjusting for Demand/GlobalDemand).
        The full structural mechanism (through Supply/Shortage) is modelled in model.py
        and is used for simulation-based Layer-2 do() and Layer-3 counterfactuals.
        """
        observed = [
            "ExportPolicy",
            "TradeValue",
            "Price",
            "Demand",
            "GlobalDemand",
            # New L2 nodes: supply substitution and fringe / cost-curve entry
            "SubstitutionSupply",   # non-dominant suppliers rerouting around restrictions
            "FringeSupply",         # high-cost entrants responding to elevated price
        ]
        unobserved = [
            "Supply",
            "Shortage",
            "Inventory",
            "Capacity",
            "SubstitutionCapacity",  # latent: total non-dominant supplier capacity
            "FringeCapacity",        # latent: total fringe producer capacity
        ]

        for var in observed:
            self.add_node(var, observed=True)
        for var in unobserved:
            self.add_node(var, observed=False)

        # Observed edges (identifiable from trade/price data)
        self.add_edge("ExportPolicy", "TradeValue")   # policy restricts trade
        self.add_edge("TradeValue", "Price")           # lower supply → higher price
        self.add_edge("GlobalDemand", "Demand")        # macro drives demand
        self.add_edge("Demand", "Price")               # demand growth → price rise

        # Supply substitution (Pearl L2 node: SubstitutionSupply)
        # Causal parents: ExportPolicy (restriction creates the gap to fill)
        #                 Price (price signal attracts non-dominant suppliers)
        #                 SubstitutionCapacity (latent: how much RoW can supply)
        self.add_edge("ExportPolicy",          "SubstitutionSupply")
        self.add_edge("Price",                 "SubstitutionSupply")
        self.add_edge("SubstitutionCapacity",  "SubstitutionSupply")
        self.add_edge("SubstitutionSupply",    "TradeValue")  # RoW fills trade gap
        self.add_edge("SubstitutionSupply",    "Shortage")    # reduces shortage

        # Fringe / cost-curve supply (Pearl L2 node: FringeSupply)
        # Causal parents: Price (entry only profitable above threshold)
        #                 FringeCapacity (latent: how much fringe capacity exists)
        # do(fringe_entry_price → low) = graph surgery: FringeCapacity → FringeSupply
        # activates at lower P, expanding effective supply and dampening price spikes
        self.add_edge("Price",          "FringeSupply")
        self.add_edge("FringeCapacity", "FringeSupply")
        self.add_edge("FringeSupply",   "TradeValue")  # adds to total trade volume
        self.add_edge("FringeSupply",   "Shortage")    # reduces shortage

        # Unobserved structural mechanism (full SCM used in simulation)
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


    def estimate_parameter(
        self,
        parameter: str,
        data: "pd.DataFrame",
        **kwargs,
    ):
        """
        Estimate a model parameter using its pre-specified identification strategy.

        Dispatches to the appropriate estimator in causal_identification based on
        the strategy in get_parameter_identifications().

        Args:
            parameter: One of 'eta_D', 'tau_K', 'alpha_P', 'policy_shock_magnitude'.
            data: Panel / time-series DataFrame with the required columns.
            **kwargs: Forwarded to the estimator. Required kwargs per parameter:
                eta_D               → instrument_var (str)
                tau_K               → treated_unit, control_units, treatment_time
                alpha_P             → threshold (float)
                policy_shock_magnitude → treatment_time (int)
              Optional for all: verbose (bool), outcome_var, treatment_var,
              unit_col, time_col, controls, bandwidth.

        Returns:
            IVResult | TreatmentEffect | RDResult | DIDResult

        Raises:
            ValueError: If parameter is unknown or a required kwarg is missing.
        """
        from .causal_identification import (
            InstrumentalVariable,
            SyntheticControl,
            RegressionDiscontinuity,
            DifferenceInDifferences,
        )

        pid_map = {p.parameter: p for p in self.get_parameter_identifications()}
        if parameter not in pid_map:
            raise ValueError(
                f"Unknown parameter '{parameter}'. Available: {sorted(pid_map)}"
            )
        pid = pid_map[parameter]
        verbose = kwargs.pop("verbose", False)

        if pid.strategy == IdentificationStrategy.INSTRUMENTAL_VARIABLE:
            if "instrument_var" not in kwargs:
                raise ValueError("estimate_parameter('eta_D') requires instrument_var=<col>")
            return InstrumentalVariable(verbose=verbose).estimate(
                data=data,
                outcome_var=kwargs.pop("outcome_var", pid.outcome),
                treatment_var=kwargs.pop("treatment_var", pid.treatment),
                instrument_var=kwargs.pop("instrument_var"),
                controls=kwargs.pop("controls", None),
            )

        if pid.strategy == IdentificationStrategy.SYNTHETIC_CONTROL:
            for req in ("treated_unit", "control_units", "treatment_time"):
                if req not in kwargs:
                    raise ValueError(f"estimate_parameter('tau_K') requires {req}=<value>")
            return SyntheticControl(verbose=verbose).estimate_treatment_effect(
                data=data,
                treated_unit=kwargs.pop("treated_unit"),
                control_units=kwargs.pop("control_units"),
                treatment_time=kwargs.pop("treatment_time"),
                outcome_var=kwargs.pop("outcome_var", pid.outcome),
                unit_col=kwargs.pop("unit_col", "country"),
                time_col=kwargs.pop("time_col", "year"),
            )

        if pid.strategy == IdentificationStrategy.REGRESSION_DISCONTINUITY:
            if "threshold" not in kwargs:
                raise ValueError("estimate_parameter('alpha_P') requires threshold=<float>")
            return RegressionDiscontinuity(verbose=verbose).estimate(
                data=data,
                running_var=kwargs.pop("running_var", pid.treatment),
                outcome_var=kwargs.pop("outcome_var", pid.outcome),
                threshold=kwargs.pop("threshold"),
                bandwidth=kwargs.pop("bandwidth", None),
            )

        if pid.strategy == IdentificationStrategy.DIFFERENCE_IN_DIFFERENCES:
            if "treatment_time" not in kwargs:
                raise ValueError("estimate_parameter('policy_shock_magnitude') requires treatment_time=<int>")
            return DifferenceInDifferences(verbose=verbose).estimate(
                data=data,
                outcome_var=kwargs.pop("outcome_var", pid.outcome),
                treatment_col=kwargs.pop("treatment_col", "treated"),
                time_col=kwargs.pop("time_col", "year"),
                treatment_time=kwargs.pop("treatment_time"),
                unit_col=kwargs.pop("unit_col", "country"),
            )

        raise NotImplementedError(f"No estimator wired for strategy '{pid.strategy.value}'")


def demonstrate_identifiability() -> None:
    """Demo: Check identifiability of key causal effects (with do-calculus derivations)."""
    print("=" * 70)
    print("CAUSAL IDENTIFIABILITY ANALYSIS (Pearl do-calculus)")
    print("=" * 70)

    print("\n📐 Do-calculus (three rules, algebraic):")
    print(do_calculus.rule_1_statement())
    print(do_calculus.rule_2_statement())
    print(do_calculus.rule_3_statement())

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

    print("\n🔬 Identifiability Analysis (formula + derivation):\n")

    for treatment, outcome in queries:
        result = dag.is_identifiable(treatment, outcome)
        print(f"Query: P({outcome}|do({treatment}))")
        print(f"  Identifiable: {'✅ YES' if result.identifiable else '❌ NO'}")
        if result.identifiable:
            print(f"  Strategy: {result.strategy.value if result.strategy else 'N/A'}")
            if result.adjustment_set:
                print(f"  Adjustment set: {result.adjustment_set}")
            print(f"  Formula: {result.formula}")
            if result.derivation_steps:
                print("  Do-calculus derivation:")
                for step in result.derivation_steps:
                    print(f"    {step}")
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
