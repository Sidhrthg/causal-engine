"""
Pearl's do-calculus: algebraic rules for causal identifiability.

The do-calculus consists of three inference rules that allow rewriting
expressions involving do(X) into expressions involving only observational
probabilities. The DAG encodes conditional independence (d-separation);
the rules are the algebra that turns those independences into equalities.

This module also implements the complete Shpitser-Pearl ID algorithm
(2006), which is the most general procedure for identifying causal
effects from observational data: it determines P(Y|do(X)) for any
identifiable query and returns a symbolic estimand, or proves
non-identifiability (a "hedge").

References:
    Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
           Cambridge University Press. (Chapter 3, Theorem 3.4.1)
    Shpitser, I. & Pearl, J. (2006). Identification of Joint Interventional
           Distributions in Recursive Semi-Markovian Causal Models. AAAI-2006.
    Tian, J. & Pearl, J. (2002). A general identification condition for
           causal effects. AAAI-2002.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

# ---------------------------------------------------------------------------
# The three rules of do-calculus (algebraic form)
# ---------------------------------------------------------------------------
# Notation: G is the causal DAG. G_X is G with all edges *into* X removed.
#           G_{\overline{X}} is G with all edges *out of* X removed.
#           (Y ⊥⊥ Z | X, W)_G means Y and Z are d-separated given X,W in G.
# ---------------------------------------------------------------------------

DO_CALCULUS_RULES = r"""
Pearl's do-calculus (three rules):

Rule 1 (Insertion/deletion of observations):
  P(Y | do(X), Z, W) = P(Y | do(X), W)
  when (Y ⊥⊥ Z | X, W)_{G_{\overline{X}}}
  "Under do(X), if Y and Z are independent given X,W in the graph with
   incoming edges to X removed, we can drop Z from the conditioning set."

Rule 2 (Action/observation exchange):
  P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)
  when (Y ⊥⊥ Z | X, W)_{G_{\overline{X}\underline{Z}}}
  "Under do(X), if Y and Z are independent given X,W in the graph with
   incoming edges to X removed and outgoing edges from Z removed,
   then do(Z) can be replaced by seeing Z."

Rule 3 (Insertion/deletion of actions):
  P(Y | do(X), do(Z), W) = P(Y | do(X), W)
  when (Y ⊥⊥ Z | X, W)_{G_{\overline{X}\overline{Z(W)}}}
  where Z(W) = Z \\ An(W)_{G_{\overline{X}}} (nodes in Z that are not
  ancestors of any W-node in G_{\overline{X}}).
  "We can remove do(Z) if Z doesn't affect Y through paths not blocked by X,W."
"""


def rule_1_statement() -> str:
    """Return Rule 1 in algebraic form."""
    return (
        "Rule 1: P(Y|do(X),Z,W) = P(Y|do(X),W)  when  (Y ⊥⊥ Z | X,W)_{G_{\\overline{X}}}"
    )


def rule_2_statement() -> str:
    """Return Rule 2 in algebraic form (action/observation exchange)."""
    return (
        "Rule 2: P(Y|do(X),do(Z),W) = P(Y|do(X),Z,W)  when  (Y ⊥⊥ Z | X,W)_{G_{\\overline{X}\\underline{Z}}}"
    )


def rule_3_statement() -> str:
    """Return Rule 3 in algebraic form."""
    return (
        "Rule 3: P(Y|do(X),do(Z),W) = P(Y|do(X),W)  when  (Y ⊥⊥ Z | X,W)_{G_{\\overline{X}\\overline{Z(W)}}}"
    )


def derivation_steps_backdoor(
    treatment: str,
    outcome: str,
    adjustment_set: Set[str],
    formula: str,
) -> List[str]:
    """
    Algebraic derivation steps for backdoor adjustment.

    Backdoor adjustment is justified by do-calculus Rule 2: under the
    backdoor condition, do(treatment) can be exchanged for seeing treatment
    when we condition on Z (the adjustment set).
    """
    X, Y = treatment, outcome
    Z = adjustment_set
    z_str = ", ".join(sorted(Z)) if Z else "∅"
    steps = [
        "Query: P(Y | do(X)) with Y = " + outcome + ", X = " + treatment + ".",
        "",
        "Backdoor condition (graphical): Z = {" + z_str + "} blocks every path between X and Y that contains an arrow into X, and no node in Z is a descendant of X.",
        "",
        "Do-calculus justification:",
        "  • In the graph with outgoing edges from X removed (G_{X̲}), Z d-separates X from Y (only backdoor paths remain, all blocked by Z).",
        "  • Rule 2 (action/observation exchange): (Y ⊥⊥ X | Z)_{G_{X̲}} implies",
        "    P(Y | do(X), Z) = P(Y | X, Z).",
        "  • Marginalizing over Z (all observed):",
        "    P(Y | do(X)) = Σ_z P(Y | X, Z=z) P(Z=z).",
        "",
        "Hence: " + formula,
    ]
    return steps


def derivation_steps_frontdoor(
    treatment: str,
    outcome: str,
    mediator_set: Set[str],
    formula: str,
) -> List[str]:
    """
    Algebraic derivation steps for frontdoor adjustment.

    Frontdoor is justified by applying Rule 2 and Rule 3 to the mediator set M.
    """
    X, Y = treatment, outcome
    m_str = ", ".join(sorted(mediator_set))
    steps = [
        "Query: P(Y | do(X)) with Y = " + outcome + ", X = " + treatment + ".",
        "",
        "Frontdoor condition (graphical): M = {" + m_str + "} intercepts all directed paths from X to Y; no unblocked backdoor path from X to M; all backdoor paths from M to Y are blocked by X.",
        "",
        "Do-calculus justification:",
        "  • Rule 2 (action/observation exchange): P(Y|do(X)) = Σ_m P(Y|do(X),M=m) P(M=m|do(X)).",
        "  • P(M=m|do(X)) = P(M=m|X) (no backdoor X→M).",
        "  • Rule 3 then Rule 2: P(Y|do(X),M=m) = Σ_{x'} P(Y|M=m,X=x') P(X=x').",
        "  • Combining: P(Y|do(X)) = Σ_m P(M=m|X) Σ_{x'} P(Y|M=m,X=x') P(X=x').",
        "",
        "Hence: " + formula,
    ]
    return steps


def derivation_steps_trivial(treatment: str, outcome: str, formula: str) -> List[str]:
    """No confounding: P(Y|do(X)) = P(Y|X) (Rule 2 with empty adjustment)."""
    return [
        "Query: P(Y | do(X)) with Y = " + outcome + ", X = " + treatment + ".",
        "",
        "No confounding: X and Y are d-separated in the graph with edges into X removed (no backdoor paths).",
        "Rule 2 with Z = ∅: P(Y | do(X)) = P(Y | X).",
        "",
        "Hence: " + formula,
    ]


def derivation_steps_for_result(
    treatment: str,
    outcome: str,
    strategy: Optional[str],
    adjustment_set: Set[str],
    formula: str,
) -> List[str]:
    """
    Return the algebraic derivation steps for a given identification result.

    Used to attach Pearl-style algebraic justification to IdentificationResult.
    """
    if strategy == "backdoor_adjustment":
        return derivation_steps_backdoor(treatment, outcome, adjustment_set, formula)
    if strategy == "frontdoor_adjustment":
        return derivation_steps_frontdoor(treatment, outcome, adjustment_set, formula)
    if "P(" + outcome + "|do(" + treatment + ")) = P(" + outcome + "|" + treatment + ")" in formula:
        return derivation_steps_trivial(treatment, outcome, formula)
    return []


def format_derivation(derivation_steps: List[str], prefix: str = "  ") -> str:
    """Format derivation steps as a single string (e.g. for CLI or Gradio)."""
    return "\n".join(prefix + line if line else "" for line in derivation_steps)


# ===========================================================================
# ID Algorithm — Shpitser & Pearl (2006)
# ===========================================================================


class NonIdentifiableError(Exception):
    """
    Raised by the ID algorithm when P(Y|do(X)) cannot be identified from
    observational data.

    This corresponds to finding a "hedge" — a pair of subgraphs (F, F') such
    that F is a C-forest for (X,Y) in G and F' is a C-forest for (X,Y) in
    G[An(Y)], indicating an irresolvable hidden confounding structure.

    Practically: you need an RCT, an instrument, or additional proxy variables.
    """
    def __init__(self, G_nodes: Set[str], S: Set[str]) -> None:
        self.G_nodes = G_nodes
        self.S = S
        super().__init__(
            f"P(Y|do(X)) is not identifiable. Hedge found: "
            f"C-component S={sorted(S)} spans the full node set V={sorted(G_nodes)}. "
            f"Requires instrument, RCT, or additional structure."
        )


def _bidirected_from_hidden(dag: Any) -> Set[FrozenSet[str]]:
    """
    Derive bidirected edges (hidden common causes) from unobserved nodes.

    In the ADMG (Acyclic Directed Mixed Graph) representation used by the ID
    algorithm, each unobserved node U with children {A, B, ...} in the full
    DAG is replaced by bidirected edges A↔B, A↔C, ... among its observed
    descendants (direct children only, or transitively through other hidden).

    This implements the standard "latent projection" onto observed nodes.
    """
    # Flatten: for each hidden node, trace observed descendants (via hidden paths)
    def _obs_descendants_of_hidden(u: str) -> Set[str]:
        """Observed nodes reachable from hidden u through hidden-only paths."""
        obs: Set[str] = set()
        stack = [u]
        visited: Set[str] = set([u])
        while stack:
            cur = stack.pop()
            for child in dag.graph.successors(cur):
                if child in dag.unobserved_vars:
                    if child not in visited:
                        visited.add(child)
                        stack.append(child)
                else:
                    obs.add(child)
        return obs

    bidirected: Set[FrozenSet[str]] = set()
    for u in dag.unobserved_vars:
        obs_children = _obs_descendants_of_hidden(u)
        obs_list = sorted(obs_children)
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                bidirected.add(frozenset([obs_list[i], obs_list[j]]))
    return bidirected


def _c_components(
    nodes: FrozenSet[str],
    bidirected: Set[FrozenSet[str]],
) -> List[FrozenSet[str]]:
    """
    Find C-components (connected components under bidirected edges).

    A C-component is a maximally connected set of variables under the
    bidirected-edge subgraph. Variables in the same C-component share
    unobserved common causes that make causal identification harder.

    Reference: Tian & Pearl (2002). A general identification condition
    for causal effects. AAAI-2002.
    """
    adj: Dict[str, Set[str]] = {n: set() for n in nodes}
    for edge in bidirected:
        edge_list = list(edge)
        if len(edge_list) == 2:
            a, b = edge_list[0], edge_list[1]
            if a in nodes and b in nodes:
                adj[a].add(b)
                adj[b].add(a)

    visited: Set[str] = set()
    components: List[FrozenSet[str]] = []

    for start in sorted(nodes):  # sorted for determinism
        if start in visited:
            continue
        component: Set[str] = set()
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in sorted(adj[node]):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(frozenset(component))

    return components


def _ancestors_in_subgraph(nodes: FrozenSet[str], graph: nx.DiGraph) -> FrozenSet[str]:
    """All ancestors (including self) of 'nodes' in the given directed graph."""
    ancestors: Set[str] = set(nodes)
    for n in nodes:
        if graph.has_node(n):
            ancestors.update(nx.ancestors(graph, n))
    return frozenset(ancestors & set(graph.nodes))


def _mutilate(X: FrozenSet[str], graph: nx.DiGraph) -> nx.DiGraph:
    """Remove all incoming edges to nodes in X (do-operator / graph surgery)."""
    g = graph.copy()
    for x in X:
        for pred in list(g.predecessors(x)):
            g.remove_edge(pred, x)
    return g


def _restrict_bidirected(
    nodes: FrozenSet[str], bidirected: Set[FrozenSet[str]]
) -> Set[FrozenSet[str]]:
    """Keep only bidirected edges whose both endpoints are in 'nodes'."""
    return {e for e in bidirected if e <= nodes}


def _make_kernel(vi: str, scope: FrozenSet[str], topo: List[str], P_label: str) -> str:
    """
    Build the Markov kernel P(vi | predecessors in topo that are in scope).

    In Line 7 of the ID algorithm, the formula for each node vi in S' is
    P(vi | V_{pi<i} intersect S', V_{pi<i} setminus S'), which equals
    P(vi | all predecessors of vi in the full topological order),
    but expressed relative to the scope S'.
    """
    try:
        idx = topo.index(vi)
    except ValueError:
        return f"{P_label}({vi})"
    all_preds = topo[:idx]
    in_scope = [v for v in all_preds if v in scope]
    out_scope = [v for v in all_preds if v not in scope]
    cond = in_scope + out_scope
    if not cond:
        return f"{P_label}({vi})"
    return f"{P_label}({vi}|{','.join(cond)})"


def _id_recursive(
    Y: FrozenSet[str],
    X: FrozenSet[str],
    G: nx.DiGraph,
    bidirected: Set[FrozenSet[str]],
    topo: List[str],
    P_label: str,
    depth: int = 0,
    trace: Optional[List[str]] = None,
) -> str:
    """
    Recursive Shpitser-Pearl ID algorithm (AAAI-2006 version).

    Given a subgraph G of the ADMG, computes a symbolic expression for
    P(Y | do(X)) in terms of observational probabilities, or raises
    NonIdentifiableError if the effect cannot be identified.

    Args:
        Y:         Target variables (set of node names).
        X:         Intervention variables (set of node names).
        G:         Current observed-variable subgraph (nx.DiGraph).
        bidirected: Bidirected edges in G (from hidden common causes).
        topo:      Full topological ordering (may contain nodes outside G).
        P_label:   Symbolic label for the current distribution (e.g., "P").
        depth:     Recursion depth (safety limit = 30).
        trace:     If provided, append algorithm step descriptions.

    Returns:
        Symbolic string for the estimand.
    """
    if depth > 30:
        raise NonIdentifiableError(set(G.nodes), set(Y))

    V = frozenset(G.nodes)
    indent = "  " * depth

    # ------------------------------------------------------------------
    # Line 1: No intervention — P(Y) = Σ_{V\Y} P(V)
    # ------------------------------------------------------------------
    if not X:
        rest = V - Y
        if trace is not None:
            trace.append(f"{indent}L1: X=∅ → marginalize {sorted(rest)} from {P_label}")
        if not rest:
            return f"{P_label}({','.join(sorted(Y))})"
        return f"Σ_{{{','.join(sorted(rest))}}} {P_label}({','.join(sorted(V))})"

    # ------------------------------------------------------------------
    # Line 2: Restrict to ancestors of Y in G
    # ------------------------------------------------------------------
    An_Y = _ancestors_in_subgraph(Y, G)
    if An_Y != V:
        marg = V - An_Y
        new_label = f"Σ_{{{','.join(sorted(marg))}}} {P_label}" if marg else P_label
        G2 = nx.subgraph(G, An_Y).copy()
        bd2 = _restrict_bidirected(An_Y, bidirected)
        if trace is not None:
            trace.append(
                f"{indent}L2: An(Y)={sorted(An_Y)} ≠ V → restrict, marginalize {sorted(marg)}"
            )
        return _id_recursive(Y, X & An_Y, G2, bd2, topo, new_label, depth + 1, trace)

    # ------------------------------------------------------------------
    # Line 3: Add non-ancestors of Y in G_{X̄} to intervention set
    # ------------------------------------------------------------------
    G_mutX = _mutilate(X, G)
    An_Y_mut = _ancestors_in_subgraph(Y, G_mutX)
    W = (V - X) - An_Y_mut
    if W:
        if trace is not None:
            trace.append(
                f"{indent}L3: W={sorted(W)} not ancestors of Y in G_X̄ → add to X"
            )
        return _id_recursive(Y, X | W, G, bidirected, topo, P_label, depth + 1, trace)

    # ------------------------------------------------------------------
    # Line 4: Decompose over C-components of G \ X
    # ------------------------------------------------------------------
    V_mX = V - X
    G_mX = nx.subgraph(G, V_mX).copy()
    bd_mX = _restrict_bidirected(V_mX, bidirected)
    cc_GmX = _c_components(V_mX, bd_mX)

    if len(cc_GmX) > 1:
        summand = V - (Y | X)
        pieces: List[str] = []
        for Si in cc_GmX:
            pieces.append(
                _id_recursive(Si, V - Si, G, bidirected, topo, P_label, depth + 1, trace)
            )
        if trace is not None:
            trace.append(
                f"{indent}L4: C(G\\X) = {[sorted(c) for c in cc_GmX]} "
                f"→ decompose into {len(cc_GmX)} sub-problems"
            )
        product = " · ".join(f"[{p}]" for p in pieces)
        if summand:
            return f"Σ_{{{','.join(sorted(summand))}}} {product}"
        return product

    # G\X has a single C-component S = V\X
    S = V_mX

    # ------------------------------------------------------------------
    # Lines 5–8: G\X is one C-component
    # ------------------------------------------------------------------
    cc_G = _c_components(V, bidirected)

    # Line 6: If C(G) = {G} (entire observed graph is one C-component), FAIL
    if len(cc_G) == 1 and cc_G[0] == V:
        if trace is not None:
            trace.append(
                f"{indent}L6: C(G)={{G}} and S=V\\X → FAIL (hedge found)"
            )
        raise NonIdentifiableError(set(V), set(S))

    # Line 7: If S itself is a C-component of G, use its kernel formula
    for Sp in cc_G:
        if S == Sp:
            topo_Sp = [v for v in topo if v in Sp]
            kernels = [_make_kernel(vi, Sp, topo, P_label) for vi in topo_Sp]
            product = " · ".join(kernels)
            summand = S - Y
            if trace is not None:
                trace.append(
                    f"{indent}L7: S={sorted(S)} ∈ C(G) → "
                    f"Σ_{{{sorted(summand)}}} ∏_{{Vi∈S}} P(Vi|...)"
                )
            if summand:
                return f"Σ_{{{','.join(sorted(summand))}}} [{product}]"
            return f"[{product}]"

    # Line 8: S ⊂ S' for some S' ∈ C(G) — recurse into G[S']
    for Sp in cc_G:
        if S < Sp:  # S is a proper subset of Sp
            topo_Sp = [v for v in topo if v in Sp]
            kernels = [_make_kernel(vi, Sp, topo, P_label) for vi in topo_Sp]
            new_P = "·".join(f"[{k}]" for k in kernels)
            G_Sp = nx.subgraph(G, Sp).copy()
            bd_Sp = _restrict_bidirected(Sp, bidirected)
            if trace is not None:
                trace.append(
                    f"{indent}L8: S={sorted(S)} ⊂ S'={sorted(Sp)} ∈ C(G) → "
                    f"recurse into G[S']"
                )
            return _id_recursive(
                Y, X & Sp, G_Sp, bd_Sp, topo,
                f"({new_P})", depth + 1, trace
            )

    # Should never reach here if the algorithm is correct
    raise NonIdentifiableError(set(V), set(S))  # pragma: no cover


def id_algorithm(dag: Any, treatment: str, outcome: str) -> Dict:
    """
    Run the complete Shpitser-Pearl ID algorithm to determine if
    P(outcome | do(treatment)) is identifiable from observational data.

    This is strictly more powerful than backdoor/frontdoor criteria:
    it handles any identifiable causal effect in a DAG with arbitrary
    hidden confounders, and constructively returns the symbolic estimand.

    The algorithm works on the ADMG (Acyclic Directed Mixed Graph) derived
    from the CausalDAG by projecting out unobserved nodes as bidirected edges.

    Args:
        dag:       A CausalDAG with observed_vars and unobserved_vars sets.
        treatment: Name of the treatment (intervention) variable.
        outcome:   Name of the outcome variable.

    Returns:
        dict with:
            identifiable (bool)
            formula (str): symbolic estimand in observational probabilities
            strategy (str): "id_algorithm"
            derivation_steps (List[str]): step-by-step trace
            bidirected_edges (List): derived hidden-confounder edges
            c_components (List): C-components of the full observed graph
            error (Optional[str]): description if not identifiable
    """
    obs_vars = frozenset(dag.observed_vars)

    # Build observed-variable-only directed graph
    G = nx.DiGraph()
    for n in dag.graph.nodes:
        if n in obs_vars:
            G.add_node(n)
    for u, v in dag.graph.edges:
        if u in obs_vars and v in obs_vars:
            G.add_edge(u, v)

    # Derive bidirected edges from hidden common causes
    bidirected = _bidirected_from_hidden(dag)
    bidirected = _restrict_bidirected(obs_vars, bidirected)

    # Topological ordering
    try:
        topo = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return {
            "identifiable": False,
            "formula": "Not identifiable: cyclic graph",
            "strategy": "id_algorithm",
            "derivation_steps": ["Graph contains a cycle — ID algorithm requires a DAG."],
            "bidirected_edges": [],
            "c_components": [],
            "error": "Cyclic graph",
        }

    # Validate inputs are observed
    if treatment not in obs_vars:
        return {
            "identifiable": False,
            "formula": f"Treatment '{treatment}' is unobserved",
            "strategy": "id_algorithm",
            "derivation_steps": [f"Treatment must be an observed variable."],
            "bidirected_edges": [],
            "c_components": [],
            "error": f"Unobserved treatment: {treatment}",
        }
    if outcome not in obs_vars:
        return {
            "identifiable": False,
            "formula": f"Outcome '{outcome}' is unobserved",
            "strategy": "id_algorithm",
            "derivation_steps": [f"Outcome must be an observed variable."],
            "bidirected_edges": [],
            "c_components": [],
            "error": f"Unobserved outcome: {outcome}",
        }

    cc_full = _c_components(obs_vars, bidirected)
    bd_list = [sorted(e) for e in bidirected]

    trace: List[str] = []
    try:
        formula = _id_recursive(
            frozenset([outcome]),
            frozenset([treatment]),
            G, bidirected, topo, "P",
            depth=0, trace=trace,
        )
        steps = [
            f"ID Algorithm (Shpitser & Pearl, 2006)",
            f"Query: P({outcome} | do({treatment}))",
            "",
            f"Observed variables ({len(obs_vars)}): {sorted(obs_vars)}",
            f"Hidden variables ({len(dag.unobserved_vars)}): {sorted(dag.unobserved_vars)}",
            f"Bidirected edges (latent projection): {bd_list if bd_list else '∅'}",
            f"C-components of observed graph: {[sorted(c) for c in cc_full]}",
            "",
            "Algorithm trace:",
        ] + [f"  {t}" for t in trace] + [
            "",
            f"Symbolic estimand: {formula}",
        ]
        return {
            "identifiable": True,
            "formula": formula,
            "strategy": "id_algorithm",
            "derivation_steps": steps,
            "bidirected_edges": bd_list,
            "c_components": [sorted(c) for c in cc_full],
            "error": None,
        }

    except NonIdentifiableError as e:
        steps = [
            f"ID Algorithm (Shpitser & Pearl, 2006)",
            f"Query: P({outcome} | do({treatment}))",
            "",
            f"Result: NOT IDENTIFIABLE",
            "",
            f"A 'hedge' was found — a C-component S that spans the full observed",
            f"graph, indicating irresolvable hidden confounding.",
            "",
            f"  C-component S = {sorted(e.S)}",
            f"  Graph nodes  V = {sorted(e.G_nodes)}",
            "",
            "Algorithm trace:",
        ] + [f"  {t}" for t in trace] + [
            "",
            "To identify this effect, you need one of:",
            "  • An instrumental variable (exogenous variation in treatment)",
            "  • A randomized experiment (breaks all back-door paths)",
            "  • Additional observed proxy variables to break the hedge",
            "  • Weaker structural assumptions (bounds, partial identification)",
        ]
        return {
            "identifiable": False,
            "formula": "Not identifiable — hedge (irresolvable hidden confounding)",
            "strategy": "id_algorithm",
            "derivation_steps": steps,
            "bidirected_edges": bd_list,
            "c_components": [sorted(c) for c in cc_full],
            "error": str(e),
        }
