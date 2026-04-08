import logging
from io import StringIO
from dowhy import CausalModel
import networkx as nx
from networkx.drawing.nx_pydot import read_dot

logger = logging.getLogger(__name__)


def load_dag_dot(path: str) -> str:
    """
    Load a DOT-format DAG file and return it as a string.
    """
    logger.info(f"Loading DAG from DOT file: {path}")
    with open(path, "r") as f:
        return f.read()


def dot_to_digraph(dot_str: str) -> nx.DiGraph:
    """
    Convert DOT string to networkx DiGraph.
    
    Args:
        dot_str: DOT format graph string
        
    Returns:
        networkx DiGraph (converted from MultiDiGraph if needed)
    """
    dot_io = StringIO(dot_str)
    graph = read_dot(dot_io)
    
    # Convert MultiDiGraph to DiGraph if needed
    if isinstance(graph, nx.MultiDiGraph):
        digraph = nx.DiGraph()
        digraph.add_nodes_from(graph.nodes())
        digraph.add_edges_from(graph.edges())
        return digraph
    
    return graph


def causal_model_from_dag(df, treatment: str, outcome: str, graph_dot: str):
    """
    Construct and return a DoWhy CausalModel from a DOT graph string.
    """

    # Debugging output so we know what's happening:
    print("\n=== DEBUG: causal_model_from_dag ===")
    print("Treatment:", treatment)
    print("Outcome:", outcome)
    print("Graph DOT (first 200 chars):", graph_dot[:200])
    print("====================================\n")

    try:
        graph_digraph = dot_to_digraph(graph_dot)
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph_digraph
        )
        return model

    except Exception as e:
        logger.error("Failed to create CausalModel", exc_info=True)
        raise
