import logging
from dowhy import CausalModel

logger = logging.getLogger(__name__)


def load_dag_dot(path: str) -> str:
    """
    Load a DOT-format DAG file and return it as a string.
    """
    logger.info(f"Loading DAG from DOT file: {path}")
    with open(path, "r") as f:
        return f.read()


def causal_model_from_dag(
    df,
    treatment: str,
    outcome: str,
    graph_dot: str | None = None,
    graph_path: str | None = None,
):
    """
    Construct and return a DoWhy CausalModel from a DOT graph.

    DoWhy accepts either a path to a .dot file or a graph string. Pass graph_path
    when you have a file (preferred) so DoWhy can load it directly.
    """
    if graph_path:
        graph_arg = graph_path
    elif graph_dot:
        graph_arg = graph_dot
    else:
        raise ValueError("Provide either graph_dot or graph_path")

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph_arg,
        )
        return model
    except Exception as e:
        logger.error("Failed to create CausalModel", exc_info=True)
        raise
