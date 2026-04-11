"""
Policy functions for POMDP.
"""

import numpy as np
from typing import Callable, Dict, Optional

from src.pomdp.schema import POMDP
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def policy_myopic(
    pomdp: POMDP,
    belief: np.ndarray,
) -> str:
    """
    Myopic policy: choose action maximizing expected immediate reward.
    
    E[R | b, a] = sum_{s,s'} b[s] * T[a][s,s'] * R[a][s,s']
    
    Args:
        pomdp: POMDP model
        belief: Current belief vector
        
    Returns:
        Action label
    """
    best_action = None
    best_value = -np.inf
    
    for a in pomdp.A:
        T_a = pomdp.T[a]
        R_a = pomdp.R[a]
        
        # Expected immediate reward
        expected_reward = np.sum(belief[:, None] * T_a * R_a)
        
        if expected_reward > best_value:
            best_value = expected_reward
            best_action = a
    
    return best_action


def policy_threshold(
    pomdp: POMDP,
    belief: np.ndarray,
    threshold: float = 0.5,
    repair_action: str = "repair",
    default_action: str = "ignore",
) -> str:
    """
    Threshold policy: if P(failed) > tau then repair else ignore.
    
    Args:
        pomdp: POMDP model
        belief: Current belief vector
        threshold: Probability threshold for repair
        repair_action: Action to take if threshold exceeded
        default_action: Action to take otherwise
        
    Returns:
        Action label
    """
    # Find failed state indices
    failed_state_indices = [
        i for i, s in enumerate(pomdp.S)
        if "failed" in s.lower() or "failure" in s.lower()
    ]
    
    if not failed_state_indices:
        logger.warning("No 'failed' state found, using default action")
        return default_action
    
    # Compute probability of failure
    failure_prob = np.sum(belief[failed_state_indices])
    
    if failure_prob > threshold:
        if repair_action not in pomdp.A:
            logger.warning(f"Repair action {repair_action} not in POMDP, using default")
            return default_action
        return repair_action
    else:
        return default_action


# ---------------------------------------------------------------------------
# QMDP policy
# ---------------------------------------------------------------------------

def _compute_q_values(
    pomdp: POMDP,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> Dict[str, np.ndarray]:
    """
    Value iteration on the fully-observable MDP induced by *pomdp*.

    Returns Q[a] as a 1-D array of shape (|S|,) where
        Q[a][s] = Σ_{s'} T[a][s,s'] * (R[a][s,s'] + γ * V*(s'))

    Complexity: O(max_iter * |A| * |S|^2).  Converges in O(log(1/tol) /
    log(1/γ)) iterations in the worst case.
    """
    n_states = len(pomdp.S)
    V = np.zeros(n_states)

    for iteration in range(max_iter):
        V_new = np.full(n_states, -np.inf)
        for a in pomdp.A:
            T_a = pomdp.T[a]       # (|S|, |S|)
            R_a = pomdp.R[a]       # (|S|, |S|)
            # Q(s, a) = Σ_{s'} T[a][s,s'] * (R[a][s,s'] + γ * V(s'))
            Q_a = np.sum(T_a * (R_a + pomdp.gamma * V[None, :]), axis=1)
            V_new = np.maximum(V_new, Q_a)

        delta = np.max(np.abs(V_new - V))
        V = V_new
        if delta < tol:
            logger.debug(f"Value iteration converged at iteration {iteration + 1}")
            break
    else:
        logger.warning(
            f"Value iteration did not converge in {max_iter} iterations "
            f"(final delta={delta:.2e})"
        )

    # Recompute Q from converged V
    Q: Dict[str, np.ndarray] = {}
    for a in pomdp.A:
        T_a = pomdp.T[a]
        R_a = pomdp.R[a]
        Q[a] = np.sum(T_a * (R_a + pomdp.gamma * V[None, :]), axis=1)

    return Q


def make_policy_qmdp(
    pomdp: POMDP,
    tol: float = 1e-6,
    max_iter: int = 10_000,
) -> Callable[[np.ndarray], str]:
    """
    Build a QMDP policy closure with pre-computed Q-values.

    QMDP approximates the optimal POMDP policy by assuming full observability
    after the current step:

        π(b) = argmax_a  Σ_s  b[s] · Q*(s, a)

    This is exact when there is no further information to gain (i.e. after
    the belief concentrates), and a useful heuristic in general.

    Usage::

        policy = make_policy_qmdp(pomdp)
        result = rollout(pomdp, policy, start_belief)

    Args:
        pomdp: Fitted POMDP model.
        tol: Convergence tolerance for value iteration.
        max_iter: Maximum value-iteration steps.

    Returns:
        A callable ``policy(belief) -> action_label``.
    """
    Q = _compute_q_values(pomdp, tol=tol, max_iter=max_iter)
    logger.info(f"QMDP: pre-computed Q-values for actions {list(Q.keys())}")

    def policy(belief: np.ndarray) -> str:
        return max(pomdp.A, key=lambda a: float(np.dot(belief, Q[a])))

    return policy


def policy_qmdp(
    pomdp: POMDP,
    belief: np.ndarray,
    q_values: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """
    QMDP policy: single-call form (recomputes Q-values every call).

    For repeated calls in a rollout use :func:`make_policy_qmdp` instead,
    which caches the Q-values.

    Args:
        pomdp: POMDP model.
        belief: Current belief vector.
        q_values: Pre-computed Q-values from ``_compute_q_values``.
            If *None*, they are recomputed (slow for long rollouts).

    Returns:
        Action label.
    """
    if q_values is None:
        q_values = _compute_q_values(pomdp)
    return max(pomdp.A, key=lambda a: float(np.dot(belief, q_values[a])))

