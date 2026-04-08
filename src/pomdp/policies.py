"""
Policy functions for POMDP.
"""

import numpy as np
from typing import Callable

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


