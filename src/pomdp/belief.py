"""
Belief state update for POMDP.
"""

import numpy as np
from typing import Tuple

from src.pomdp.schema import POMDP
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def belief_update(
    pomdp: POMDP,
    belief: np.ndarray,
    action: str,
    observation: str,
) -> np.ndarray:
    """
    Update belief state: b' ∝ Z[a][:,o] * (T[a].T @ b)
    
    Args:
        pomdp: POMDP model
        belief: Current belief vector (|S|,)
        action: Action taken
        observation: Observation received
        
    Returns:
        Updated belief vector (normalized)
    """
    if action not in pomdp.A:
        raise ValueError(f"Action {action} not in POMDP actions")
    
    if observation not in pomdp.O:
        logger.warning(f"Observation {observation} not in POMDP observations, using uniform")
        # Handle unknown observations by returning uniform belief
        return np.ones(len(pomdp.S)) / len(pomdp.S)
    
    # Get indices
    a_idx = pomdp.A.index(action)
    o_idx = pomdp.O.index(observation)
    
    # b' = Z[a][:,o] * (T[a].T @ b)
    # First: predict next belief: T[a].T @ b
    T_a = pomdp.T[action]
    predicted_belief = T_a.T @ belief
    
    # Second: update with observation: Z[a][:,o] * predicted_belief
    Z_a = pomdp.Z[action]
    obs_likelihood = Z_a[:, o_idx]
    
    new_belief = obs_likelihood * predicted_belief
    
    # Normalize
    norm = new_belief.sum()
    if norm < 1e-10:
        logger.warning("Belief update resulted in near-zero probability, using uniform")
        return np.ones(len(pomdp.S)) / len(pomdp.S)
    
    new_belief = new_belief / norm
    
    # Ensure non-negativity
    new_belief = np.maximum(new_belief, 0.0)
    new_belief = new_belief / new_belief.sum()
    
    return new_belief


