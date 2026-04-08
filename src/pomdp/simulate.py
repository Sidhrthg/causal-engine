"""
POMDP simulation and rollouts.
"""

from typing import List, Tuple, Callable, Dict, Any, Optional
import numpy as np

from src.pomdp.schema import POMDP
from src.pomdp.belief import belief_update
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def rollout(
    pomdp: POMDP,
    policy: Callable[[np.ndarray], str],
    start_belief: np.ndarray,
    horizon: int = 25,
    true_state: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Simulate a POMDP rollout.
    
    Args:
        pomdp: POMDP model
        policy: Policy function that maps belief to action
        start_belief: Initial belief vector
        horizon: Number of steps to simulate
        true_state: True hidden state (if None, sample from belief)
        rng: Random number generator
        
    Returns:
        Dict with keys:
            - total_reward: Cumulative reward
            - belief_history: List of belief vectors
            - action_history: List of actions taken
            - observation_history: List of observations
            - state_history: List of true states
            - reward_history: List of step rewards
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Initialize
    belief = start_belief.copy()
    if true_state is None:
        # Sample initial state from belief
        true_state_idx = rng.choice(len(pomdp.S), p=belief)
        true_state = pomdp.S[true_state_idx]
    else:
        true_state_idx = pomdp.S.index(true_state)
    
    belief_history = [belief.copy()]
    action_history = []
    observation_history = []
    state_history = [true_state]
    reward_history = []
    total_reward = 0.0
    
    current_state_idx = true_state_idx
    
    for step in range(horizon):
        # Select action
        action = policy(belief)
        action_history.append(action)
        
        # Sample next state from T[a][s, :]
        T_a = pomdp.T[action]
        next_state_idx = rng.choice(len(pomdp.S), p=T_a[current_state_idx, :])
        next_state = pomdp.S[next_state_idx]
        state_history.append(next_state)
        
        # Sample observation from Z[a][s', :]
        Z_a = pomdp.Z[action]
        obs_idx = rng.choice(len(pomdp.O), p=Z_a[next_state_idx, :])
        observation = pomdp.O[obs_idx]
        observation_history.append(observation)
        
        # Get reward
        R_a = pomdp.R[action]
        reward = R_a[current_state_idx, next_state_idx]
        reward_history.append(reward)
        total_reward += reward
        
        # Update belief
        belief = belief_update(pomdp, belief, action, observation)
        belief_history.append(belief.copy())
        
        # Update state
        current_state_idx = next_state_idx
    
    return {
        "total_reward": total_reward,
        "belief_history": belief_history,
        "action_history": action_history,
        "observation_history": observation_history,
        "state_history": state_history,
        "reward_history": reward_history,
    }

