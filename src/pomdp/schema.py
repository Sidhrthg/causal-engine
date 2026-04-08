"""
POMDP schema definitions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class POMDP:
    """
    Partially Observable Markov Decision Process.
    
    Attributes:
        S: List of state labels
        A: List of action labels
        O: List of observation labels
        T: Transition probabilities T[a][s, s'] = P(s' | s, a)
        Z: Observation probabilities Z[a][s, o] = P(o | s', a)
        R: Reward function R[a][s, s'] = reward for transition s -> s' under action a
        gamma: Discount factor
    """
    S: List[str]
    A: List[str]
    O: List[str]
    T: Dict[str, np.ndarray]  # T[a] is |S| x |S| matrix
    Z: Dict[str, np.ndarray]  # Z[a] is |S| x |O| matrix
    R: Dict[str, np.ndarray]  # R[a] is |S| x |S| matrix
    gamma: float = 0.95

    def __post_init__(self):
        """Validate POMDP structure."""
        n_states = len(self.S)
        n_actions = len(self.A)
        n_obs = len(self.O)

        # Check T matrices
        for a in self.A:
            if a not in self.T:
                raise ValueError(f"Missing transition matrix for action {a}")
            T_a = self.T[a]
            if T_a.shape != (n_states, n_states):
                raise ValueError(
                    f"T[{a}] has shape {T_a.shape}, expected ({n_states}, {n_states})"
                )
            if not np.allclose(T_a.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"T[{a}] rows do not sum to 1")

        # Check Z matrices
        for a in self.A:
            if a not in self.Z:
                raise ValueError(f"Missing emission matrix for action {a}")
            Z_a = self.Z[a]
            if Z_a.shape != (n_states, n_obs):
                raise ValueError(
                    f"Z[{a}] has shape {Z_a.shape}, expected ({n_states}, {n_obs})"
                )
            if not np.allclose(Z_a.sum(axis=1), 1.0, atol=1e-6):
                raise ValueError(f"Z[{a}] rows do not sum to 1")

        # Check R matrices (optional validation)
        for a in self.A:
            if a not in self.R:
                raise ValueError(f"Missing reward matrix for action {a}")
            R_a = self.R[a]
            if R_a.shape != (n_states, n_states):
                raise ValueError(
                    f"R[{a}] has shape {R_a.shape}, expected ({n_states}, {n_states})"
                )


@dataclass
class Priors:
    """
    Prior knowledge for POMDP parameter estimation.
    
    Attributes:
        priors_T_alpha: Dirichlet alpha parameters for transitions
            priors_T_alpha[a][s, s'] = alpha for P(s' | s, a)
        priors_Z_alpha: Dirichlet alpha parameters for emissions
            priors_Z_alpha[a][s, o] = alpha for P(o | s', a)
        action_costs: Cost per action (negative rewards)
        failure_penalty: Large negative reward for failure states
        thresholds: Domain-specific thresholds (e.g., failure_prob_threshold)
    """
    priors_T_alpha: Optional[Dict[str, np.ndarray]] = None
    priors_Z_alpha: Optional[Dict[str, np.ndarray]] = None
    action_costs: Optional[Dict[str, float]] = None
    failure_penalty: float = -100.0
    thresholds: Dict[str, float] = field(default_factory=lambda: {"failure_prob": 0.5})

    def __post_init__(self):
        """Validate priors if provided."""
        if self.priors_T_alpha is not None:
            for a, alpha_mat in self.priors_T_alpha.items():
                if np.any(alpha_mat < 0):
                    raise ValueError(f"priors_T_alpha[{a}] contains negative values")
        
        if self.priors_Z_alpha is not None:
            for a, alpha_mat in self.priors_Z_alpha.items():
                if np.any(alpha_mat < 0):
                    raise ValueError(f"priors_Z_alpha[{a}] contains negative values")
        
        if self.action_costs is not None:
            for a, cost in self.action_costs.items():
                if cost < 0:
                    raise ValueError(f"action_costs[{a}] should be non-negative (costs are subtracted)")


