"""
POMDP parameter estimation from data.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.pomdp.schema import POMDP, Priors
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def fit_pomdp(
    df: pd.DataFrame,
    state_col: Optional[str] = None,
    action_col: Optional[str] = None,
    obs_col: str = "observation",
    episode_col: str = "episode_id",
    default_actions: Optional[List[str]] = None,
    priors: Optional[Priors] = None,
    smoothing: float = 1.0,  # Laplace smoothing
    rng: Optional[np.random.Generator] = None,
) -> Tuple[POMDP, Dict[str, Any]]:
    """
    Fit POMDP transition and emission matrices from data.
    
    Args:
        df: DataFrame with columns: episode_id, observation, [state], [action]
        state_col: Column name for true states (if available, otherwise infer)
        action_col: Column name for actions (if None, use default_actions)
        obs_col: Column name for observations
        episode_col: Column name for episode IDs
        default_actions: Default actions if action_col is None
        priors: Optional Priors object for constrained fitting
        smoothing: Laplace smoothing parameter (alpha)
        rng: Random number generator
        
    Returns:
        Tuple of (fitted POMDP, metadata dict)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Infer states if not provided
    if state_col is None or state_col not in df.columns:
        logger.info("No state column found, inferring states from failure pattern")
        # Use Failure column if available, else infer from observation patterns
        if "Failure" in df.columns:
            df = df.copy()
            df["state"] = df["Failure"].map({0: "healthy", 1: "failed"})
            state_col = "state"
        else:
            # Infer states: use quantiles of observations to create states
            logger.warning("Inferring states heuristically from observations")
            df = df.copy()
            obs_counts = df.groupby(obs_col, observed=True).size()
            # Simple heuristic: bottom 20% = healthy, top 20% = failed, middle = degrading
            thresholds = obs_counts.quantile([0.2, 0.8])
            def infer_state(obs):
                count = obs_counts.get(obs, 0)
                if count <= thresholds.iloc[0]:
                    return "healthy"
                elif count >= thresholds.iloc[1]:
                    return "failed"
                else:
                    return "degrading"
            df["state"] = df[obs_col].apply(infer_state)
            state_col = "state"
    
    # Infer actions if not provided
    if action_col is None or action_col not in df.columns:
        logger.info("No action column found, using default actions")
        if default_actions is None:
            default_actions = ["ignore", "calibrate", "repair"]
        df = df.copy()
        # Assign actions uniformly at random per episode (for demonstration)
        df["action"] = df.groupby(episode_col, observed=True)[episode_col].transform(
            lambda x: rng.choice(default_actions, size=len(x))
        )
        action_col = "action"
    
    # Get unique states, actions, observations
    states = sorted(df[state_col].unique().tolist())
    actions = sorted(df[action_col].unique().tolist())
    observations = sorted(df[obs_col].unique().tolist())
    
    logger.info(f"States: {states}")
    logger.info(f"Actions: {actions}")
    logger.info(f"Observations: {len(observations)} unique")
    
    n_states = len(states)
    n_actions = len(actions)
    n_obs = len(observations)
    
    # Create state/action/obs to index mappings
    state_to_idx = {s: i for i, s in enumerate(states)}
    action_to_idx = {a: i for i, a in enumerate(actions)}
    obs_to_idx = {o: i for i, o in enumerate(observations)}
    
    # Initialize count matrices
    T_counts = {a: np.zeros((n_states, n_states)) for a in actions}
    Z_counts = {a: np.zeros((n_states, n_obs)) for a in actions}
    
    # Count transitions and emissions from episodes
    for episode_id in df[episode_col].unique():
        episode_df = df[df[episode_col] == episode_id].sort_index()
        
        for i in range(len(episode_df) - 1):
            curr_row = episode_df.iloc[i]
            next_row = episode_df.iloc[i + 1]
            
            curr_state = curr_row[state_col]
            next_state = next_row[state_col]
            action = curr_row[action_col]
            obs = next_row[obs_col]  # Observation after action
            
            # Count transition
            s_idx = state_to_idx[curr_state]
            s_next_idx = state_to_idx[next_state]
            T_counts[action][s_idx, s_next_idx] += 1
            
            # Count emission (observation given next state and action)
            o_idx = obs_to_idx[obs]
            Z_counts[action][s_next_idx, o_idx] += 1
    
    # Apply priors (Dirichlet smoothing)
    T_alpha = {a: np.full((n_states, n_states), smoothing) for a in actions}
    Z_alpha = {a: np.full((n_states, n_obs), smoothing) for a in actions}
    
    if priors is not None:
        if priors.priors_T_alpha is not None:
            for a in actions:
                if a in priors.priors_T_alpha:
                    T_alpha[a] = priors.priors_T_alpha[a]
                    logger.info(f"Using custom priors for T[{a}]")
        
        if priors.priors_Z_alpha is not None:
            for a in actions:
                if a in priors.priors_Z_alpha:
                    Z_alpha[a] = priors.priors_Z_alpha[a]
                    logger.info(f"Using custom priors for Z[{a}]")
    
    # Estimate probabilities with Dirichlet smoothing
    T = {}
    Z = {}
    
    for a in actions:
        # Transition probabilities
        counts_smoothed = T_counts[a] + T_alpha[a]
        row_sums = counts_smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        T[a] = counts_smoothed / row_sums
        logger.info(f"T[{a}] fitted: shape {T[a].shape}, rows sum to {T[a].sum(axis=1)}")
        
        # Emission probabilities
        counts_smoothed = Z_counts[a] + Z_alpha[a]
        row_sums = counts_smoothed.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        Z[a] = counts_smoothed / row_sums
        logger.info(f"Z[{a}] fitted: shape {Z[a].shape}, rows sum to {Z[a].sum(axis=1)}")
    
    # Build reward function
    R = {}
    failure_penalty = priors.failure_penalty if priors else -100.0
    action_costs = priors.action_costs if (priors and priors.action_costs) else {}
    
    for a in actions:
        R[a] = np.zeros((n_states, n_states))
        
        # Apply action cost
        action_cost = action_costs.get(a, 0.0)
        
        # Apply failure penalty
        for i, s in enumerate(states):
            for j, s_next in enumerate(states):
                reward = -action_cost
                if "failed" in s_next.lower() or "failure" in s_next.lower():
                    reward += failure_penalty
                elif "healthy" in s_next.lower():
                    reward += 10.0  # Small reward for healthy
                R[a][i, j] = reward
    
    pomdp = POMDP(
        S=states,
        A=actions,
        O=observations,
        T=T,
        Z=Z,
        R=R,
        gamma=0.95,
    )
    
    metadata = {
        "n_episodes": df[episode_col].nunique(),
        "n_samples": len(df),
        "state_counts": df[state_col].value_counts().to_dict(),
        "action_counts": df[action_col].value_counts().to_dict(),
    }
    
    return pomdp, metadata

