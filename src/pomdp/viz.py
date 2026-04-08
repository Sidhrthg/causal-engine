"""
Visualization utilities for POMDP (DOT graph export).
"""

from pathlib import Path
from typing import Optional
import numpy as np

from src.pomdp.schema import POMDP
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def export_dot_transitions(
    pomdp: POMDP,
    out_dir: str,
    top_k: int = 5,
    min_prob: float = 0.01,
) -> None:
    """
    Export transition matrices as DOT graphs (one per action).
    
    Args:
        pomdp: POMDP model
        out_dir: Output directory for .dot files
        top_k: Maximum number of edges to show per state
        min_prob: Minimum probability to include an edge
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for action in pomdp.A:
        T_a = pomdp.T[action]
        filename = out_path / f"transitions_{action}.dot"
        
        with open(filename, "w") as f:
            f.write(f'digraph transitions_{action} {{\n')
            f.write('  rankdir=LR;\n')
            f.write('  node [shape=circle];\n\n')
            
            # For each state, show top-k transitions
            for i, s_from in enumerate(pomdp.S):
                # Get probabilities for this state
                probs = T_a[i, :]
                
                # Get top-k indices
                top_indices = np.argsort(probs)[::-1][:top_k]
                
                for j in top_indices:
                    prob = probs[j]
                    if prob >= min_prob:
                        s_to = pomdp.S[j]
                        # Escape state names for DOT
                        s_from_escaped = s_from.replace('"', '\\"')
                        s_to_escaped = s_to.replace('"', '\\"')
                        f.write(f'  "{s_from_escaped}" -> "{s_to_escaped}" [label="{prob:.3f}"];\n')
            
            f.write('}\n')
        
        logger.info(f"Exported transition graph for action '{action}' to {filename}")


def export_dot_emissions(
    pomdp: POMDP,
    out_dir: str,
    top_k: int = 5,
    min_prob: float = 0.01,
) -> None:
    """
    Export emission matrices as DOT graphs (one per action).
    
    Args:
        pomdp: POMDP model
        out_dir: Output directory for .dot files
        top_k: Maximum number of observations to show per state
        min_prob: Minimum probability to include an edge
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    for action in pomdp.A:
        Z_a = pomdp.Z[action]
        filename = out_path / f"emissions_{action}.dot"
        
        with open(filename, "w") as f:
            f.write(f'digraph emissions_{action} {{\n')
            f.write('  rankdir=LR;\n')
            f.write('  node [shape=circle];\n\n')
            
            # For each state, show top-k observations
            for i, state in enumerate(pomdp.S):
                # Get probabilities for this state
                probs = Z_a[i, :]
                
                # Get top-k indices
                top_indices = np.argsort(probs)[::-1][:top_k]
                
                for j in top_indices:
                    prob = probs[j]
                    if prob >= min_prob:
                        obs = pomdp.O[j]
                        # Escape names for DOT
                        state_escaped = state.replace('"', '\\"')
                        obs_escaped = obs.replace('"', '\\"')
                        # Truncate observation names if too long
                        obs_display = obs_escaped[:20] + "..." if len(obs_escaped) > 20 else obs_escaped
                        f.write(f'  "{state_escaped}" -> "{obs_display}" [label="{prob:.3f}"];\n')
            
            f.write('}\n')
        
        logger.info(f"Exported emission graph for action '{action}' to {filename}")


