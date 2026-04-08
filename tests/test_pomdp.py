"""
Tests for POMDP module.
"""

import pytest
import numpy as np
from pathlib import Path

from src.pomdp import (
    POMDP,
    Priors,
    belief_update,
    fit_pomdp,
    export_dot_transitions,
    export_dot_emissions,
)
from src.pomdp.preprocess import load_sensor_csv, infer_episodes, discretize_observations


def test_belief_update_normalized():
    """Test that belief update returns normalized distribution."""
    # Create a simple POMDP
    states = ["healthy", "failed"]
    actions = ["ignore"]
    observations = ["obs1", "obs2"]
    
    # Simple transition: stay in same state
    T = {
        "ignore": np.array([[0.9, 0.1], [0.1, 0.9]])
    }
    
    # Simple emission: obs1 from healthy, obs2 from failed
    Z = {
        "ignore": np.array([[0.8, 0.2], [0.2, 0.8]])
    }
    
    # Simple reward
    R = {
        "ignore": np.array([[10.0, -100.0], [-100.0, 10.0]])
    }
    
    pomdp = POMDP(
        S=states,
        A=actions,
        O=observations,
        T=T,
        Z=Z,
        R=R,
        gamma=0.95,
    )
    
    # Initial belief
    belief = np.array([0.7, 0.3])
    
    # Update belief
    new_belief = belief_update(pomdp, belief, "ignore", "obs1")
    
    # Check normalization
    assert np.isclose(new_belief.sum(), 1.0), "Belief should be normalized"
    assert np.all(new_belief >= 0), "Belief should be non-negative"
    assert len(new_belief) == len(states), "Belief should have correct length"


def test_belief_update_nonnegativity():
    """Test that belief update returns non-negative values."""
    states = ["healthy", "failed"]
    actions = ["ignore"]
    observations = ["obs1"]
    
    T = {"ignore": np.array([[0.5, 0.5], [0.5, 0.5]])}
    Z = {"ignore": np.array([[1.0], [1.0]])}
    R = {"ignore": np.array([[0.0, 0.0], [0.0, 0.0]])}
    
    pomdp = POMDP(
        S=states,
        A=actions,
        O=observations,
        T=T,
        Z=Z,
        R=R,
    )
    
    belief = np.array([0.5, 0.5])
    new_belief = belief_update(pomdp, belief, "ignore", "obs1")
    
    assert np.all(new_belief >= 0), "Belief should be non-negative"
    assert np.all(new_belief <= 1.0 + 1e-6), "Belief probabilities should be <= 1"


def test_fit_produces_row_stochastic():
    """Test that fit_pomdp produces row-stochastic matrices."""
    import pandas as pd
    
    # Create synthetic data
    n_samples = 100
    data = {
        "episode_id": [0] * 50 + [1] * 50,
        "state": ["healthy"] * 30 + ["failed"] * 20 + ["healthy"] * 40 + ["failed"] * 10,
        "action": ["ignore"] * 50 + ["repair"] * 50,
        "observation": [f"obs_{i % 5}" for i in range(100)],
    }
    df = pd.DataFrame(data)
    
    # Fit POMDP
    pomdp, metadata = fit_pomdp(df, state_col="state", action_col="action")
    
    # Check that T matrices are row-stochastic
    for action in pomdp.A:
        T_a = pomdp.T[action]
        row_sums = T_a.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"T[{action}] should be row-stochastic"
        assert np.all(T_a >= 0), f"T[{action}] should be non-negative"
    
    # Check that Z matrices are row-stochastic
    for action in pomdp.A:
        Z_a = pomdp.Z[action]
        row_sums = Z_a.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), f"Z[{action}] should be row-stochastic"
        assert np.all(Z_a >= 0), f"Z[{action}] should be non-negative"


def test_dot_export_creates_files():
    """Test that dot export creates files containing 'digraph'."""
    import tempfile
    
    states = ["healthy", "failed"]
    actions = ["ignore", "repair"]
    observations = ["obs1", "obs2"]
    
    T = {
        "ignore": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "repair": np.array([[0.5, 0.5], [0.8, 0.2]]),
    }
    Z = {
        "ignore": np.array([[0.8, 0.2], [0.2, 0.8]]),
        "repair": np.array([[0.9, 0.1], [0.3, 0.7]]),
    }
    R = {
        "ignore": np.array([[10.0, -100.0], [-100.0, 10.0]]),
        "repair": np.array([[5.0, -50.0], [-50.0, 5.0]]),
    }
    
    pomdp = POMDP(
        S=states,
        A=actions,
        O=observations,
        T=T,
        Z=Z,
        R=R,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Export transitions
        export_dot_transitions(pomdp, tmpdir)
        
        # Check that files were created
        transition_files = list(Path(tmpdir).glob("transitions_*.dot"))
        assert len(transition_files) > 0, "Should create transition dot files"
        
        # Check that files contain "digraph"
        for fpath in transition_files:
            content = fpath.read_text()
            assert "digraph" in content, f"File {fpath} should contain 'digraph'"
        
        # Export emissions
        export_dot_emissions(pomdp, tmpdir)
        
        # Check that files were created
        emission_files = list(Path(tmpdir).glob("emissions_*.dot"))
        assert len(emission_files) > 0, "Should create emission dot files"
        
        # Check that files contain "digraph"
        for fpath in emission_files:
            content = fpath.read_text()
            assert "digraph" in content, f"File {fpath} should contain 'digraph'"


def test_fit_with_priors():
    """Test that fit_pomdp accepts and uses priors."""
    import pandas as pd
    
    # Create synthetic data
    data = {
        "episode_id": [0] * 20 + [1] * 20,
        "state": ["healthy"] * 25 + ["failed"] * 15,
        "action": ["ignore"] * 40,
        "observation": [f"obs_{i % 3}" for i in range(40)],
    }
    df = pd.DataFrame(data)
    
    # Create priors
    priors_T_alpha = {
        "ignore": np.array([[2.0, 1.0], [1.0, 2.0]])
    }
    priors = Priors(priors_T_alpha=priors_T_alpha)
    
    # Fit POMDP with priors
    pomdp, metadata = fit_pomdp(
        df,
        state_col="state",
        action_col="action",
        priors=priors,
    )
    
    # Check that POMDP was created successfully
    assert len(pomdp.S) > 0
    assert len(pomdp.A) > 0
    assert len(pomdp.O) > 0
    
    # Check that T matrices are row-stochastic
    for action in pomdp.A:
        T_a = pomdp.T[action]
        assert np.allclose(T_a.sum(axis=1), 1.0, atol=1e-6)


