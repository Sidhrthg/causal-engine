#!/usr/bin/env python3
"""
Build POMDP from sensor data and run simulation.
"""

import argparse
import json
import hashlib
from pathlib import Path
import numpy as np

from src.pomdp import (
    load_sensor_csv,
    infer_episodes,
    discretize_observations,
    fit_pomdp,
    Priors,
    rollout,
    policy_threshold,
    export_dot_transitions,
    export_dot_emissions,
)
from src.utils.logging_utils import get_logger
from src.config import Config

logger = get_logger(__name__)


def compute_checksum(matrix: np.ndarray) -> str:
    """Compute MD5 checksum of matrix."""
    return hashlib.md5(matrix.tobytes()).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Build POMDP from sensor data")
    parser.add_argument("--data", type=str, default="data/sensor_test_data.csv",
                       help="Path to sensor data CSV")
    parser.add_argument("--priors", type=str, default=None,
                       help="Path to priors JSON file")
    parser.add_argument("--out-dir", type=str, default="artifacts/pomdp",
                       help="Output directory for artifacts")
    parser.add_argument("--horizon", type=int, default=25,
                       help="Simulation horizon")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Setup
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = load_sensor_csv(args.data)
    
    # Infer episodes
    df = infer_episodes(df, rng=rng)
    
    # Discretize observations
    bin_edges_path = out_dir / "bin_edges.json"
    df, bin_edges = discretize_observations(
        df,
        n_bins=10,
        method="quantile",
        seed=args.seed,
        bin_edges_path=str(bin_edges_path),
    )
    
    # Load priors if provided
    priors = None
    if args.priors:
        logger.info(f"Loading priors from {args.priors}")
        with open(args.priors, "r") as f:
            priors_data = json.load(f)
        
        # Convert priors_data to Priors object
        priors_T_alpha = None
        priors_Z_alpha = None
        if "priors_T_alpha" in priors_data:
            priors_T_alpha = {
                a: np.array(alpha_mat) for a, alpha_mat in priors_data["priors_T_alpha"].items()
            }
        if "priors_Z_alpha" in priors_data:
            priors_Z_alpha = {
                a: np.array(alpha_mat) for a, alpha_mat in priors_data["priors_Z_alpha"].items()
            }
        
        priors = Priors(
            priors_T_alpha=priors_T_alpha,
            priors_Z_alpha=priors_Z_alpha,
            action_costs=priors_data.get("action_costs"),
            failure_penalty=priors_data.get("failure_penalty", -100.0),
            thresholds=priors_data.get("thresholds", {}),
        )
    
    # Fit POMDP
    logger.info("Fitting POMDP")
    pomdp, metadata = fit_pomdp(
        df,
        default_actions=["ignore", "calibrate", "repair"],
        priors=priors,
        smoothing=1.0,
        rng=rng,
    )
    
    # Run simulation
    logger.info("Running simulation")
    start_belief = np.ones(len(pomdp.S)) / len(pomdp.S)  # Uniform initial belief
    
    def policy_fn(belief):
        return policy_threshold(pomdp, belief, threshold=0.5)
    
    results = rollout(
        pomdp,
        policy_fn,
        start_belief,
        horizon=args.horizon,
        rng=rng,
    )
    
    # Export DOT graphs
    dot_dir = out_dir / "graphs"
    logger.info(f"Exporting DOT graphs to {dot_dir}")
    export_dot_transitions(pomdp, str(dot_dir))
    export_dot_emissions(pomdp, str(dot_dir))
    
    # Print summary
    print("\n" + "="*60)
    print("POMDP Summary")
    print("="*60)
    print(f"States (|S|): {len(pomdp.S)}")
    print(f"  {pomdp.S}")
    print(f"Actions (|A|): {len(pomdp.A)}")
    print(f"  {pomdp.A}")
    print(f"Observations (|O|): {len(pomdp.O)}")
    print(f"  (showing first 10) {pomdp.O[:10]}")
    print()
    print("Transition matrices checksums:")
    for a in pomdp.A:
        checksum = compute_checksum(pomdp.T[a])
        print(f"  T[{a}]: {checksum}")
    print("Emission matrices checksums:")
    for a in pomdp.A:
        checksum = compute_checksum(pomdp.Z[a])
        print(f"  Z[{a}]: {checksum}")
    print()
    print(f"Rollout results:")
    print(f"  Total reward: {results['total_reward']:.2f}")
    print(f"  Final belief: {dict(zip(pomdp.S, results['belief_history'][-1]))}")
    print()
    print(f"Artifacts saved to: {out_dir}")
    print("="*60)


if __name__ == "__main__":
    main()


