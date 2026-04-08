"""
System dynamics simulation layer for critical minerals supply chain.

Uses causally-identified parameters from causal_inference.py and causal_identification.py.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd

from .schema import ScenarioConfig
from .simulate import run_scenario


class SystemDynamicsModel:
    """
    System Dynamics Model with Causally-Identified Parameters

    This model simulates P(Y|do(X)) given an assumed causal structure.
    Parameters are causally identified using methods from causal_inference.py:

    Parameters and their identification strategies:
    - eta_D (demand elasticity): Instrumental Variables (supply shocks as instrument)
    - tau_K (capacity adjustment): Synthetic Control (see causal_identification.py)
    - alpha_P (price adjustment): Regression Discontinuity (policy thresholds)
    - policy_shock: Difference-in-Differences (quota implementations)

    For formal identifiability proofs, see:
    src/minerals/causal_inference.py - GraphiteSupplyChainDAG

    Note: This model performs simulation given identified parameters.
    It does NOT perform causal identification itself.
    Identification happens in causal_identification.py and causal_inference.py.
    """

    def run(self, cfg: ScenarioConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Run scenario simulation; delegates to run_scenario."""
        return run_scenario(cfg)
