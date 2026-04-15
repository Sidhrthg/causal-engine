"""
RAG-powered parameter calibration for the supply-chain causal model.

Usage
-----
    from src.minerals.rag_calibration import calibrate_parameters, STABILITY_BOUNDS
    from src.minerals.rag_pipeline import RAGPipeline

    rag = RAGPipeline()
    params = calibrate_parameters("graphite", rag)
    # params is a dict of validated, stability-checked parameter overrides

Architecture
------------
1. Issue a targeted RAG query for each parameter group.
2. Parse the LLM answer with a regex heuristic (no external NLP dependency).
3. Clamp every extracted value to STABILITY_BOUNDS — the ranges in which the
   system-dynamics model does NOT produce explosive price oscillations.
4. Return only the parameters the LLM was confident about; callers merge these
   into a baseline ParametersConfig.

Stability bounds (derived empirically from the model):
- alpha_P > ~2.0 with eta_D < ~-0.5 creates positive feedback (oscillations).
- tau_K < 1.0 makes capacity respond faster than prices, also destabilising.
- Safe combination: alpha_P ≤ 1.5, eta_D ≥ -0.5, tau_K ≥ 2.0.

References
----------
- Pearl (2009) §7 structural equations
- Sterman (2000) Business Dynamics, Ch. 20 (commodity cycles)
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ── Stability bounds ──────────────────────────────────────────────────────────

STABILITY_BOUNDS: Dict[str, tuple[float, float]] = {
    "eta_D":   (-0.90,  -0.10),   # demand price elasticity: negative, bounded away from 0
    "alpha_P": ( 0.20,   1.50),   # price adjustment speed: max 1.5 for stability
    "tau_K":   ( 1.50,  10.00),   # capacity adjustment lag (years)
    "eta_K":   ( 0.10,   1.00),   # capacity price elasticity
    "demand_reversion_rate": (0.00, 1.00),
}

# ── Parameter extraction helpers ──────────────────────────────────────────────

def _extract_float(text: str, pattern: str) -> Optional[float]:
    """
    Find the first float matching a regex ``pattern`` in ``text``.
    Returns None if nothing matches.
    """
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except (IndexError, ValueError):
            pass
    return None


def _clamp(value: float, lo: float, hi: float, name: str) -> float:
    """Clamp ``value`` to [lo, hi] with a warning if clamped."""
    if value < lo:
        print(f"  [rag_calibration] {name}={value:.3f} below stability floor {lo:.3f}; clamped.")
        return lo
    if value > hi:
        print(f"  [rag_calibration] {name}={value:.3f} above stability ceiling {hi:.3f}; clamped.")
        return hi
    return value


# ── Commodity-specific priors ─────────────────────────────────────────────────

# Fallback priors when RAG is unavailable or returns no useful answer.
# Values are the safe-default stable parameters used in graphite_baseline.yaml.
_DEFAULT_PRIORS: Dict[str, Dict[str, float]] = {
    "graphite": {"eta_D": -0.25, "alpha_P": 0.80, "tau_K": 3.0, "eta_K": 0.40},
    "lithium":  {"eta_D": -0.30, "alpha_P": 0.90, "tau_K": 4.0, "eta_K": 0.50},
    "cobalt":   {"eta_D": -0.20, "alpha_P": 1.00, "tau_K": 5.0, "eta_K": 0.40},
    "nickel":   {"eta_D": -0.25, "alpha_P": 0.70, "tau_K": 4.0, "eta_K": 0.45},
    "copper":   {"eta_D": -0.20, "alpha_P": 0.60, "tau_K": 5.0, "eta_K": 0.40},
}

# ── Query templates ───────────────────────────────────────────────────────────

_QUERY_TEMPLATES = {
    "eta_D": (
        "What is the price elasticity of demand for {commodity}? "
        "Provide a numeric estimate (negative number, e.g. -0.3)."
    ),
    "alpha_P": (
        "How quickly do {commodity} spot prices adjust to supply-demand imbalances? "
        "Express as an annual price-adjustment speed coefficient (positive number, "
        "typical range 0.5–2.0)."
    ),
    "tau_K": (
        "What is the typical lead time in years for new {commodity} mining capacity "
        "to come online from the investment decision? "
        "Provide a numeric estimate (years, e.g. 3–6)."
    ),
    "eta_K": (
        "What is the price elasticity of {commodity} mining capacity investment? "
        "Provide a numeric estimate (positive number, e.g. 0.3–0.7)."
    ),
}

# Patterns to extract the first numeric value from an LLM answer.
_EXTRACT_PATTERNS = {
    "eta_D":   r"(-\s*0\.\d+|-\s*\d+\.\d+)",    # negative float
    "alpha_P": r"(\d+\.?\d*)",                    # positive float
    "tau_K":   r"(\d+\.?\d*)",                    # positive float
    "eta_K":   r"(\d+\.?\d*)",                    # positive float
}


# ── Public API ────────────────────────────────────────────────────────────────

def calibrate_parameters(
    commodity: str,
    rag,
    top_k: int = 6,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Query the RAG pipeline for commodity-specific parameter estimates and
    return a dict of stable, clamped parameter overrides.

    Parameters
    ----------
    commodity:
        Commodity name, e.g. ``"graphite"`` or ``"lithium"``.
    rag:
        An initialised ``RAGPipeline`` instance (or any object with an
        ``.ask(question, top_k)`` method that returns ``{"answer": str}``.
        Pass ``None`` to use commodity priors without RAG.
    top_k:
        Number of RAG chunks to retrieve per parameter query.
    verbose:
        If True, print each query and extracted value.

    Returns
    -------
    dict
        A flat dict of parameter names → floats.  Only parameters the RAG
        (or prior) could estimate are included.  Merge with a baseline
        ``ParametersConfig`` using ``model_copy(update=...)``.

    Examples
    --------
    >>> params = calibrate_parameters("graphite", rag=None)
    >>> params
    {'eta_D': -0.25, 'alpha_P': 0.8, 'tau_K': 3.0, 'eta_K': 0.4}
    """
    commodity = commodity.lower()
    priors = _DEFAULT_PRIORS.get(commodity, _DEFAULT_PRIORS["graphite"])

    if rag is None:
        return dict(priors)

    result: Dict[str, Any] = {}

    for param_name, query_template in _QUERY_TEMPLATES.items():
        query = query_template.format(commodity=commodity)
        try:
            response = rag.ask(query, top_k=top_k)
            answer = response.get("answer", "") if isinstance(response, dict) else str(response)
        except Exception as exc:
            if verbose:
                print(f"  [rag_calibration] RAG failed for {param_name}: {exc}; using prior.")
            result[param_name] = priors[param_name]
            continue

        pattern = _EXTRACT_PATTERNS[param_name]
        value = _extract_float(answer, pattern)

        if value is None:
            if verbose:
                print(f"  [rag_calibration] No numeric value found for {param_name}; using prior {priors[param_name]}.")
            result[param_name] = priors[param_name]
            continue

        lo, hi = STABILITY_BOUNDS[param_name]
        clamped = _clamp(value, lo, hi, param_name)
        result[param_name] = clamped

        if verbose:
            print(f"  [rag_calibration] {param_name}: RAG={value:.3f} → clamped={clamped:.3f}")

    return result


def stable_params_for_commodity(commodity: str) -> Dict[str, float]:
    """Return the default stable parameters for a commodity (no RAG required)."""
    return dict(_DEFAULT_PRIORS.get(commodity.lower(), _DEFAULT_PRIORS["graphite"]))
