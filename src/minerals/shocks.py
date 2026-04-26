from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from .schema import ShockConfig

if TYPE_CHECKING:
    from .knowledge_graph import CausalKnowledgeGraph


@dataclass(frozen=True)
class ShockSignals:
    export_restriction: float = 0.0  # fraction of supply blocked
    demand_surge: float = 0.0  # fractional demand boost
    capex_shock: float = 0.0  # fractional reduction in investment target
    stockpile_release: float = 0.0  # one-time inventory delta (tons)
    # Multipliers for new shock types (1.0 = no effect)
    policy_supply_mult: float = 1.0  # policy_shock: quota reduction
    capacity_supply_mult: float = 1.0  # capacity_reduction: supply cut
    demand_destruction_mult: float = 1.0  # macro_demand_shock: demand drop (e.g. 0.6 = 40% drop)


def _active(shock: ShockConfig, year: int) -> bool:
    return shock.start_year <= year <= shock.end_year


def apply_shocks(t: float, shocks: List[ShockConfig]) -> Dict[str, float]:
    """
    Apply shock multipliers for current time.
    Returns dict with shock impacts (multipliers; 1.0 = no effect).

    Keys:
        export_restriction  — 1.0 - cumulative magnitude (multiplicative; 1.0 = no restriction)
        demand_shock        — cumulative demand multiplier (macro_demand_shock only)
        policy_shock        — cumulative supply quota multiplier
        capacity_shock      — cumulative capacity multiplier
    Note: shocks_for_year() uses export_restriction additively for the ShockSignals
    struct; this key is kept here for direct callers / tests of apply_shocks.
    """
    shock_impacts = {
        "export_restriction": 1.0,
        "demand_shock": 1.0,
        "policy_shock": 1.0,
        "capacity_shock": 1.0,
    }

    for shock in shocks:
        if shock.start_year <= t <= shock.end_year:
            if shock.type == "export_restriction":
                shock_impacts["export_restriction"] *= 1.0 - shock.magnitude
            elif shock.type == "policy_shock":
                quota_reduction = shock.quota_reduction if shock.quota_reduction is not None else shock.magnitude
                shock_impacts["policy_shock"] *= 1.0 - quota_reduction
            elif shock.type == "macro_demand_shock":
                demand_destruction = shock.demand_destruction if shock.demand_destruction is not None else shock.magnitude
                shock_impacts["demand_shock"] *= 1.0 + demand_destruction
            elif shock.type == "capacity_reduction":
                shock_impacts["capacity_shock"] *= 1.0 - shock.magnitude
            elif shock.type == "capex_shock":
                shock_impacts["capacity_shock"] *= 1.0 - shock.magnitude

    return shock_impacts


def shocks_for_year(
    shocks: List[ShockConfig],
    year: int,
    kg: Optional["CausalKnowledgeGraph"] = None,
    commodity: str = "",
) -> ShockSignals:
    """
    Build ShockSignals for *year*.

    When *kg* and *commodity* are provided, any ShockConfig with a ``country``
    field set has its ``export_restriction`` magnitude scaled by
    ``kg.effective_control_at(country, commodity, year)['effective_share']``.
    This converts "fraction of country's exports restricted" into the
    equivalent global-supply fraction that the ODE model consumes.

    Callers that do not pass *kg* get the original, unscaled behaviour.
    """
    # Build multipliers from apply_shocks
    impacts = apply_shocks(float(year), shocks)

    # Legacy additive signals (for export_restriction, demand_surge, capex_shock, stockpile_release)
    export_restriction = 0.0
    demand_surge = 0.0
    capex_shock = 0.0
    stockpile_release = 0.0

    for s in shocks:
        if not _active(s, year):
            continue
        if s.type == "export_restriction":
            mag = s.magnitude
            if s.country and kg is not None and commodity:
                eff = kg.effective_control_at(s.country, commodity, year)
                if eff["effective_share"] is not None:
                    mag = s.magnitude * eff["effective_share"]
            export_restriction += mag
        elif s.type == "demand_surge":
            demand_surge += s.magnitude
        elif s.type == "capex_shock":
            capex_shock += s.magnitude
        elif s.type == "stockpile_release":
            stockpile_release += s.magnitude

    export_restriction = min(max(export_restriction, 0.0), 0.95)
    capex_shock = min(max(capex_shock, 0.0), 0.95)
    # demand_surge: clamped symmetric to [-0.95, 2.0] — allows up to 3x demand
    # without an upper cap the multiplier (1+surge) can become unphysically large
    demand_surge = min(max(demand_surge, -0.95), 2.0)
    stockpile_release = max(stockpile_release, -1000.0)

    return ShockSignals(
        export_restriction=export_restriction,
        demand_surge=demand_surge,
        capex_shock=capex_shock,
        stockpile_release=stockpile_release,
        policy_supply_mult=impacts["policy_shock"],
        capacity_supply_mult=impacts["capacity_shock"],
        demand_destruction_mult=impacts["demand_shock"],
    )
