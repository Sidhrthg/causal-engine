"""
Central registry for mineral-specific constants and ODE structural defaults.

All values here are single sources of truth — update here and every script
picks up the change automatically.
"""

# ── US net import reliance (USGS Mineral Commodity Summaries 2024) ────────────
# rare_earths net reliance is 14%, but processing dependence on China is ~80%;
# 0.80 is used for supply-security analysis throughout.
US_IMPORT_RELIANCE: dict[str, float] = {
    "graphite":    1.00,
    "rare_earths": 0.80,
    "cobalt":      0.76,
    "lithium":     0.50,
    "nickel":      0.40,
    "uranium":     0.95,
    # Byproduct minerals — added 2026 (China Aug 2023 export controls)
    "germanium":   0.50,   # USGS MCS 2024 (estimated)
    "gallium":     0.87,   # USGS MCS 2024
}

# ── Circumvention rate ────────────────────────────────────────────────────────
# Fraction of a China export restriction that can be rerouted through third
# countries within two years of the ban onset.
# Sources: Section 5.3.2 trade-flow analysis (graphite); literature (others).
CIRCUMVENTION_RATE: dict[str, float] = {
    "graphite":    0.06,
    "rare_earths": 0.10,
    "cobalt":      0.20,
    "lithium":     0.30,
    "nickel":      0.25,
    "uranium":     0.15,
    # Byproduct minerals: very limited circumvention (no fringe at scale)
    "germanium":   0.05,   # Recycling provides ~15-20% globally; routing minimal
    "gallium":     0.05,   # Japan recovery from Al refining is the only credible offset
}

# ── L3 normalisation lag (years at factual restriction end) ───────────────────
# Source: l3_duration_analysis.py benchmark_T results.
# τ_K/2 proxy used where L3 data is unavailable.
NORM_LAG_YRS: dict[str, int] = {
    "graphite":    3,
    "rare_earths": 1,
    "cobalt":      2,
    "lithium":     1,
    "nickel":      3,
    "uranium":     4,
    # Byproduct minerals: tau_K/2 proxy until L3 calibration available.
    # Germanium tau_K ~12 yr (Zn smelter cycle); gallium tau_K ~6 yr (Al refining).
    "germanium":   6,
    "gallium":     3,
}

# ── ODE structural defaults ───────────────────────────────────────────────────
# Fixed across all minerals and episodes.
# Episode-specific values (tau_K, alpha_P, eta_D, g) live in predictability.py.
ODE_DEFAULTS: dict = dict(
    eps=1e-9,
    u0=0.92,
    beta_u=0.10,
    u_min=0.70,
    u_max=1.00,
    eta_K=0.40,
    retire_rate=0.0,
    cover_star=0.20,
    lambda_cover=0.60,
    sigma_P=0.0,
)

# ── Per-mineral substitution and fringe-supply parameters ─────────────────────
# Sources: predictability.py episode calibrations.
# substitution_elasticity: how quickly non-dominant suppliers fill the export gap
# substitution_cap:        max fraction of restricted supply that can be substituted
# fringe_capacity_share:   high-cost entrant capacity as fraction of K0
# fringe_entry_price:      normalised price at which fringe producers first compete
SCENARIO_EXTRAS: dict[str, dict] = {
    "graphite":    dict(substitution_elasticity=0.8, substitution_cap=0.6),
    "rare_earths": dict(substitution_elasticity=0.5, substitution_cap=0.4),
    "cobalt":      dict(substitution_elasticity=0.5, substitution_cap=0.4),
    "lithium":     dict(substitution_elasticity=0.6, substitution_cap=0.5,
                        fringe_capacity_share=0.4,   fringe_entry_price=1.1),
    "nickel":      dict(substitution_elasticity=0.5, substitution_cap=0.4,
                        fringe_capacity_share=0.45,  fringe_entry_price=1.15),
    "uranium":     {},
    # Byproduct minerals (Ge, Ga): no credible fringe at near-current prices.
    # Substitution comes from recycling (Ge: ~15-20%; Ga: Japan Al-refining recovery).
    # Set fringe_capacity_share very low and fringe_entry_price > 5 to effectively
    # disable the fringe mechanism while preserving the substitution channel.
    "germanium":   dict(substitution_elasticity=0.3, substitution_cap=0.2,
                        fringe_capacity_share=0.05,  fringe_entry_price=5.0),
    "gallium":     dict(substitution_elasticity=0.3, substitution_cap=0.2,
                        fringe_capacity_share=0.05,  fringe_entry_price=5.0),
}
