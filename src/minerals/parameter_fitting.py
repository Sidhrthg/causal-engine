"""
Empirical parameter fitting from CEPII / Comtrade trade data.

Fits the three causally-identified model parameters:
  eta_D   — demand price elasticity       (IV / 2SLS)
  alpha_P — price adjustment speed        (log-linear OLS on tightness proxy)
  tau_K   — capacity adjustment time      (synthetic control / AR fit)

Each estimator corresponds to the identification strategy declared in
causal_inference.CommoditySupplyChainDAG.get_parameter_identifications().

Works with any commodity for which you have CEPII-format bilateral trade data
(columns: year, exporter, importer, product, value_kusd, quantity_tonnes).

Usage
-----
    from src.minerals.parameter_fitting import fit_commodity_parameters

    # Graphite (uses bundled canonical data by default)
    params = fit_commodity_parameters("graphite")
    print(params.summary())

    # Lithium with a custom path
    params = fit_commodity_parameters(
        "lithium",
        data_path="data/canonical/cepii_lithium.csv",
        dominant_exporter="Australia",
    )

    # Backwards-compatible alias:
    from src.minerals.parameter_fitting import fit_graphite_parameters
    params = fit_graphite_parameters()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.minerals.causal_identification import InstrumentalVariable, IVResult
from src.minerals.country_codes import normalize_country_names

logger = logging.getLogger(__name__)

_CANONICAL_DATA_DIR = Path(__file__).parents[2] / "data" / "canonical"
_DEFAULT_CEPII = _CANONICAL_DATA_DIR / "cepii_graphite.csv"

_COMMODITY_DEFAULTS: Dict[str, Dict] = {
    "graphite": {
        "data_path": str(_CANONICAL_DATA_DIR / "cepii_graphite.csv"),
        "dominant_exporter": "China",
    },
    "lithium": {
        "data_path": str(_CANONICAL_DATA_DIR / "cepii_lithium.csv"),
        "dominant_exporter": "Australia",
    },
    "cobalt": {
        "data_path": str(_CANONICAL_DATA_DIR / "cepii_cobalt.csv"),
        "dominant_exporter": "DRC",
    },
    "nickel": {
        "data_path": str(_CANONICAL_DATA_DIR / "cepii_nickel.csv"),
        "dominant_exporter": "Indonesia",
    },
}


# ---------------------------------------------------------------------------
# Fitted parameter container
# ---------------------------------------------------------------------------

@dataclass
class FittedParameters:
    """
    Causally-identified model parameters estimated from real trade data.

    All parameters correspond to those used in model.py / schema.py.
    Confidence intervals are 95% (1.96 * SE from the estimator).
    """
    eta_D: float          # demand price elasticity (negative; log-log)
    eta_D_ci: Tuple[float, float]
    eta_D_se: float
    eta_D_first_stage_F: float   # IV weak-instrument diagnostic (want > 10)

    alpha_P: float        # price adjustment speed (log ΔP / tightness)
    alpha_P_ci: Tuple[float, float]
    alpha_P_se: float

    tau_K: float          # capacity mean-reversion half-life (years)
    tau_K_ci: Tuple[float, float]
    tau_K_se: float

    n_obs: int
    year_range: Tuple[int, int]
    notes: Dict[str, str] = field(default_factory=dict)
    commodity: str = "graphite"
    dominant_exporter: str = "China"

    def as_dict(self) -> Dict[str, float]:
        """Return point estimates in the format expected by ParametersConfig."""
        return {
            "eta_D": self.eta_D,
            "alpha_P": self.alpha_P,
            "tau_K": self.tau_K,
        }

    def summary(self) -> str:
        lines = [
            f"=== Fitted Parameters ({self.commodity}, {self.dominant_exporter}, empirical) ===",
            f"  eta_D  (demand elasticity):   {self.eta_D:+.3f}  "
            f"95% CI [{self.eta_D_ci[0]:+.3f}, {self.eta_D_ci[1]:+.3f}]  "
            f"First-stage F={self.eta_D_first_stage_F:.1f}"
            + (" ⚠ weak instrument" if self.eta_D_first_stage_F < 10 else ""),
            f"  alpha_P (price adj speed):    {self.alpha_P:+.3f}  "
            f"95% CI [{self.alpha_P_ci[0]:+.3f}, {self.alpha_P_ci[1]:+.3f}]",
            f"  tau_K   (capacity half-life): {self.tau_K:.2f} years  "
            f"95% CI [{self.tau_K_ci[0]:.2f}, {self.tau_K_ci[1]:.2f}]",
            f"  N={self.n_obs} annual obs, {self.year_range[0]}–{self.year_range[1]}",
        ]
        for k, v in self.notes.items():
            lines.append(f"  note[{k}]: {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data construction
# ---------------------------------------------------------------------------

def _build_panel(df: pd.DataFrame, dominant_exporter: str = "China") -> pd.DataFrame:
    """
    Aggregate CEPII bilateral flows to annual world-level panel.

    Works for any commodity: pass ``dominant_exporter`` to identify the
    primary supply-side actor used as an IV instrument.

    Derived columns
    ---------------
    implied_price_usd_t  — world average unit value (USD/tonne)
    world_qty_t          — total world trade quantity (tonnes)
    dom_supply_t         — dominant_exporter total export quantity (tonnes)
    dom_share            — dominant_exporter share of world exports (0-1)
    log_price            — log of implied price
    log_demand           — log of world quantity
    log_dom_supply       — log of dominant_exporter supply (instrument for IV)
    dlog_price           — first difference of log price
    dlog_demand          — first difference of log demand
    tightness            — (demand - supply proxy) / demand proxy (lagged)

    Note: legacy column names (china_supply_t, china_share, log_china_supply,
    log_china_share) are kept as aliases for backwards compatibility.
    """
    df = normalize_country_names(df)
    dom = (
        df[df["exporter"] == dominant_exporter]
        .groupby("year")["quantity_tonnes"]
        .sum()
        .reset_index(name="dom_supply_t")
    )
    world = df.groupby("year").agg(
        world_trade_kusd=("value_kusd", "sum"),
        world_qty_t=("quantity_tonnes", "sum"),
    ).reset_index()

    panel = world.merge(dom, on="year", how="left")
    panel["dom_supply_t"] = panel["dom_supply_t"].fillna(0.0)

    panel["implied_price_usd_t"] = (
        panel["world_trade_kusd"] * 1000.0
        / panel["world_qty_t"].replace(0, np.nan)
    )
    panel["dom_share"] = panel["dom_supply_t"] / panel["world_qty_t"].replace(0, np.nan)

    panel["log_price"] = np.log(panel["implied_price_usd_t"].replace(0, np.nan))
    panel["log_demand"] = np.log(panel["world_qty_t"].replace(0, np.nan))
    panel["log_dom_supply"] = np.log(panel["dom_supply_t"].replace(0, np.nan))
    panel["log_dom_share"] = np.log(panel["dom_share"].replace(0, np.nan))

    # Legacy aliases (graphite code used "china_*" names)
    panel["china_supply_t"] = panel["dom_supply_t"]
    panel["china_share"] = panel["dom_share"]
    panel["log_china_supply"] = panel["log_dom_supply"]
    panel["log_china_share"] = panel["log_dom_share"]

    panel = panel.sort_values("year").reset_index(drop=True)

    panel["dlog_price"] = panel["log_price"].diff()
    panel["dlog_demand"] = panel["log_demand"].diff()
    panel["dlog_dom_supply"] = panel["log_dom_supply"].diff()
    panel["dlog_china_supply"] = panel["dlog_dom_supply"]  # alias
    panel["year_trend"] = panel["year"] - panel["year"].min()

    # Instrument for eta_D: 2-year lag of dominant_exporter supply changes.
    #
    # The contemporaneous dominant_exporter supply change (dlog_dom_supply) is
    # endogenous: large producers both supply and demand the commodity, so supply
    # and demand shocks are correlated.  A 2-year lag breaks the contemporaneous
    # correlation while preserving the supply-side identifying variation (mine
    # capacity decisions made 2 years prior are driven by geology and capital
    # cycles, not current world demand).
    panel["iv_lag2_dom_supply"] = panel["dlog_dom_supply"].shift(2)

    # Tightness proxy for alpha_P: use log-price deviation from HP-filtered trend.
    # This avoids the endogeneity in the raw supply-shortfall proxy
    # (lagged demand - lagged supply conflates supply and demand shocks).
    try:
        from statsmodels.tsa.filters.hp_filter import hpfilter
        log_p = panel["log_price"].ffill().bfill()
        _, log_p_trend = hpfilter(log_p.values, lamb=100)
        panel["price_gap"] = log_p.values - log_p_trend   # deviation from trend
    except Exception:
        # Fallback: simple first-difference as gap proxy
        panel["price_gap"] = panel["dlog_price"].fillna(0.0)

    # Original tightness proxy retained for backwards compat
    panel["tightness_proxy"] = (
        panel["log_demand"].shift(1) - panel["log_dom_supply"].shift(1)
    )

    return panel.dropna(subset=["log_price", "log_demand", "log_dom_supply"])


# ---------------------------------------------------------------------------
# eta_D — OLS (log-log levels) with IV attempted as diagnostic
# ---------------------------------------------------------------------------

def _fit_eta_D(panel: pd.DataFrame) -> Tuple[float, Tuple[float, float], float, float]:
    """
    Estimate demand price elasticity (eta_D) via 2SLS IV regression.

    Strategy: first-differences with 2-year lagged dominant_exporter supply
    changes as the instrument for price changes.

    First stage:  dlog(price_t) = a + b * dlog(dom_supply_{t-2}) + ε
    Second stage: dlog(demand_t) = a + eta_D * dlog_price_hat_t + ε

    The 2-year lag breaks contemporaneous endogeneity: supply decisions made
    2 years prior (mine permitting, capacity investments) are driven by geology
    and capital cycles, not current world demand fluctuations.

    Fallback: if the first-stage F < 5 (instrument still weak), falls back to
    OLS in log-log levels with year trend, which is biased toward zero.

    Returns (eta_D, ci_95, se, first_stage_F).
    """
    sub = panel.dropna(subset=["dlog_demand", "dlog_price", "iv_lag2_dom_supply"]).copy()
    n = len(sub)

    y = sub["dlog_demand"].values.astype(float)
    P = sub["dlog_price"].values.astype(float)
    Z = sub["iv_lag2_dom_supply"].values.astype(float)

    # --- First stage: P ~ Z ---
    X1 = np.column_stack([np.ones(n), Z])
    b1, _, _, _ = np.linalg.lstsq(X1, P, rcond=None)
    P_hat = X1 @ b1
    resid1 = P - P_hat
    ss_full = np.sum(resid1 ** 2)
    ss_restr = np.sum((P - np.mean(P)) ** 2)
    first_stage_F = float(((ss_restr - ss_full) / 1.0) / (ss_full / max(n - 2, 1)))

    if first_stage_F >= 5.0:
        # 2SLS second stage: demand ~ P_hat
        X2 = np.column_stack([np.ones(n), P_hat])
        b2, _, _, _ = np.linalg.lstsq(X2, y, rcond=None)
        eta_D = float(b2[1])
        estimator = "2SLS (lag-2 instrument)"
    else:
        # Weak instrument fallback: OLS in log-log levels with year trend
        logger.warning(
            f"Lagged instrument weak (F={first_stage_F:.1f} < 5); "
            "falling back to OLS log-log levels. Estimate biased toward zero."
        )
        sub2 = panel.dropna(subset=["log_demand", "log_price", "year_trend"])
        n2 = len(sub2)
        y2 = sub2["log_demand"].values.astype(float)
        P2 = sub2["log_price"].values.astype(float)
        trend = sub2["year_trend"].values.astype(float)
        X_ols = np.column_stack([np.ones(n2), P2, trend])
        b_ols, _, _, _ = np.linalg.lstsq(X_ols, y2, rcond=None)
        eta_D = float(b_ols[1])
        n, y, P_hat = n2, y2, P2
        estimator = "OLS (IV weak, fallback)"

    # Enforce sign: demand elasticity must be <= 0
    # Positive estimate implies demand is rising with price (supply-driven shock
    # in a growing market). Apply a soft floor at literature prior.
    if eta_D > 0:
        logger.warning(
            f"eta_D = {eta_D:.3f} > 0 (wrong sign). "
            "Likely supply-demand identification failure. "
            "Clamping to literature prior -0.15."
        )
        eta_D = -0.15

    # --- Bootstrap CI (circular block bootstrap, block = 3) ---
    rng = np.random.default_rng(42)
    T = len(y)
    block = max(2, int(np.ceil(T ** (1.0 / 3.0))))
    boot_vals = []
    for _ in range(500):
        n_blocks = int(np.ceil(T / block))
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([
            np.arange(s, s + block) % T for s in starts
        ])[:T]
        y_b = y[idx]
        P_b = P_hat[idx]
        X_b = np.column_stack([np.ones(T), P_b])
        try:
            b, _, _, _ = np.linalg.lstsq(X_b, y_b, rcond=None)
            boot_vals.append(float(b[1]))
        except Exception:
            pass

    boot_arr = np.array(boot_vals)
    se = float(boot_arr.std(ddof=1)) if len(boot_arr) > 1 else 0.15
    ci = (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)))

    logger.info(f"eta_D estimator: {estimator}, F={first_stage_F:.1f}")
    return eta_D, ci, se, first_stage_F


# ---------------------------------------------------------------------------
# alpha_P — price adjustment speed (OLS on log ΔP ~ tightness)
# ---------------------------------------------------------------------------

def _fit_alpha_P(panel: pd.DataFrame) -> Tuple[float, Tuple[float, float], float]:
    """
    Estimate price adjustment speed (alpha_P) from log-price dynamics.

    From the structural price equation:
        Δlog(P_t) ≈ alpha_P * price_gap_{t-1} + noise
    where price_gap = log(P) - log(P_trend) is the deviation of the log-price
    from its HP-filtered trend (lamb=100, annual data).

    This replaces the old supply-shortfall tightness proxy which was endogenous
    (lagged demand - lagged dom_supply conflates supply and demand shocks).
    The HP cycle component is a cleaner measure of price over/under-shooting.

    Falls back to the supply-shortfall proxy if the price_gap column is absent
    or the HP-filter failed.

    Returns (alpha_P, ci_95, se).
    """
    # Prefer HP-filtered price gap; fall back to tightness_proxy
    if "price_gap" in panel.columns and panel["price_gap"].notna().sum() >= 8:
        gap_col = "price_gap"
        # Regress dlog_price on lagged price_gap (mean-reversion)
        sub = panel.copy()
        sub["lagged_gap"] = sub[gap_col].shift(1)
        sub = sub.dropna(subset=["dlog_price", "lagged_gap"])
        x_col = "lagged_gap"
    else:
        sub = panel.dropna(subset=["dlog_price", "tightness_proxy"])
        x_col = "tightness_proxy"

    n = len(sub)
    if n < 5:
        logger.warning("alpha_P: too few observations; using fallback 0.30")
        return 0.30, (0.10, 0.60), 0.12

    y = sub["dlog_price"].values.astype(float)
    X = np.column_stack([np.ones(n), sub[x_col].values.astype(float)])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha_P = float(beta[1])

    resid = y - X @ beta
    sigma2 = np.sum(resid ** 2) / max(n - 2, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    t_crit = float(stats.t.ppf(0.975, df=max(n - 2, 1)))
    ci = (alpha_P - t_crit * se, alpha_P + t_crit * se)

    # alpha_P must be positive (prices mean-revert upward when below trend)
    if alpha_P <= 0:
        logger.warning(
            f"alpha_P estimate {alpha_P:.4f} is non-positive "
            f"(regressor: {x_col}). "
            "Possible cause: demand growth dominates price signal. "
            "Applying literature prior 0.30."
        )
        alpha_P = 0.30
        ci = (max(0.05, ci[0]), max(0.60, ci[1]))

    return alpha_P, ci, se


# ---------------------------------------------------------------------------
# tau_K — capacity adjustment half-life (AR(1) on log supply)
# ---------------------------------------------------------------------------

def _fit_tau_K(panel: pd.DataFrame) -> Tuple[float, Tuple[float, float], float]:
    """
    Estimate capacity mean-reversion speed from China supply dynamics.

    From model.py's capacity equation (dt=1):
        K_{t+1} = K_t + (K*_t - K_t)/tau_K - retire * K_t
    In log-form near the trend: Δlog(K_t) ≈ -1/tau_K * (log(K_t) - log(K*))
    This is an AR(1) with coefficient ρ = 1 - 1/tau_K.

    We estimate ρ from AR(1) on log(china_supply_t):
        log(S_t) = c + ρ * log(S_{t-1}) + ε_t
        tau_K = 1 / (1 - ρ)

    Bootstrap SE on tau_K = 1/(1-ρ) via delta method.
    Returns (tau_K_years, ci, se).
    """
    sub = panel.dropna(subset=["log_dom_supply"])
    sub = sub.sort_values("year")

    y = sub["log_dom_supply"].values[1:].astype(float)
    X = np.column_stack([
        np.ones(len(y)),
        sub["log_dom_supply"].values[:-1].astype(float),
    ])
    n = len(y)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rho = float(beta[1])

    resid = y - X @ beta
    sigma2 = np.sum(resid ** 2) / max(n - 2, 1)
    xtx_inv = np.linalg.pinv(X.T @ X)
    se_rho = float(np.sqrt(sigma2 * xtx_inv[1, 1]))

    # rho must be < 1 for stationarity; if >= 1 cap at 0.95 (tau_K = 20 yr)
    rho = min(rho, 0.95)
    rho = max(rho, 0.0)  # disallow explosive process

    tau_K = 1.0 / (1.0 - rho)

    # Delta method: d(tau_K)/d(rho) = 1/(1-rho)^2
    se_tau = se_rho / (1.0 - rho) ** 2

    t_crit = float(stats.t.ppf(0.975, df=max(n - 2, 1)))
    ci = (max(1.0, tau_K - t_crit * se_tau), tau_K + t_crit * se_tau)

    return tau_K, ci, se_tau


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def fit_commodity_parameters(
    commodity: str = "graphite",
    data_path: Optional[str] = None,
    dominant_exporter: Optional[str] = None,
) -> FittedParameters:
    """
    Fit eta_D, alpha_P, tau_K from CEPII-format bilateral trade data.

    Works for any commodity; defaults are pre-configured for graphite,
    lithium, cobalt, and nickel (see ``_COMMODITY_DEFAULTS``).

    Args:
        commodity: Commodity name (e.g. "graphite", "lithium", "cobalt").
            Used to look up default data_path and dominant_exporter.
        data_path: Override path to CEPII CSV. Required if commodity is not
            in ``_COMMODITY_DEFAULTS`` and has no bundled data.
        dominant_exporter: Override the dominant exporter (instrument for IV).
            Defaults to the value in ``_COMMODITY_DEFAULTS`` or "China".

    Returns:
        FittedParameters with point estimates, CIs, and diagnostics.
    """
    defaults = _COMMODITY_DEFAULTS.get(commodity, {})
    resolved_path = Path(data_path) if data_path else Path(defaults.get("data_path", ""))
    resolved_exporter = dominant_exporter or defaults.get("dominant_exporter", "China")

    if not resolved_path or not resolved_path.exists():
        raise FileNotFoundError(
            f"Trade data not found at '{resolved_path}' for commodity '{commodity}'. "
            "Supply data_path= explicitly or add an entry to _COMMODITY_DEFAULTS."
        )

    logger.info(f"Fitting parameters for {commodity} (dominant exporter: {resolved_exporter})")
    logger.info(f"Loading data from {resolved_path}")
    df = pd.read_csv(resolved_path)
    panel = _build_panel(df, dominant_exporter=resolved_exporter)
    logger.info(
        f"Panel: {len(panel)} annual observations, "
        f"{panel['year'].min()}–{panel['year'].max()}"
    )

    notes: dict = {}

    # --- eta_D ---
    try:
        eta_D, eta_D_ci, eta_D_se, eta_D_F = _fit_eta_D(panel)
        if eta_D_F < 10:
            notes["eta_D"] = (
                f"Weak instrument (F={eta_D_F:.1f} < 10). "
                "Estimate may be biased. Consider additional instruments."
            )
        logger.info(f"eta_D = {eta_D:.3f} (F={eta_D_F:.1f})")
    except Exception as e:
        logger.warning(f"eta_D fitting failed: {e}; using fallback -0.3")
        eta_D, eta_D_ci, eta_D_se, eta_D_F = -0.3, (-0.6, 0.0), 0.15, 0.0
        notes["eta_D"] = f"Fitting failed ({e}); using literature fallback -0.3"

    # --- alpha_P ---
    try:
        alpha_P, alpha_P_ci, alpha_P_se = _fit_alpha_P(panel)
        logger.info(f"alpha_P = {alpha_P:.3f}")
    except Exception as e:
        logger.warning(f"alpha_P fitting failed: {e}; using fallback 0.5")
        alpha_P, alpha_P_ci, alpha_P_se = 0.5, (0.2, 0.8), 0.15
        notes["alpha_P"] = f"Fitting failed ({e}); using literature fallback 0.5"

    # --- tau_K ---
    try:
        tau_K, tau_K_ci, tau_K_se = _fit_tau_K(panel)
        logger.info(f"tau_K = {tau_K:.2f} years")
    except Exception as e:
        logger.warning(f"tau_K fitting failed: {e}; using fallback 5.0")
        tau_K, tau_K_ci, tau_K_se = 5.0, (3.0, 8.0), 1.0
        notes["tau_K"] = f"Fitting failed ({e}); using literature fallback 5.0"

    return FittedParameters(
        eta_D=eta_D,
        eta_D_ci=eta_D_ci,
        eta_D_se=eta_D_se,
        eta_D_first_stage_F=eta_D_F,
        alpha_P=alpha_P,
        alpha_P_ci=alpha_P_ci,
        alpha_P_se=alpha_P_se,
        tau_K=tau_K,
        tau_K_ci=tau_K_ci,
        tau_K_se=tau_K_se,
        n_obs=len(panel),
        year_range=(int(panel["year"].min()), int(panel["year"].max())),
        notes=notes,
        commodity=commodity,
        dominant_exporter=resolved_exporter,
    )


def fit_graphite_parameters(
    cepii_path: Optional[str] = None,
) -> FittedParameters:
    """Backwards-compatible alias for fit_commodity_parameters('graphite')."""
    return fit_commodity_parameters(
        commodity="graphite",
        data_path=cepii_path,
        dominant_exporter="China",
    )
