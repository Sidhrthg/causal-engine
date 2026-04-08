from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import ShockConfig


def _get_shock_years(shocks: List[ShockConfig]) -> set[int]:
    """Get set of all years where any shock is active."""
    shock_years = set()
    for shock in shocks:
        for year in range(shock.start_year, shock.end_year + 1):
            shock_years.add(year)
    return shock_years


def _get_last_shock_year(shocks: List[ShockConfig]) -> Optional[int]:
    """Get the last year where any shock is active, or None if no shocks."""
    if not shocks:
        return None
    return max(shock.end_year for shock in shocks)


def compute_metrics(df: pd.DataFrame, shocks: Optional[List[ShockConfig]] = None) -> Dict[str, float]:
    # Expect columns: year, P, K, I, Q_eff, D, shortage, cover
    total_shortage = float(df["shortage"].sum())
    peak_shortage = float(df["shortage"].max())
    avg_price = float(df["P"].mean())
    final_inventory_cover = float(df["cover"].iloc[-1])
    
    metrics = {
        "total_shortage": total_shortage,
        "peak_shortage": peak_shortage,
        "avg_price": avg_price,
        "final_inventory_cover": final_inventory_cover,
    }
    
    # Add shock-window metrics if shocks are provided
    if shocks:
        shock_years = _get_shock_years(shocks)
        last_shock_year = _get_last_shock_year(shocks)
        
        # shock_window_total_shortage: sum over years where any shock is active
        if shock_years:
            shock_window_mask = df["year"].isin(shock_years)
            shock_window_total_shortage = float(df.loc[shock_window_mask, "shortage"].sum())
            metrics["shock_window_total_shortage"] = shock_window_total_shortage
            
            # shock_year_shortage: shortage in the first shock year
            first_shock_year = min(shock_years)
            first_shock_row = df[df["year"] == first_shock_year]
            if len(first_shock_row) > 0:
                shock_year_shortage = float(first_shock_row["shortage"].iloc[0])
                metrics["shock_year_shortage"] = shock_year_shortage
        
        # post_shock_total_shortage: sum over years strictly after last shock year
        if last_shock_year is not None:
            post_shock_mask = df["year"] > last_shock_year
            if post_shock_mask.any():
                post_shock_total_shortage = float(df.loc[post_shock_mask, "shortage"].sum())
                metrics["post_shock_total_shortage"] = post_shock_total_shortage
            else:
                metrics["post_shock_total_shortage"] = 0.0
    
    return metrics
