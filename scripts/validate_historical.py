#!/usr/bin/env python3
"""Validate model simulations against historical Comtrade trade data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_simulation_data(run_dir: Path, year: int) -> Dict[str, float]:
    """Load simulation data for a specific year."""
    ts_path = run_dir / "timeseries.csv"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timeseries.csv in {run_dir}")
    
    df = pd.read_csv(ts_path)
    if "year" not in df.columns:
        raise ValueError("timeseries.csv must have 'year' column")
    
    year_data = df[df["year"] == year]
    if len(year_data) == 0:
        raise ValueError(f"No data found for year {year} in simulation")
    
    row = year_data.iloc[0]
    return {
        "year": int(year),
        "P": float(row.get("P", 0.0)),  # Price
        "Q": float(row.get("Q", 0.0)),  # Production/Supply
        "D": float(row.get("D", 0.0)),  # Demand
        "I": float(row.get("I", 0.0)),  # Inventory
        "shortage": float(row.get("shortage", 0.0)),
        "cover": float(row.get("cover", 0.0)),
    }


def load_comtrade_data(comtrade_path: Path, year: int) -> Optional[Dict[str, float]]:
    """Load Comtrade data for a specific year."""
    if not comtrade_path.exists():
        raise FileNotFoundError(f"Comtrade data not found: {comtrade_path}")
    
    df = pd.read_csv(comtrade_path)
    
    # Handle different possible column names
    year_col = None
    for col in ["date", "year", "refYear", "period"]:
        if col in df.columns:
            year_col = col
            break
    
    if year_col is None:
        raise ValueError("Comtrade CSV must have a year/date column")
    
    # Filter to target year
    year_data = df[pd.to_numeric(df[year_col], errors="coerce") == year]
    
    if len(year_data) == 0:
        logger.warning(f"No Comtrade data found for year {year}")
        return None
    
    # Aggregate trade values (sum if multiple rows)
    value_col = None
    for col in ["value", "trade_value_usd", "primaryValue", "fobvalue"]:
        if col in df.columns:
            value_col = col
            break
    
    if value_col is None:
        raise ValueError("Comtrade CSV must have a value column")
    
    total_value = pd.to_numeric(year_data[value_col], errors="coerce").sum()
    
    return {
        "year": int(year),
        "trade_value_usd": float(total_value),
        "n_records": int(len(year_data)),
    }


def calculate_metrics(
    sim_data: Dict[str, float],
    comtrade_data: Dict[str, float],
) -> Dict[str, float]:
    """Calculate validation metrics comparing simulation vs actual data."""
    metrics: Dict[str, float] = {}
    
    # For comparison, we'll use model's supply (Q) as proxy for trade flow
    # and compare normalized price index vs trade value
    model_trade_proxy = sim_data.get("Q", 0.0) * sim_data.get("P", 1.0)
    actual_trade = comtrade_data.get("trade_value_usd", 0.0)
    
    if actual_trade > 0:
        # Magnitude error (percentage)
        magnitude_error = abs(model_trade_proxy - actual_trade) / actual_trade
        metrics["magnitude_error_pct"] = float(magnitude_error * 100)
        
        # RMSE (using single point, so just squared error)
        metrics["rmse"] = float(np.sqrt((model_trade_proxy - actual_trade) ** 2))
        
        # Directional accuracy (did model predict increase/decrease correctly?)
        # For single year, we'd need previous year for comparison
        # For now, just compare sign of deviation
        deviation = model_trade_proxy - actual_trade
        metrics["absolute_deviation"] = float(abs(deviation))
        metrics["relative_deviation_pct"] = float((deviation / actual_trade) * 100)
    else:
        metrics["magnitude_error_pct"] = float("inf")
        metrics["rmse"] = float("inf")
        metrics["absolute_deviation"] = float("inf")
        metrics["relative_deviation_pct"] = float("inf")
    
    # Price comparison (normalized)
    model_price = sim_data.get("P", 1.0)
    # Normalize trade value to price index (rough approximation)
    # Use a baseline year's trade value as reference
    # For now, just report model price
    metrics["model_price_index"] = float(model_price)
    
    return metrics


def generate_llm_explanation(
    sim_data: Dict[str, float],
    comtrade_data: Dict[str, float],
    metrics: Dict[str, float],
    year: int,
) -> str:
    """Generate LLM-synthesized explanation of validation results."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, returning template explanation")
        return _template_explanation(sim_data, comtrade_data, metrics, year)
    
    if Anthropic is None:
        logger.warning("anthropic package not installed, returning template explanation")
        return _template_explanation(sim_data, comtrade_data, metrics, year)
    
    try:
        client = Anthropic(api_key=api_key)
        
        prompt = f"""You are analyzing a minerals supply chain model validation.

Model Simulation Results for {year}:
- Price (P): {sim_data.get('P', 0):.4f}
- Supply (Q): {sim_data.get('Q', 0):.2f}
- Demand (D): {sim_data.get('D', 0):.2f}
- Inventory (I): {sim_data.get('I', 0):.2f}
- Shortage: {sim_data.get('shortage', 0):.4f}
- Inventory Cover: {sim_data.get('cover', 0):.4f}

Actual Comtrade Trade Data for {year}:
- Trade Value (USD): ${comtrade_data.get('trade_value_usd', 0):,.0f}
- Number of Records: {comtrade_data.get('n_records', 0)}

Validation Metrics:
- Magnitude Error: {metrics.get('magnitude_error_pct', 0):.2f}%
- RMSE: {metrics.get('rmse', 0):,.0f}
- Absolute Deviation: {metrics.get('absolute_deviation', 0):,.0f}
- Relative Deviation: {metrics.get('relative_deviation_pct', 0):.2f}%

Provide a concise analysis (3-4 paragraphs) explaining:
1. What the model got right (if anything)
2. What the model got wrong and why
3. Potential reasons for discrepancies (model limitations, data quality, etc.)
4. Overall assessment of model performance for this year

Be specific and reference the actual numbers."""
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        
        explanation = message.content[0].text if message.content else ""
        return explanation.strip()
    
    except Exception as e:
        logger.error(f"Anthropic API call failed: {e}")
        return _template_explanation(sim_data, comtrade_data, metrics, year)


def _template_explanation(
    sim_data: Dict[str, float],
    comtrade_data: Dict[str, float],
    metrics: Dict[str, float],
    year: int,
) -> str:
    """Fallback template explanation when LLM is unavailable."""
    error_pct = metrics.get("magnitude_error_pct", float("inf"))
    
    if error_pct < 10:
        assessment = "excellent"
    elif error_pct < 25:
        assessment = "good"
    elif error_pct < 50:
        assessment = "moderate"
    else:
        assessment = "poor"
    
    return f"""Model Validation Summary for {year}

The model simulation was compared against actual UN Comtrade trade data for natural graphite.

Model Results:
- Price Index: {sim_data.get('P', 0):.4f}
- Supply: {sim_data.get('Q', 0):.2f}
- Demand: {sim_data.get('D', 0):.2f}
- Shortage: {sim_data.get('shortage', 0):.4f}

Actual Trade Data:
- Trade Value: ${comtrade_data.get('trade_value_usd', 0):,.0f}

Validation Metrics:
- Magnitude Error: {error_pct:.2f}%
- RMSE: {metrics.get('rmse', 0):,.0f}

Overall Assessment: {assessment.capitalize()} agreement between model and actual data.
Note: This is a template explanation. Set ANTHROPIC_API_KEY for detailed LLM analysis."""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate model simulation against historical Comtrade data"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to simulation run directory containing timeseries.csv",
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to validate (must exist in both simulation and Comtrade data)",
    )
    parser.add_argument(
        "--comtrade-path",
        type=str,
        default="data/canonical/comtrade_graphite_trade.csv",
        help="Path to normalized Comtrade CSV file",
    )
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    comtrade_path = Path(args.comtrade_path)
    year = args.year
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    logger.info(f"Loading simulation data from {run_dir} for year {year}")
    sim_data = load_simulation_data(run_dir, year)
    
    logger.info(f"Loading Comtrade data from {comtrade_path} for year {year}")
    comtrade_data = load_comtrade_data(comtrade_path, year)
    
    if comtrade_data is None:
        logger.error(f"No Comtrade data available for year {year}")
        return 1
    
    logger.info("Calculating validation metrics")
    metrics = calculate_metrics(sim_data, comtrade_data)
    
    logger.info("Generating LLM explanation")
    explanation = generate_llm_explanation(sim_data, comtrade_data, metrics, year)
    
    # Build validation report
    report = {
        "validation_year": year,
        "simulation_data": sim_data,
        "comtrade_data": comtrade_data,
        "metrics": metrics,
        "explanation": explanation,
        "run_directory": str(run_dir),
        "comtrade_source": str(comtrade_path),
    }
    
    # Save report
    output_path = run_dir / f"validation_{year}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nValidation Report for {year}")
    print("=" * 60)
    print(f"Model Price (P): {sim_data['P']:.4f}")
    print(f"Model Supply (Q): {sim_data['Q']:.2f}")
    print(f"Actual Trade Value: ${comtrade_data['trade_value_usd']:,.0f}")
    print(f"\nMetrics:")
    print(f"  Magnitude Error: {metrics['magnitude_error_pct']:.2f}%")
    print(f"  RMSE: {metrics['rmse']:,.0f}")
    print(f"  Relative Deviation: {metrics['relative_deviation_pct']:.2f}%")
    print(f"\nExplanation:")
    print(explanation)
    print(f"\nReport saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
