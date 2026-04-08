"""Validate model against aggregate trade data."""

import pandas as pd
import json
from pathlib import Path


def validate_2008_shock():
    """Compare model prediction to actual aggregate data."""
    
    # Load actual Comtrade data
    actual = pd.read_csv('data/canonical/comtrade_graphite_trade.csv')
    actual['year'] = pd.to_datetime(actual['date']).dt.year
    actual = actual[['year', 'value']].rename(columns={'value': 'actual_trade_value'})
    
    # Load model simulation for 2008 shock
    # Find most recent run
    runs_dir = Path('runs/graphite_trade_shock_2008')
    if not runs_dir.exists():
        print("No runs found. Run: python -m scripts.run_scenario --scenario scenarios/graphite_trade_shock_2008.yaml")
        return
    
    latest_run = sorted(runs_dir.glob('*'))[-1]
    sim = pd.read_csv(latest_run / 'timeseries.csv')
    
    # Merge
    comparison = actual.merge(sim[['year', 'shortage', 'P']], on='year', how='inner')
    
    # Calculate changes
    baseline_years = [2005, 2006, 2007]
    baseline_trade = comparison[comparison['year'].isin(baseline_years)]['actual_trade_value'].mean()
    comparison['pct_change_from_baseline'] = (
        (comparison['actual_trade_value'] - baseline_trade) / baseline_trade * 100
    )
    
    print("=== Model vs Actual Comparison (2008 Shock) ===")
    print(comparison[comparison['year'] >= 2008][['year', 'actual_trade_value', 'pct_change_from_baseline', 'shortage', 'P']])
    
    # 2008 specific
    trade_2008 = comparison[comparison['year'] == 2008]['actual_trade_value'].iloc[0]
    pct_change_2008 = comparison[comparison['year'] == 2008]['pct_change_from_baseline'].iloc[0]
    shortage_2008 = comparison[comparison['year'] == 2008]['shortage'].iloc[0]
    
    print(f"\n2008 Analysis:")
    print(f"  Actual trade value: ${trade_2008:,.0f}")
    print(f"  % change from baseline: {pct_change_2008:.1f}%")
    print(f"  Model predicted shortage: {shortage_2008:.1f}")
    
    # Use LLM to synthesize (if API key available)
    try:
        from anthropic import Anthropic
        import os
        
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        
        prompt = f"""
        Analyze this graphite supply chain model validation for 2008:
        
        Model Prediction (2008 shock scenario):
        - Predicted shortage: {shortage_2008:.1f} units
        - Shock magnitude: 30% export restriction
        
        Actual Trade Data (UN Comtrade):
        - 2008 trade value: ${trade_2008:,.0f} USD
        - Change from baseline (2005-2007 avg): {pct_change_2008:.1f}%
        - Baseline: ${baseline_trade:,.0f} USD
        
        Questions:
        1. Is the model directionally correct? (Did trade increase or decrease as expected?)
        2. What's the magnitude discrepancy?
        3. What might explain the gap?
        4. What data limitations affect this comparison?
        
        Note: This is aggregate World→USA data, not country-specific.
        """
        
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        
        print("\n=== LLM Analysis ===")
        print(response.content[0].text)
        
    except Exception as e:
        print(f"\nLLM analysis skipped: {e}")


if __name__ == "__main__":
    validate_2008_shock()
