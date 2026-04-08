"""Test synthetic control on actual Comtrade data."""

import pandas as pd
from pathlib import Path
from src.minerals.causal_identification import SyntheticControl


def load_comtrade_data():
    """Load and prepare Comtrade data."""
    
    # Load your normalized Comtrade file
    data_path = Path("data/canonical/comtrade_graphite_trade.normalized.csv")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Comtrade data not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Check columns
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    
    return df


def test_2008_shock():
    """Test synthetic control on 2008 trade shock."""
    
    df = load_comtrade_data()
    
    # Initialize synthetic control
    sc = SyntheticControl(verbose=True)
    
    # Estimate treatment effect
    # Adjust column names based on your actual data
    result = sc.estimate_treatment_effect(
        data=df,
        treated_unit="USA",  # Adjust based on your data
        control_units=["DEU", "JPN", "IND"],  # ISO codes or country names
        treatment_time=2008,
        outcome_var="trade_value_usd",  # Adjust to your column name
        unit_col="to_entity",  # or "country"
        time_col="year"
    )
    
    print("\n=== 2008 Trade Shock Analysis ===")
    print(f"\nTreatment Effect:")
    print(result.treatment_effect)
    print(f"\nControl Unit Weights:")
    for unit, weight in result.weights.items():
        print(f"  {unit}: {weight:.3f}")
    print(f"\nPre-treatment fit (RMSE): {result.pre_treatment_rmse:.2f}")
    
    # Calculate percent change
    baseline = result.actual.iloc[0]
    pct_change = (result.treatment_effect / baseline * 100).mean()
    print(f"\nAverage % change from counterfactual: {pct_change:.1f}%")
    
    return result


if __name__ == "__main__":
    result = test_2008_shock()
