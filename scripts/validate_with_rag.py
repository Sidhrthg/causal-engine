"""Validate model predictions against historical data using RAG."""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from datetime import datetime


class ModelValidator:
    """Compare model predictions to actual trade data with LLM analysis."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = Anthropic(api_key=self.api_key)
    
    def validate_run(
        self,
        run_dir: str,
        reference_year: int = None,
        comtrade_path: str = "data/canonical/comtrade_graphite_trade.csv"
    ) -> dict:
        """
        Validate a simulation run against historical data.
        
        Args:
            run_dir: Path to simulation run directory
            reference_year: Year to focus validation on (optional)
            comtrade_path: Path to Comtrade data
            
        Returns:
            Validation report with comparison and LLM analysis
        """
        
        print(f"\n🔍 Validating run: {run_dir}")
        print(f"📊 Loading data...\n")
        
        # Load simulation results
        sim_data = self._load_simulation(run_dir)
        
        # Load actual trade data
        actual_data = self._load_comtrade(comtrade_path)
        
        # Merge and compare
        comparison = self._compare_data(sim_data, actual_data, reference_year)
        
        # Generate LLM analysis with RAG
        analysis = self._generate_rag_analysis(
            sim_data=sim_data,
            actual_data=actual_data,
            comparison=comparison,
            reference_year=reference_year
        )
        
        # Save validation report
        report = {
            'run_dir': run_dir,
            'reference_year': reference_year,
            'timestamp': datetime.now().isoformat(),
            'comparison': comparison,
            'llm_analysis': analysis,
            'data_sources': {
                'simulation': str(run_dir),
                'comtrade': comtrade_path
            }
        }
        
        output_path = Path(run_dir) / f"validation_report_{reference_year or 'full'}.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Report saved to: {output_path}\n")
        
        return report
    
    def _load_simulation(self, run_dir: str) -> dict:
        """Load simulation timeseries and metrics."""
        run_path = Path(run_dir)
        
        # Load timeseries
        ts_path = run_path / "timeseries.csv"
        if not ts_path.exists():
            raise FileNotFoundError(f"Timeseries not found: {ts_path}")
        
        timeseries = pd.read_csv(ts_path)
        
        # Load metrics
        metrics_path = run_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        return {
            'timeseries': timeseries,
            'metrics': metrics
        }
    
    def _load_comtrade(self, path: str) -> pd.DataFrame:
        """Load and prepare Comtrade data."""
        df = pd.read_csv(path)
        
        # date column is integer years (2000, 2001, ...)
        df['year'] = df['date'].astype(int)
        
        # Rename for consistency
        df = df.rename(columns={'value': 'trade_value_usd'})
        
        return df[['year', 'trade_value_usd', 'material', 'from_entity', 'to_entity']]
    
    def _compare_data(
        self,
        sim_data: dict,
        actual_data: pd.DataFrame,
        reference_year: int = None
    ) -> dict:
        """Compare simulation predictions to actual data."""
        
        sim_ts = sim_data['timeseries']
        
        # Calculate baseline (pre-shock average)
        if reference_year:
            baseline_years = [reference_year - 3, reference_year - 2, reference_year - 1]
        else:
            # Use first 3 years as baseline
            baseline_years = sim_ts['year'].head(3).tolist()
        
        baseline_trade = actual_data[
            actual_data['year'].isin(baseline_years)
        ]['trade_value_usd'].mean()
        
        # Merge simulation and actual data
        merged = sim_ts.merge(
            actual_data[['year', 'trade_value_usd']],
            on='year',
            how='inner'
        )
        
        if len(merged) == 0:
            return {
                'error': 'No overlapping years between simulation and actual data',
                'sim_years': sim_ts['year'].tolist(),
                'actual_years': actual_data['year'].tolist()
            }
        
        # Calculate key metrics
        merged['trade_pct_change'] = (
            (merged['trade_value_usd'] - baseline_trade) / baseline_trade * 100
        )
        
        comparison = {
            'baseline_trade_usd': float(baseline_trade),
            'baseline_years': baseline_years,
            'overlapping_years': merged['year'].tolist(),
            'n_years': len(merged)
        }
        
        # Focus on reference year if specified
        if reference_year and reference_year in merged['year'].values:
            ref_data = merged[merged['year'] == reference_year].iloc[0]
            
            comparison['reference_year'] = {
                'year': reference_year,
                'model_prediction': {
                    'shortage': float(ref_data.get('shortage', 0)),
                    'price': float(ref_data.get('P', 0)),
                    'inventory_cover': float(ref_data.get('I', 0) / ref_data.get('D', 1)) if 'I' in ref_data and 'D' in ref_data else None
                },
                'actual_data': {
                    'trade_value_usd': float(ref_data['trade_value_usd']),
                    'pct_change_from_baseline': float(ref_data['trade_pct_change'])
                }
            }
        
        # Overall comparison stats
        comparison['summary'] = {
            'avg_model_shortage': float(merged['shortage'].mean()) if 'shortage' in merged else None,
            'avg_model_price': float(merged['P'].mean()) if 'P' in merged else None,
            'avg_actual_trade': float(merged['trade_value_usd'].mean()),
            'trade_volatility': float(merged['trade_value_usd'].std())
        }
        
        return comparison
    
    def _generate_rag_analysis(
        self,
        sim_data: dict,
        actual_data: pd.DataFrame,
        comparison: dict,
        reference_year: int = None
    ) -> str:
        """Generate LLM analysis using RAG approach."""
        
        # Prepare context from retrieved data
        context = self._prepare_context(sim_data, actual_data, comparison)
        
        prompt = f"""You are analyzing a graphite supply chain causal model's predictions against actual UN Comtrade trade data.

## Model Predictions (Simulation):
{context['model_summary']}

## Actual Historical Data (UN Comtrade):
{context['actual_summary']}

## Comparison:
{context['comparison_summary']}

## Your Task:
Analyze this validation with rigorous causal reasoning:

1. **Directional Accuracy**: Is the model directionally correct? Did both show shortage/surplus in the same direction?

2. **Magnitude Assessment**: What's the quantitative discrepancy? Is it within reasonable bounds?

3. **Mechanism Diagnosis**: Based on the comparison, what causal mechanisms might be:
   - Correctly specified?
   - Missing or mis-specified?
   - Over/under-estimated?

4. **Parameter Implications**: What do these results suggest about key parameters like:
   - Demand elasticity (η_D)
   - Capacity adjustment speed (τ_K)
   - Price responsiveness (α_P)

5. **Data Limitations**: What limitations in the comparison should we acknowledge?
   - Aggregate vs. partner-level data
   - Trade value vs. physical quantity
   - External factors not in model

Be specific, quantitative, and focus on actionable insights for model improvement.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _prepare_context(
        self,
        sim_data: dict,
        actual_data: pd.DataFrame,
        comparison: dict
    ) -> dict:
        """Prepare context strings for LLM analysis."""
        
        metrics = sim_data['metrics']
        
        def _fmt(key, default=0):
            v = metrics.get(key, default)
            return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
        
        model_summary = f"""
Model Scenario Results:
- Total shortage over period: {_fmt('total_shortage')} units
- Peak shortage: {_fmt('peak_shortage')} units
- Average price: ${_fmt('avg_price')}
- Final inventory cover: {_fmt('final_inventory_cover')} years

Key Parameters Used:
- Demand elasticity (η_D): -0.25
- Capacity adjustment time (τ_K): 3.0 years
- Price adjustment speed (α_P): 0.80
"""
        
        actual_summary = f"""
UN Comtrade Trade Data:
- Baseline trade value (3-year avg): ${comparison.get('baseline_trade_usd', 0):,.0f} USD
- Years available: {comparison.get('n_years', 0)} years overlap with simulation
- Average trade value: ${comparison.get('summary', {}).get('avg_actual_trade', 0):,.0f} USD
- Trade volatility (std dev): ${comparison.get('summary', {}).get('trade_volatility', 0):,.0f} USD
"""
        
        if 'reference_year' in comparison:
            ref = comparison['reference_year']
            comparison_summary = f"""
Focus Year: {ref['year']}

Model Predicted:
- Shortage: {ref['model_prediction']['shortage']:.2f} units
- Price: ${ref['model_prediction']['price']:.2f}

Actual Data:
- Trade value: ${ref['actual_data']['trade_value_usd']:,.0f} USD
- Change from baseline: {ref['actual_data']['pct_change_from_baseline']:.1f}%

Note: Direct comparison is difficult as model outputs shortage/price while data shows aggregate trade value.
"""
        else:
            summary = comparison.get('summary') or {}
            avg_short = summary.get('avg_model_shortage', 0) or 0
            avg_price = summary.get('avg_model_price', 0) or 0
            avg_trade = summary.get('avg_actual_trade', 0)
            comparison_summary = f"""
Overall Period Comparison:
- Model avg shortage: {avg_short:.2f} units
- Model avg price: ${avg_price:.2f}
- Actual avg trade: ${avg_trade:,.0f} USD

Years analyzed: {comparison.get('n_years', 0)} overlapping years
"""
        
        return {
            'model_summary': model_summary,
            'actual_summary': actual_summary,
            'comparison_summary': comparison_summary
        }


def main():
    parser = argparse.ArgumentParser(description="Validate model against historical data")
    parser.add_argument("--run-dir", required=True, help="Path to simulation run directory")
    parser.add_argument("--year", type=int, help="Reference year to focus validation on")
    parser.add_argument("--comtrade", default="data/canonical/comtrade_graphite_trade.csv",
                       help="Path to Comtrade data")
    parser.add_argument("--api-key", help="Anthropic API key")
    
    args = parser.parse_args()
    
    validator = ModelValidator(api_key=args.api_key)
    
    report = validator.validate_run(
        run_dir=args.run_dir,
        reference_year=args.year,
        comtrade_path=args.comtrade
    )
    
    print("=" * 70)
    print("VALIDATION ANALYSIS")
    print("=" * 70)
    print(report['llm_analysis'])
    print("\n" + "=" * 70)
    print(f"\n✅ Full report saved to: {Path(args.run_dir)}/validation_report_{args.year or 'full'}.json\n")


if __name__ == "__main__":
    main()
