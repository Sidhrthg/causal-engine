"""Natural language interface to the causal engine."""

import os
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime
from anthropic import Anthropic


class CausalEngineInterface:
    """LLM-powered query interface for scenario generation and execution."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        
        self.client = Anthropic(api_key=self.api_key)
        
        # Load baseline config as template
        baseline_path = Path("scenarios/graphite_baseline.yaml")
        if baseline_path.exists():
            with open(baseline_path) as f:
                self.baseline_config = yaml.safe_load(f)
        else:
            self.baseline_config = None
    
    def query(self, user_query: str, execute: bool = True) -> dict:
        """
        Process natural language query.
        
        Args:
            user_query: Natural language question
            execute: If True, run the scenario after generating
            
        Returns:
            Dict with scenario, results (if executed), and explanation
        """
        
        # Generate scenario from query
        scenario_yaml = self._generate_scenario(user_query)
        
        if not execute:
            return {
                'query': user_query,
                'scenario': scenario_yaml,
                'executed': False
            }
        
        # Save scenario
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = scenario_yaml.get('name', f"llm_generated_{timestamp}")
        scenario_path = Path(f"scenarios/{scenario_name}.yaml")
        
        with open(scenario_path, 'w') as f:
            yaml.dump(scenario_yaml, f)
        
        print(f"\n📝 Scenario saved to: {scenario_path}")
        print(f"🚀 Running simulation...\n")
        
        # Execute scenario using subprocess
        import subprocess
        import sys
        
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.run_scenario', '--scenario', str(scenario_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"⚠️  Scenario execution had issues:")
            print(result.stderr)
            return {
                'query': user_query,
                'scenario': scenario_yaml,
                'scenario_path': str(scenario_path),
                'executed': False,
                'error': result.stderr
            }
        
        print(result.stdout)
        
        # Load results
        runs_dir = Path("runs") / scenario_name
        if not runs_dir.exists():
            return {
                'query': user_query,
                'scenario': scenario_yaml,
                'scenario_path': str(scenario_path),
                'executed': False,
                'error': "Run directory not found"
            }
        
        latest_run = sorted(runs_dir.glob("*"))[-1]
        
        results = {
            'timeseries': self._load_csv(latest_run / "timeseries.csv"),
            'metrics': self._load_json(latest_run / "metrics.json")
        }
        
        # Generate explanation
        explanation = self._explain_results(user_query, scenario_yaml, results)
        
        return {
            'query': user_query,
            'scenario': scenario_yaml,
            'scenario_path': str(scenario_path),
            'results': results,
            'explanation': explanation,
            'executed': True
        }
    
    def _generate_scenario(self, user_query: str) -> dict:
        """Use LLM to generate scenario YAML from natural language."""
        
        baseline_str = yaml.dump(self.baseline_config) if self.baseline_config else "No baseline available"
        
        prompt = f"""
You are a scenario generator for a graphite supply chain model.

Convert this user query into a valid scenario YAML configuration:

Query: {user_query}

Template (use baseline values unless query specifies otherwise):
{baseline_str}

Rules:
1. Extract shock type from query:
   - "restrict/restriction/export ban" → export_restriction
   - "demand surge/increase/spike" → demand_surge
   - "capacity shock" → capex_shock
   
2. Extract magnitude (convert to decimal):
   - "40%" → 0.40
   - "double" → 1.0
   
3. Extract timing:
   - "in 2025" → start_year: 2025, end_year: 2025
   - "from 2025 to 2027" → start_year: 2025, end_year: 2027
   
4. Generate descriptive name from query

5. Keep all baseline parameters unless query explicitly changes them

6. Set time horizon to cover shock year + 10 years for post-shock analysis

Return ONLY valid YAML, no explanation.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        
        yaml_text = response.content[0].text
        
        # Clean up markdown if present
        if "```yaml" in yaml_text:
            yaml_text = yaml_text.split("```yaml")[1].split("```")[0]
        elif "```" in yaml_text:
            yaml_text = yaml_text.split("```")[1].split("```")[0]
        
        scenario = yaml.safe_load(yaml_text)
        
        return scenario
    
    def _explain_results(self, query: str, scenario: dict, results: dict) -> str:
        """Generate natural language explanation of results."""
        
        metrics = results['metrics']
        
        prompt = f"""
User asked: {query}

Model ran this scenario:
{yaml.dump(scenario)}

Results:
- Total shortage: {metrics.get('total_shortage', 'N/A')}
- Peak shortage: {metrics.get('peak_shortage', 'N/A')}
- Average price: {metrics.get('avg_price', 'N/A')}
- Final inventory cover: {metrics.get('final_inventory_cover', 'N/A')}

Explain the results in 2-3 sentences:
1. What happened (shortage, price impact)
2. Why it happened (mechanism)
3. Key takeaway for policy

Be concise and specific to the numbers.
"""
        
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def _load_csv(self, path: Path) -> dict:
        """Load CSV as dict for serialization."""
        import pandas as pd
        df = pd.read_csv(path)
        return {
            'columns': df.columns.tolist(),
            'shape': df.shape,
            'head': df.head(5).to_dict('records')
        }
    
    def _load_json(self, path: Path) -> dict:
        """Load JSON file."""
        with open(path) as f:
            return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Query the causal engine in natural language")
    parser.add_argument("query", help="Natural language query")
    parser.add_argument("--no-execute", action="store_true", 
                       help="Generate scenario only, don't execute")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    interface = CausalEngineInterface(api_key=args.api_key)
    
    print(f"\n🤖 Processing query: {args.query}\n")
    
    result = interface.query(args.query, execute=not args.no_execute)
    
    print("=" * 60)
    print("SCENARIO GENERATED:")
    print("=" * 60)
    print(yaml.dump(result['scenario'], default_flow_style=False))
    
    if result['executed']:
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(json.dumps(result['results']['metrics'], indent=2))
        
        print("\n" + "=" * 60)
        print("EXPLANATION:")
        print("=" * 60)
        print(result['explanation'])
        print()
    else:
        print("\nScenario generated but not executed (use without --no-execute to run)")


if __name__ == "__main__":
    main()
