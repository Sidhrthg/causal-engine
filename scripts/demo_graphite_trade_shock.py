import json
from pathlib import Path

import networkx as nx
import pandas as pd


NORMALIZED = Path("data/canonical/comtrade_graphite_trade.normalized.csv")
OUT_JSON = Path("outputs/demo_graphite_trade_shock.json")


def load_normalized_series(path: Path) -> pd.Series:
    """
    Your normalized schema is:
    year,P,D,Q,I,source,units,notes

    We treat P as the signal value (trade_value_usd).
    Returns a Series indexed by year (int).
    """
    df = pd.read_csv(path)
    s = pd.Series(df["P"].astype(float).values, index=df["year"].astype(int).values)
    return s.sort_index()


def shock_multiply(series: pd.Series, year: int, pct: float) -> pd.Series:
    """
    pct=-0.3 => -30%
    Returns a new shocked series (does not mutate input).
    """
    s = series.copy()
    if year not in s.index:
        raise ValueError(f"Year {year} not found in series index")
    s.loc[year] = s.loc[year] * (1.0 + pct)
    return s


def main():
    # 1) Minimal structural graph
    G = nx.DiGraph()
    G.add_node("country:World", kind="country")
    G.add_node("country:USA", kind="country")

    signal_id = "graphite_trade_world_usa"
    G.add_edge(
        "country:World",
        "country:USA",
        material="graphite_natural",
        relation="import_trade",
        units="USD",
        signal_id=signal_id,
        signal_path=str(NORMALIZED),
    )

    # 2) Load signal by reference (edge attribute)
    edge = G.edges["country:World", "country:USA"]
    base = load_normalized_series(Path(edge["signal_path"]))

    # 3) Apply a minimal scenario shock
    shock_year = 2008
    shock_pct = -0.30
    shocked = shock_multiply(base, year=shock_year, pct=shock_pct)

    # 4) Write a small, audit-friendly output
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "edge": ["country:World", "country:USA"],
        "material": edge["material"],
        "relation": edge["relation"],
        "signal_id": edge["signal_id"],
        "signal_path": edge["signal_path"],
        "shock": {"type": "multiplicative", "year": shock_year, "pct": shock_pct},
        "before": float(base.loc[shock_year]),
        "after": float(shocked.loc[shock_year]),
        "delta": float(shocked.loc[shock_year] - base.loc[shock_year]),
        "date_range": [int(base.index.min()), int(base.index.max())],
        "n_points": int(len(base)),
    }

    print(json.dumps(out, indent=2))
    OUT_JSON.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()

