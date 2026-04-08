# scripts/plot_run.py

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        required=True,
        help="Path to run directory containing timeseries.csv"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (default: outputs/<run_name>.png)"
    )
    args = parser.parse_args()

    run_dir = Path(args.run)
    ts_path = run_dir / "timeseries.csv"

    if not ts_path.exists():
        raise FileNotFoundError(f"Missing timeseries.csv in {run_dir}")

    df = pd.read_csv(ts_path)

    # ---- Basic sanity checks ----
    required = {"year", "shortage", "P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(df["year"], df["shortage"], color="crimson", linewidth=2)
    ax1.set_ylabel("Shortage", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.set_xlabel("Year")

    ax2 = ax1.twinx()
    ax2.plot(df["year"], df["P"], color="navy", linestyle="--", linewidth=2)
    ax2.set_ylabel("Price (P)", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")

    title = run_dir.name.replace("_", " ")
    plt.title(title)

    fig.tight_layout()

    # ---- Save ----
    out_path = (
        Path(args.out)
        if args.out
        else Path("outputs") / f"{run_dir.name}.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved figure → {out_path}")

if __name__ == "__main__":
    main()

