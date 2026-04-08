import pandas as pd
from pathlib import Path

IN_PATH = Path("data/raw/comtrade/TradeData_1_20_2026_18_16_50.csv")
OUT_PATH = Path("data/canonical/comtrade_graphite_trade.csv")

# Load
df = pd.read_csv(IN_PATH).reset_index(drop=True)

# ---- Filters ----
flow = df["flowCode"].astype(str).str.strip().str.lower()
cmd = df["cmdCode"].astype(str).str.strip().str.lower()

df = df[
    flow.eq("import") &
    (cmd.eq("graphite; natural"))
].copy()

# ---- Date (THIS IS THE ONLY DATE LINE) ----
date = df["refPeriodId"].astype(int)

# ---- Value ----
value = pd.to_numeric(df["fobvalue"], errors="coerce")

# ---- Output ----
out = pd.DataFrame({
    "date": date,
    "series": "trade_value_usd",
    "value": value,
    "unit": "USD",
    "material": "graphite_natural",
    "from_entity": "country:" + df["partnerISO"].astype(str),
    "to_entity": "country:" + df["reporterISO"].astype(str),
    "source": "UN_Comtrade",
})

out = out.dropna(subset=["date", "value"]).sort_values("date")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_PATH, index=False)

print(f"Wrote {len(out)} rows to {OUT_PATH}")
print(out.head())
print("Date range:", out["date"].min(), "to", out["date"].max())
