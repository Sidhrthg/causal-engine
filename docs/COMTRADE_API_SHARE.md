# Comtrade API – Code to share

Use this to get UN Comtrade bilateral trade data (e.g. graphite HS 2504) with a subscription key.

---

## 1. Get an API key

- Sign up at **https://comtradeplus.un.org/** and obtain a **subscription key** for the API.

---

## 2. Store the key (do not commit it)

Create a `.env` file in the project root:

```env
COMTRADE_API_KEY=your_subscription_key_here
```

Add `.env` to `.gitignore` so the key is never committed.

---

## 3. Python: request one bilateral flow

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COMTRADE_API_KEY")
if not API_KEY:
    raise ValueError("COMTRADE_API_KEY not set in .env")

# UN Comtrade API (HS classification, annual)
BASE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"

def get_bilateral_flow(reporter_code: str, partner_code: str, year: str) -> dict | None:
    """Get one bilateral flow: reporter imports from partner. HS 2504 = natural graphite."""
    params = {
        "reporterCode": reporter_code,   # e.g. "842" = USA
        "partnerCode": partner_code,     # e.g. "76" = Brazil, "0" = World
        "cmdCode": "2504",               # HS 2504 = graphite; natural
        "period": year,                  # e.g. "2010"
        "flowCode": "M",                 # M = imports
        "subscription-key": API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    if r.status_code == 200:
        return r.json()
    return None

# Example: USA imports from World, 2010
data = get_bilateral_flow("842", "0", "2010")
# Response has "data" list with primaryValue (USD), netWgt (kg), etc.
```

---

## 4. Common reporter / partner codes

| Code | Country / area |
|------|-----------------|
| 842  | USA             |
| 156  | China           |
| 76   | Brazil          |
| 32   | Argentina       |
| 0    | World (total)   |

Full lists: UN Comtrade country codes.

---

## 5. Run the project’s downloader

With `COMTRADE_API_KEY` in `.env`:

```bash
pip install pandas requests python-dotenv
python -m scripts.download_comtrade_specific
```

Menu options: USA imports (Africa + South America + East Asia), China imports, intra-region flows. Outputs go to `data/canonical/comtrade_*.csv`.

---

**Summary:** Put the subscription key in `.env` as `COMTRADE_API_KEY`, then use the `subscription-key` query parameter in requests to `https://comtradeapi.un.org/data/v1/get/C/A/HS` with `reporterCode`, `partnerCode`, `cmdCode` (2504 for graphite), `period`, and `flowCode` (M for imports).
