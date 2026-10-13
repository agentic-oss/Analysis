# Data Schema Specification

## Daily Machine-Readable JSON Report Schema (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "gold": {
    "price": 0.0,
    "return_1d_pct": 0.0,
    "technical_score": 0.0,
    "macro_score": 0.0,
    "composite_score": 0.0,
    "regime": "Strong Bullish Trend | Normal Volatility",
    "signals": {
      "research_bias": "BULLISH",
      "confidence_score": 75.0,
      "confidence_reasons": [],
      "horizons": {}
    },
    "analogues": {}
  },
  "silver": {
    "price": 0.0,
    "return_1d_pct": 0.0,
    "technical_score": 0.0,
    "macro_score": 0.0,
    "composite_score": 0.0,
    "regime": "Bullish Trend | High Volatility",
    "signals": {},
    "analogues": {}
  },
  "gold_silver_ratio": {},
  "macro": {},
  "data_quality": {}
}
```
