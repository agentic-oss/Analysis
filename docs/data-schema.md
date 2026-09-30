# Data Schema Specification

## Daily Analysis JSON Schema (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "gold": {
    "price": 0.0,
    "daily_change_pct": 0.0,
    "volatility_20d": 0.0,
    "rsi_14": 0.0,
    "composite_score": 0.0,
    "scores": {}
  },
  "silver": {
    "price": 0.0,
    "daily_change_pct": 0.0
  },
  "gold_silver_ratio": {
    "current_ratio": 0.0,
    "z_score": 0.0,
    "percentile": 0.0
  },
  "macro": {
    "primary_regime": "",
    "explainability": ""
  },
  "analogues": {},
  "forecasts": {},
  "alerts": {}
}
```
