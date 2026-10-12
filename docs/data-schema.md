# Data Schema Specifications

## Daily Analysis JSON Schema (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "gold": {
    "price": 2000.0,
    "return_1d_pct": 0.5,
    "market_regime": "Strong Bullish Trend",
    "scores": {
      "technical_score": 65.0,
      "macro_score": 75.0,
      "composite_score": 68.5
    }
  },
  "silver": {
    "price": 24.5,
    "return_1d_pct": 1.2
  },
  "indian_market": {
    "gold_inr_per_10g": 65000.0,
    "silver_inr_per_kg": 78000.0,
    "usdinr_rate": 83.0
  },
  "macro": {
    "regime": "Inflationary"
  },
  "gold_silver_ratio": {
    "current_ratio": 81.6,
    "z_score": 0.2
  },
  "historical_analogues": {
    "top_k": 5,
    "analogues": []
  },
  "forecasts": {},
  "data_quality": {
    "invalid_records": 0
  }
}
```
