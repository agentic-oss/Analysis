# Data Schema Specification

## Daily Analysis JSON Schema (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "2026-10-02",
  "gold": {
    "price": 2500.0,
    "return_1d_pct": 0.5,
    "return_5d_pct": 1.2,
    "return_20d_pct": 3.4,
    "technical_score": 72.0,
    "macro_score": 75.0,
    "regime": "Strong Bullish Trend (Normal Volatility)",
    "research_bias": "Bullish",
    "rsi_14": 62.5,
    "macd": 12.4,
    "dist_200dma": 5.2,
    "scores": {
      "technical_score": 72.0,
      "macro_score": 75.0,
      "momentum_score": 65.0,
      "volatility_score": 70.0,
      "relative_value_score": 50.0,
      "historical_pattern_score": 64.0,
      "composite_score": 66.5,
      "score_interpretation": "Bullish"
    },
    "forward_statistics": {
      "forecasts": {
        "10d": {
          "positive_return_probability": 0.64,
          "expected_return_pct": 1.8,
          "research_signal": "Bullish Signal",
          "confidence_score": 73.0
        }
      }
    }
  },
  "silver": {},
  "gold_silver_ratio": {
    "current_ratio": 85.2,
    "ratio_zscore": 1.8,
    "relative_value_regime": "Ratio Elevated"
  },
  "macro": {},
  "data_quality": {
    "summary": {
      "total_records": 200,
      "valid_count": 195,
      "suspicious_count": 5,
      "invalid_count": 0
    }
  }
}
```

## Quality Report Schema (`data/quality/YYYY-MM-DD.json`)

```json
{
  "date": "2026-10-02",
  "expected_symbol": "ALL",
  "warnings": [],
  "errors": [],
  "summary": {
    "total_records": 100,
    "valid_count": 100,
    "suspicious_count": 0,
    "invalid_count": 0
  }
}
```
