# Data Schemas & Storage Format

## Machine-Readable Daily JSON (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-31",
  "gold": {
    "price": 2150.5,
    "inr_price": 65000.0,
    "fx_rate_used": 83.0,
    "composite_score": 78.5,
    "scores": {
      "technical_score": 80.0,
      "macro_score": 75.0,
      "momentum_score": 60.0,
      "volatility_score": 70.0,
      "relative_value_score": 75.0,
      "historical_pattern_score": 64.0,
      "model_score": 62.0
    },
    "regime": {
      "trend": "Bullish Trend",
      "volatility": "Normal Volatility",
      "composite_regime": "Bullish Trend | Normal Volatility | Neutral Momentum"
    },
    "signals": {
      "5d": {
        "horizon_days": 5,
        "research_bias": "Bullish Signal",
        "prob_positive_return": 62.5,
        "prob_negative_return": 37.5,
        "expected_return_pct": 1.2,
        "expected_volatility_pct": 4.5,
        "confidence_score": 68.0
      }
    }
  },
  "silver": {},
  "gold_silver_ratio": {
    "current": 84.6,
    "z_score": 1.8,
    "percentile": 88.5,
    "signal": "Extreme High GSR - Favor Silver Outperformance"
  },
  "macro": {},
  "data_quality": {},
  "alerts": []
}
```
