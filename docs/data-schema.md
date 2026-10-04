# Data Schema Specification

## Daily Machine-Readable Analysis Output (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "gold": {
    "price": float,
    "return_1d_pct": float,
    "return_5d_pct": float,
    "return_20d_pct": float,
    "trend_regime": string,
    "volatility_regime": string,
    "rsi_14": float,
    "macd": float,
    "sma_50": float,
    "sma_200": float,
    "bb_upper": float,
    "dist_sma_200_pct": float,
    "composite_score": float,
    "research_bias": string,
    "signals": {
      "1d": {
        "horizon_days": int,
        "prob_positive": float,
        "prob_negative": float,
        "expected_return_pct": float,
        "median_return_pct": float,
        "volatility_pct": float,
        "historical_sample_count": int,
        "signal_direction": string,
        "disclaimer": string
      }
    },
    "analogue_sample_count": int
  },
  "silver": { ... },
  "gold_silver_ratio": {
    "current_ratio": float,
    "zscore": float,
    "percentile_252": float
  },
  "macro": {
    "macro_regime": string,
    "explainable_components": {
      "dxy_return_20d": float,
      "us10y_change_20d": float,
      "sp500_return_20d": float,
      "oil_return_20d": float,
      "vix_level": float
    }
  },
  "data_quality": {
    "total_instruments_validated": int,
    "total_errors": int,
    "total_warnings": int
  }
}
```

## Daily Quality Report Output (`data/quality/YYYY-MM-DD.json`)

```json
{
  "date": "YYYY-MM-DD",
  "instruments": {
    "GOLD": {
      "symbol": "GOLD",
      "record_count": int,
      "warnings": [string],
      "errors": [string],
      "valid_count": int,
      "suspicious_count": int,
      "invalid_count": int
    }
  },
  "total_instruments_validated": int,
  "total_errors": int,
  "total_warnings": int
}
```
