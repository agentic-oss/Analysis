# Data Schema Specifications

This document outlines the JSON and Parquet schemas used across raw storage, features, daily analysis, quality reports, and alerts.

## 1. Raw & Processed Time-Series Parquet Schema

File path: `data/raw/YYYY/MM/YYYY-MM-DD/{dataset}.parquet` or `data/processed/metals/{dataset}.parquet`

| Field Name | Type | Description | Example |
|---|---|---|---|
| `date` | String (ISO) | Trade date | `"2026-03-31"` |
| `open` | Float64 | Opening price | `2015.50` |
| `high` | Float64 | Period high price | `2030.00` |
| `low` | Float64 | Period low price | `2005.20` |
| `close` | Float64 | Closing price | `2025.10` |
| `volume` | Int64 / Float64 | Trading volume | `125000` |
| `provider` | String | Data provider | `"yfinance"` |
| `retrieval_timestamp` | String (ISO) | Retrieval time in UTC | `"2026-03-31T18:00:00Z"` |
| `instrument` | String | Symbol name | `"GOLD"` |
| `market` | String | Market type | `"spot"` |
| `currency` | String | ISO currency code | `"USD"` |
| `unit` | String | Measurement unit | `"oz"` |
| `exchange` | String | Primary exchange | `"LBMA"` |
| `timezone` | String | Timezone | `"UTC"` |
| `validation_status` | String | Record quality status | `"valid"` |

## 2. Daily Analysis JSON Schema

File path: `data/analysis/daily/YYYY-MM-DD.json`

```json
{
  "date": "2026-03-31",
  "gold": {
    "metrics": {
      "price": 2025.10,
      "return_1d": 0.005,
      "return_5d": 0.012,
      "rsi_14": 62.5,
      "macd": 12.4,
      "volatility_20d": 0.14
    },
    "regime": {
      "trend_regime": "Strong Bullish Trend",
      "volatility_regime": "Normal Volatility"
    },
    "scores": {
      "composite_score": 68.5,
      "confidence_score": 85.0,
      "confidence_reasons": ["65% positive forward return frequency"]
    },
    "predictions": {
      "predicted_direction": "Bullish",
      "predicted_return": 0.008,
      "model_agreement": "3/3"
    }
  },
  "silver": { "..." : "..." },
  "gold_silver_ratio": {
    "current_ratio": 82.5,
    "ratio_zscore": 1.2
  },
  "macro": {
    "primary_regime": "Inflationary"
  },
  "data_quality": {
    "gold_records": 100,
    "silver_records": 100,
    "warnings": [],
    "errors": []
  }
}
```
