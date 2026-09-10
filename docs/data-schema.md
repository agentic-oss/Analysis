# Data Schema

## Raw & Processed Schema

Every market observation record contains:

```json
{
  "date": "YYYY-MM-DD",
  "open": float,
  "high": float,
  "low": float,
  "close": float,
  "volume": int/float,
  "symbol": string,
  "provider": string,
  "source": string,
  "market": string,
  "currency": string,
  "unit": string,
  "timezone": "UTC",
  "retrieval_timestamp": string
}
```

## Derived INR Conversion Schema

```json
{
  "date": "YYYY-MM-DD",
  "close": float, // USD/oz
  "usdinr": float,
  "converted_inr_10g": float, // Gold: (price / 31.1034768) * usdinr * 10
  "converted_inr_kg": float,  // Silver: (price / 31.1034768) * usdinr * 1000
  "conversion_rate_used": float,
  "conversion_timestamp": string
}
```

## Daily Analysis Schema (`data/analysis/daily/YYYY-MM-DD.json`)

Contains nested keys: `gold`, `silver`, `gold_silver_ratio`, `macro`, `scores`, `forecast_signals`, `data_quality`.
