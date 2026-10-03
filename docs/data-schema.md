# Data Schemas

## 1. Raw & Processed Price Series (`data/processed/metals/gold_daily.parquet`)
- `Date` (datetime64): Observation date (YYYY-MM-DD)
- `Open` (float64): Opening price
- `High` (float64): Session high price
- `Low` (float64): Session low price
- `Close` (float64): Session closing price
- `Adj Close` (float64): Adjusted closing price
- `Volume` (int64/float64): Session trading volume
- `validation_status` (string): Classification (`valid`, `suspicious`, `invalid`)
- `USDINR_rate` (float64): USD/INR exchange rate used for conversion
- `Close_INR_10g` (float64): Converted Indian price per 10 grams (for Gold)
- `Close_INR_kg` (float64): Converted Indian price per kilogram (for Silver)
- `conversion_timestamp` (string): ISO timestamp of conversion

## 2. Daily Analysis JSON Output (`data/analysis/daily/YYYY-MM-DD.json`)
```json
{
  "date": "YYYY-MM-DD",
  "gold": {
    "price": float,
    "price_inr_10g": float,
    "daily_move_pct": float,
    "rsi": float,
    "dist_200dma": float,
    "regime": {
      "trend_regime": string,
      "volatility_regime": string
    },
    "scores": {
      "composite_score": float,
      "components": object
    },
    "confidence": {
      "research_bias": string,
      "confidence_score": int,
      "reasons": array
    },
    "forward_statistics": object
  },
  "silver": { ... },
  "gold_silver_ratio": { ... },
  "macro": { ... },
  "market_regime": { ... },
  "data_quality": { ... },
  "alerts": { ... }
}
```

## 3. Daily ML Feature Dataset (`data/features/daily/YYYY-MM-DD_gold_features.parquet`)
Contains current state observations (`rsi_14`, `macd_hist`, `volatility_20d`, `distance_sma_200`, `macro_*`) alongside future target returns (`future_return_1d`, `future_return_3d`, `future_return_5d`, `future_return_10d`, `future_return_20d`, `future_return_60d`).
