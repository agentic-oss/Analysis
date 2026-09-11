# Data Schemas & Storage Specifications

## Directory Structure

Data is persisted hierarchically in `data/`:
* `data/raw/YYYY/MM/YYYY-MM-DD/`: Partitioned raw OHLCV datasets per symbol.
* `data/processed/metals/`: Derived indicator and converted datasets (`gold_processed.parquet`, `silver_processed.parquet`, `indian_metals_processed.parquet`).
* `data/features/daily/`: Daily ML-ready feature datasets (`gold_features_YYYY-MM-DD.parquet`).
* `data/analysis/daily/`: Machine-readable JSON summary outputs (`YYYY-MM-DD.json`).
* `data/analysis/alerts/`: Triggered alert JSON logs (`YYYY-MM-DD.json`).
* `data/quality/`: Ingestion data validation reports (`YYYY-MM-DD.json`).
* `data/predictions/daily/`: Recorded model forecasts for performance tracking.
* `data/models/`: Model performance evaluation metrics (`model-performance.json`).

## Machine-Readable JSON Schema (`data/analysis/daily/YYYY-MM-DD.json`)

```json
{
  "date": "2026-03-31",
  "generated_at": "2026-03-31T18:00:00Z",
  "scores": {
    "composite_score": 62.5,
    "technical_score": 65.0,
    "macro_score": 60.0,
    "momentum_score": 58.0,
    "volatility_score": 70.0,
    "relative_value_score": 55.0,
    "historical_pattern_score": 64.0,
    "model_score": 50.0
  },
  "gold": {
    "price_usd": 2050.0,
    "price_inr_10g": 57920.0,
    "return_1d_pct": 0.45,
    "rsi_14": 58.2,
    "trend_regime": "Bullish Trend"
  },
  "silver": {
    "price_usd": 25.5,
    "price_inr_kg": 72100.0,
    "return_1d_pct": 0.80
  },
  "gold_silver_ratio": {
    "gold_silver_ratio": 80.39,
    "ratio_percentile_252": 62.4,
    "ratio_zscore": 0.45
  },
  "macro": {
    "primary_regime": "Inflationary"
  },
  "historical_analogues": {},
  "probabilistic_signals": {},
  "alerts": [],
  "data_quality": {}
}
```
