# Data Schema Specifications

## Processed Market Time-Series Schema
- `date` (string, YYYY-MM-DD)
- `instrument_id` (string)
- `open`, `high`, `low`, `close` (float)
- `volume` (float)
- `currency` (string)
- `unit` (string)
- `provider` (string)
- `retrieval_timestamp` (ISO timestamp)

## Features Daily Dataset Schema (`data/features/daily/`)
- Technical indicators: `rsi_14`, `macd`, `stoch_rsi`, `atr_14`, `volatility_20d`, `dist_sma_200`, `bb_width`, etc.
- Macro indicators: `macro_dxy_ret20d`, `macro_us_real_yield_ret20d`, `gold_silver_ratio`, `gold_silver_ratio_zscore`
- Target fields (strictly isolated): `future_return_1d`, `future_return_5d`, `future_return_10d`, `future_return_20d`, `future_return_60d`
