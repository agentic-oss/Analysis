# Data Schema Specification

## Processed Time-Series Parquet Schema (`data/processed/{symbol}.parquet`)
- `date` (string, ISO YYYY-MM-DD): Observation trading date.
- `symbol` (string): Instrument ticker identifier.
- `open`, `high`, `low`, `close` (float64): Normalized OHLC market prices.
- `volume` (float64): Traded volume.
- `currency` (string): Currency code (USD, INR).
- `unit` (string): Measurement unit (oz, 10g, kg, rate, index, points).
- `sma_5` ... `sma_200`, `ema_9` ... `ema_200` (float64): Moving average values.
- `rsi_14`, `stoch_rsi`, `macd`, `macd_signal`, `macd_hist`, `atr_14`, `volatility_20d` (float64): Technical indicator metrics.
- `quality_status` (string): Validation classification (`valid`, `suspicious`, `invalid`).

## Feature Store Parquet Schema (`data/features/daily/{instrument}_{date}.parquet`)
- Standardized technical and macro metrics.
- Target variables (`future_return_1d` ... `future_return_60d`, `future_direction_1d` ... `future_direction_60d`).
