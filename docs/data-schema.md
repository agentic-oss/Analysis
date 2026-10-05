# Data Schema Specifications

## Processed Metals Schema (`data/processed/metals/metals_processed.parquet`)

- `date` (string): Observation date (YYYY-MM-DD).
- `symbol` (string): Instrument symbol (`GOLD`, `SILVER`, `GOLD_FUT`, etc.).
- `open` (float): Opening price.
- `high` (float): High price.
- `low` (float): Low price.
- `close` (float): Closing price.
- `volume` (float): Volume.
- `provider` (string): Source provider (`yahoo`, `synthetic`).
- `market` (string): Market category (`spot`, `futures`, `mcx`).
- `currency` (string): Currency (`USD`, `INR`).
- `unit` (string): Unit (`oz`, `10g`, `kg`).

## Technical Indicators Schema
- `return_1d`, `return_5d`, `return_20d`, `return_60d` (float): Percentage returns.
- `volatility_20d`, `volatility_60d` (float): Annualized rolling volatility.
- `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200` (float): Moving averages.
- `ema_9`, `ema_20`, `ema_50`, `ema_200` (float): Exponential moving averages.
- `rsi_14`, `stoch_rsi` (float): RSI indicators (0-100).
- `macd`, `macd_signal`, `macd_hist` (float): MACD components.
- `atr_14` (float): Average True Range.
- `bb_upper`, `bb_lower`, `bb_width` (float): Bollinger Bands.

## Machine-Readable Research Schema (`data/analysis/daily/YYYY-MM-DD.json`)
- `date`: YYYY-MM-DD
- `gold`: Composite scores, market regime, forecast signals, and price.
- `silver`: Composite scores, market regime, forecast signals, and price.
- `gold_silver_ratio`: Current ratio, Z-score, and percentile rank.
- `macro`: Detected macro regime and components.
- `alerts`: List of triggered alert events.
- `data_quality`: Records summary, warnings, and error counts.
