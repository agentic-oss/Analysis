# Data Schema

## Processed Metals Schema (`data/processed/metals/<symbol>.parquet`)
- `date` (string): Trading date YYYY-MM-DD
- `open`, `high`, `low`, `close` (float): Standard OHLC prices
- `volume` (float): Trading volume
- `symbol` (string): Instrument symbol
- `quality_status` (string): Quality classification (`valid`, `suspicious`, `invalid`)
- `sma_5`, `sma_10`, `sma_20`, `sma_50`, `sma_100`, `sma_200` (float)
- `ema_9`, `ema_20`, `ema_50`, `ema_200` (float)
- `rsi_14` (float): Relative Strength Index
- `stoch_rsi` (float): Stochastic RSI
- `macd`, `macd_signal`, `macd_hist` (float)
- `atr_14`, `volatility_20d` (float): Volatility metrics
- `bollinger_upper`, `bollinger_lower`, `bollinger_width` (float)
- `dist_sma_200_pct`, `dist_52w_high_pct`, `dist_52w_low_pct` (float)
- `return_1d`, `return_5d`, `return_20d`, `return_60d`, `return_252d` (float)

## Machine-Readable Daily JSON (`data/analysis/daily/YYYY-MM-DD.json`)
Contains asset prices, sub-scores, composite scores, market regimes, forward probabilistic signals, and data quality summary.
