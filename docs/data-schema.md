# Data Schema Documentation

## Feature Store (`data/features/daily/YYYY-MM-DD.parquet`)
- `date`: YYYY-MM-DD string
- `symbol`: Instrument symbol
- `open`, `high`, `low`, `close`, `volume`
- `sma_20`, `sma_50`, `sma_200`, `ema_9`, `ema_20`, `ema_50`, `ema_200`
- `rsi_14`, `stoch_rsi`, `macd`, `macd_signal`, `macd_hist`
- `atr_14`, `volatility_20d`, `volatility_60d`
- `dist_sma_200`, `dist_52w_high`, `dist_52w_low`
- `days_to_next_event`, `days_since_prev_event`
- `future_return_1d`, `future_return_5d`, `future_return_20d` (targets)
- `future_direction_1d`, `future_direction_5d`, `future_direction_20d` (targets)
