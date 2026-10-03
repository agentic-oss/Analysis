# Probabilistic Forecasting & Historical Analogues

## Historical Analogue Matching
When evaluating the market on date T, the system constructs a standardized feature state vector:
- `rsi_14`, `macd_hist`, `volatility_20d`, `distance_sma_20`, `distance_sma_50`, `distance_sma_200`, `return_5d`, `return_20d`.

It searches all historical observations up to date T (strictly excluding future dates) to find the top K nearest Euclidean neighbors.

## Forward Return Distributions
For horizons 1D, 3D, 5D, 10D, 20D, 60D, empirical distributions are derived:
- Sample count
- Positive-return probability (%)
- Negative-return probability (%)
- Mean return (%)
- Median return (%)
- Percentile distribution (P10, P25, P50, P75, P90)
- Historical min/max range
