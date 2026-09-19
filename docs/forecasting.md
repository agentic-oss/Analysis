# Forecasting Engine & Historical Analogues

## Historical Analogue Matching
1. Normalizes multi-factor state vector (RSI, MACD, ATR, Volatility, Distance from 200 SMA) using rolling standardization.
2. Computes Euclidean distance against historical states.
3. Filters top $K=15$ closest historical matches prior to $t - \max(\text{horizons})$.
4. Extracts empirical forward return distributions across 1D, 3D, 5D, 10D, 20D, and 60D horizons.

## Probabilistic Output Schema
- `probability_positive_pct`: % of historical analogue outcomes with $>0\%$ return.
- `probability_negative_pct`: % of historical analogue outcomes with $<0\%$ return.
- `expected_mean_return_pct`: Mean return across matched historical analogues.
- `median_return_pct`: Median return across matched historical analogues.
- `expected_volatility_pct`: Standard deviation of return outcomes.
- `confidence_score`: 0-100 score reflecting sample size, composite score alignment, and probability agreement.
