# Forecasting & Historical Analogue Search Engine

## Historical Analogue Engine

The historical analogue search engine identifies past trading days whose market state vector closely matches today's market state.

### State Vector Feature Components
1. RSI(14)
2. MACD Histogram
3. Distance to 20 DMA (%)
4. Distance to 50 DMA (%)
5. Distance to 200 DMA (%)
6. 20-Day Annualized Volatility (%)
7. 5-Day Return (%)
8. 20-Day Return (%)

### Matching & Outcome Calculation
- Features are z-score standardized over historical lookback data.
- Euclidean distances between today's vector and all historical vectors (excluding the most recent 60 trading days) are computed.
- Top $N$ analogues (default 15) are selected.
- Forward outcomes across 1D, 3D, 5D, 10D, 20D, 60D horizons are evaluated to construct statistical distributions (mean, median, positive probability, percentile distributions, max gain, max loss).
