# Forecasting & Historical Analogue Engine

The historical analogue engine (`src/patterns/` and `src/forecasting/`) evaluates today's market state against historical states:
1. State vectors normalized across RSI, distance from moving averages, and rolling volatility.
2. Euclidean similarity scoring yields top historical analogue dates.
3. Subsequent realized returns across 1D, 3D, 5D, 10D, 20D, and 60D horizons form conditional probabilistic distributions.
