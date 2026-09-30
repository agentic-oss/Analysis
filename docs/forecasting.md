# Forecasting & Analogue Engine

## Historical Analogue Engine

Finds the $K$ most similar historical market states using weighted Euclidean distance across normalized features:

- RSI(14)
- MACD Normalized
- SMA Distances (20d, 200d)
- 20d Volatility
- Gold/Silver Ratio Z-Score
- DXY 20d Return
- 10Y Yield 20d Change
- VIX Level

For selected analogues, the system calculates forward return distributions across 1D, 3D, 5D, 10D, 20D, and 60D horizons.

## Statistical Models

Ensemble of expanding-window models:
1. Historical Base Rate
2. Logistic Regression
3. Random Forest
4. Gradient Boosting
