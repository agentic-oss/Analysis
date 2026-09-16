# Forecasting Engine

## Historical Analogue Engine
Uses multi-factor Euclidean similarity scoring across normalized technical and macro feature vectors (RSI, 200DMA distance, Volatility) to locate top $K$ historical market states. Computes forward return statistics for horizons 1D, 3D, 5D, 10D, 20D, and 60D.

## Machine Learning Models
- Classification: Random Forest Classifier & Gradient Boosting Classifier for directional signals.
- Regression: Ridge Regression for continuous forward return forecasts.
- Validation: Walk-forward expanding window validation without random shuffling.
