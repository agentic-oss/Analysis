# Forecasting & Historical Analogue Engine

## Historical Analogue Engine (`src/patterns/analogue_engine.py`)

Searches past states ($T - 60$ days prior) to match current normalized feature vectors across RSI, MACD, Volatility, SMA distances, and Gold/Silver Ratio Z-score.

Calculates forward return metrics across top $K=15$ matches for horizons:
* 1 Day
* 3 Days
* 5 Days
* 10 Days
* 20 Days
* 60 Days

Calculates probability of positive return, expected mean/median return, volatility, and max gain/loss.

## Probabilistic Signal Generation (`src/forecasting/probabilistic.py`)

Combines analogue positive return frequencies with technical and macro composite scores into probabilistic research signals with confidence scoring.
