# Methodology and Forecasting Principles

## Research Philosophy
All price forecasts in this system are strictly expressed as **probabilistic research signals** and historical conditional distributions, rather than guaranteed future prices.

## Historical Analogue Engine
When today's market state is observed, the system normalizes key feature vectors (RSI, Moving Average Distances, Volatility, Gold/Silver Ratio Z-score, Macro variables) and searches historical observations prior to date $T$ using Euclidean distance:

$$d(x_t, x_i) = \sqrt{\sum_{k} w_k \left( \frac{x_{t,k} - \mu_k}{\sigma_k} - \frac{x_{i,k} - \mu_k}{\sigma_k} \right)^2}$$

The top $N$ closest historical analogues are selected, and their realized forward returns $R_{i, h}$ at horizons $h \in \{1, 3, 5, 10, 20, 60\}$ are analyzed to calculate:
- Probability of positive return: $P(R_{h} > 0)$
- Expected / Median return
- Volatility and range metrics

## Temporal Correctness & Look-Ahead Protection
- Features generated at date $T$ strictly use observations available on or before date $T$.
- Model training uses expanding-window walk-forward validation.
- Forecast performance is tracked by evaluating prediction records saved at date $T$ against actual outcomes observed at date $T+h$.
