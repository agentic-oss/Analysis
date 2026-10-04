# Historical Analogue Engine & Forecasting

## Multi-Factor Similarity Search

Given today's market vector $X_T = [\text{RSI}_{14}, \text{MACD}, \text{Volatility}_{20d}, \text{DistSMA}_{200}, \dots]$, the historical analogue engine searches all prior dates $t < T$:

1. Feature Standardization: $Z_{t,i} = \frac{X_{t,i} - \mu_i}{\sigma_i}$
2. Distance Metric: Euclidean Distance $d(T, t) = \sqrt{\sum_i (Z_{T,i} - Z_{t,i})^2}$
3. Top-$K$ Selection: Identifies top $K=20$ most similar historical market states.
4. Conditional Probability Distribution: Computes mean, median, standard deviation, and positive/negative return probabilities for horizons $h \in \{1, 3, 5, 10, 20, 60\}$ days.
