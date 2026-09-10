# Forecasting & Analogue Engine

## Historical Analogue Engine

The historical analogue engine finds past market environments similar to today's state vector $V_T = [\text{RSI}_{14}, \text{dist\_SMA}_{20}, \text{Volatility}_{20d}, \text{MACD}]$.

1. **Standardization:** Features across historical dates $t < T$ are Z-score standardized:
   $$Z_{t, i} = \frac{X_{t, i} - \mu_i}{\sigma_i}$$
2. **Euclidean Distance:**
   $$D(T, t) = \sqrt{\sum_{i=1}^M (Z_{T, i} - Z_{t, i})^2}$$
3. **Top K Match Filtering:** Selects top $K=25$ closest historical dates.

## Forward Return Probability Estimation

For each prediction horizon $H \in \{1, 3, 5, 10, 20, 60\}$ trading days, forward returns $R_{t+H}$ from the $K$ matches are extracted:
- **Positive Return Probability:** $P(R_{t+H} > 0) = \frac{1}{K} \sum_{k=1}^K I(R_{t_k+H} > 0)$
- **Expected Return:** $\mathbb{E}[R_{H}] = \frac{1}{K} \sum_{k=1}^K R_{t_k+H}$
- **Percentiles:** 10th percentile (downside risk), 50th percentile (median), 90th percentile (upside potential).
