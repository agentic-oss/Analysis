# Historical Analogue & Forecasting Methodology

## Historical Analogue Engine
When today's market state is observed, the Historical Analogue Engine searches the historical database for multi-factor comparable situations strictly prior to today (`target_date - 30 days`).

### Scaled Euclidean Similarity Score
$$ \text{Distance} = \sqrt{ \sum_{i} \left( \frac{x_{t,i} - x_{\text{hist},i}}{\sigma_{i}} \right)^2 } $$
$$ \text{Similarity Score} = \frac{100}{1 + \text{Distance}} $$

## Probabilistic Signal Generation
Signals are generated across horizons: 1D, 3D, 5D, 10D, 20D, and 60D.
For each horizon, conditional probabilities are calculated:
- Probability of positive return
- Expected median return
- Maximum gain and drawdown in analogue sample
- Confidence score (sample size, model agreement, score magnitude, regime stability)
