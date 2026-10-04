# Forecast Evaluation & Model Monitoring

## Outcome Verification Engine

Every daily forecast stored at date $T$ is eventually evaluated on date $T+h$ against realized close prices:

* **Directional Accuracy**: $\frac{1}{N} \sum \mathbb{I}(\text{sign}(\hat{y}) == \text{sign}(y))$
* **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |\hat{y} - y|$
* **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (\hat{y} - y)^2}$
* **Regime Breakdown**: Performance tracking split across market and macro regimes saved to `data/models/model-performance.json`.
