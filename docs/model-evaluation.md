# Model Evaluation & Monitoring

## Walk-Forward Validation

To prevent look-ahead bias and data leakage, ML models (Logistic Regression, Random Forest, Gradient Boosting) are evaluated using walk-forward expanding window validation:
- Minimum initial training window $N_{train} = 100$ business days.
- Step size $\Delta N = 20$ business days.
- Out-of-sample predictions are collected across all step periods.

## Evaluated Metrics

- **Classification:** Accuracy, Precision, Recall, F1 score.
- **Regression:** MAE, RMSE, $R^2$.
- **Drift Monitoring:** Historical directional forecasts generated on date $T$ are stored alongside realized outcomes once $T+H$ date elapses. Aggregate directional accuracy and MAE are updated continuously in `data/models/model-performance.json`.
