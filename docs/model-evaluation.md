# Model Evaluation & Tracking

The `ForecastEvaluationTracker` (`src/models/forecasting_models.py`) evaluates historical predictions recorded in `data/predictions/daily/` against actual outcomes as days progress.

Tracks metrics:
* Directional Accuracy (%)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Performance breakdown by regime and volatility
* Feature drift and model degradation alerts
