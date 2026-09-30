# Forecast Performance Tracking & Model Evaluation

Every generated forecast on date T is stored in `data/predictions/daily/`. As time progresses, `ForecastTracker` compares predictions against actual subsequent prices, tracking:

- Directional Accuracy (%)
- MAE / RMSE
- Calibration across market regimes and volatility regimes
- Performance summary output to `data/models/model-performance.json`
