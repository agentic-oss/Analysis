# Model Evaluation & Performance Tracking Specification

## Continuous Forecast Tracking
Every forecast generated on date $T$ is logged in `data/predictions/daily/history.json`. On subsequent dates $T + h$, actual price moves are recorded to track:
- Directional accuracy
- MAE & RMSE
- Model performance summary in `data/models/model-performance.json`
- Feature and prediction drift alerts
