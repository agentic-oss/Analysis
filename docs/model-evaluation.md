# Model Evaluation & Forecast Calibration Tracking

The forecast evaluation engine (`src/evaluation/`):
1. Loads historical prediction JSON files generated on date T.
2. Compares expected returns and directional probabilities with realized actual market returns as time elapses.
3. Calculates Directional Accuracy, MAE, and RMSE across forecast horizons.
4. Updates model monitoring logs at `data/models/model-performance.json`.
