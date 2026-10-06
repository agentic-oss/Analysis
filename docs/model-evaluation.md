# Model Evaluation & Forecast Performance Tracking

To ensure continuous system self-evaluation and accountability, all predictions are systematically logged and tracked over time.

## Forecast Tracking Architecture (`src/evaluation/evaluator.py`)

1. **Prediction Recording**: When a forecast is generated on date T for horizon $H$, an immutable record is stored in `data/models/prediction_history.json`:
   - `prediction_date`
   - `instrument`
   - `horizon`
   - `predicted_direction`
   - `predicted_return`
   - `confidence`

2. **Outcome Settlement**: When the pipeline runs on date $T + H$, the actual subsequent return is computed and the entry is updated:
   - `actual_return`
   - `prediction_correct` (True / False)

3. **Model Performance Monitoring**: Evaluates aggregated classification metrics (Accuracy, Precision, Recall, F1) and regression metrics (MAE, RMSE, $R^2$) logged to `data/models/model-performance.json`.
