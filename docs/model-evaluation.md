# Model Evaluation & Tracking

## Validation Methodology
- **Time-Series Split**: Uses non-shuffled 70/30 temporal train/test split.
- **Walk-Forward Validation**: Evaluates models across expanding window time periods.

## Tracked Evaluation Metrics
- **Logistic Regression**: Directional Accuracy %, Precision, Recall, ROC-AUC.
- **Random Forest Regressor**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- **Persistent Storage**: Saved to `data/models/model-performance.json`.
