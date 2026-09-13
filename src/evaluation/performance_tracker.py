import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ForecastPerformanceTracker:
    """Tracks historical prediction accuracy vs actual outcomes over time."""

    def evaluate_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        predictions_df columns:
        [prediction_date, symbol, horizon, predicted_direction, expected_return, actual_return]
        """
        if predictions_df.empty or 'actual_return' not in predictions_df.columns:
            return {"status": "no_evaluable_predictions"}

        df = predictions_df.dropna(subset=['actual_return']).copy()
        if len(df) == 0:
            return {"status": "no_actual_outcomes_yet"}

        df['actual_direction'] = np.where(df['actual_return'] > 0, "Bullish", "Bearish")
        df['prediction_correct'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
        df['abs_error'] = (df['expected_return'] - df['actual_return']).abs()
        df['sq_error'] = (df['expected_return'] - df['actual_return']) ** 2

        accuracy = float(df['prediction_correct'].mean())
        mae = float(df['abs_error'].mean())
        rmse = float(np.sqrt(df['sq_error'].mean()))

        # Evaluation breakdown by horizon
        horizon_metrics = {}
        for h, group in df.groupby('horizon'):
            horizon_metrics[str(h)] = {
                "count": len(group),
                "directional_accuracy": float(group['prediction_correct'].mean()),
                "mae": float(group['abs_error'].mean()),
                "rmse": float(np.sqrt(group['sq_error'].mean()))
            }

        return {
            "total_evaluations": len(df),
            "overall_directional_accuracy": round(accuracy, 4),
            "overall_mae": round(mae, 4),
            "overall_rmse": round(rmse, 4),
            "accuracy_by_horizon": horizon_metrics
        }
