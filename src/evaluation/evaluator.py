import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ForecastEvaluator:
    """Tracks historical predictions vs actual outcomes over time and logs model performance."""

    @staticmethod
    def evaluate_predictions(
        predictions_history_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Evaluates past recorded predictions against actual realized returns.
        Calculates directional accuracy, MAE, and performance breakdown.
        """
        if predictions_history_df.empty or 'actual_return' not in predictions_history_df.columns:
            return {"status": "no_eval_data"}

        evaluated = predictions_history_df.dropna(subset=['actual_return', 'predicted_return']).copy()
        if evaluated.empty:
            return {"status": "no_matched_records"}

        evaluated['directional_correct'] = (
            (evaluated['predicted_return'] > 0) == (evaluated['actual_return'] > 0)
        )

        acc = float(evaluated['directional_correct'].mean() * 100)
        mae = float((evaluated['predicted_return'] - evaluated['actual_return']).abs().mean())
        rmse = float(np.sqrt(((evaluated['predicted_return'] - evaluated['actual_return']) ** 2).mean()))

        return {
            "status": "success",
            "total_evaluated_forecasts": len(evaluated),
            "directional_accuracy_pct": round(acc, 2),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4)
        }
