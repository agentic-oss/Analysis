import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    """Evaluates past predictions against actual realized market outcomes and tracks performance metrics."""

    @staticmethod
    def evaluate_predictions(
        predictions_history_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare predicted returns/directions against actual realized outcomes."""
        if predictions_history_df.empty:
            return {"status": "no_predictions_available"}

        df = predictions_history_df.dropna(subset=["predicted_return", "actual_return"]).copy()
        if len(df) == 0:
            return {"status": "no_completed_outcomes"}

        pred_ret = df["predicted_return"].astype(float)
        act_ret = df["actual_return"].astype(float)

        mae = float(np.mean(np.abs(pred_ret - act_ret)))
        rmse = float(np.sqrt(np.mean((pred_ret - act_ret) ** 2)))
        dir_correct = ((pred_ret > 0) == (act_ret > 0)).astype(int)
        dir_accuracy = float(np.mean(dir_correct))

        # Accuracy breakdown by regime if available
        regime_perf = {}
        if "market_regime" in df.columns:
            for regime, grp in df.groupby("market_regime"):
                if len(grp) > 0:
                    r_pred = grp["predicted_return"].astype(float)
                    r_act = grp["actual_return"].astype(float)
                    regime_perf[str(regime)] = {
                        "sample_count": len(grp),
                        "directional_accuracy": round(float(np.mean((r_pred > 0) == (r_act > 0))), 4),
                        "mae": round(float(np.mean(np.abs(r_pred - r_act))), 4)
                    }

        return {
            "total_evaluated": len(df),
            "directional_accuracy": round(dir_accuracy, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "regime_breakdown": regime_perf
        }

    @staticmethod
    def save_model_performance(
        performance_data: Dict[str, Any],
        output_path: str = "data/models/model-performance.json"
    ) -> str:
        """Save model performance evaluation report."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(performance_data, f, indent=2)
        return output_path
