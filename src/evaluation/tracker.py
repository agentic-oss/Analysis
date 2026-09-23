import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List

class ForecastEvaluator:
    """
    Evaluates historical predictions against realized actual outcomes.

    Tracks:
    - Directional Accuracy (%)
    - Mean Absolute Error (MAE)
    - Root Mean Squared Error (RMSE)
    - Accuracy by Horizon (1d, 3d, 5d, 10d, 20d, 60d)
    - Model Performance Monitoring output saved to data/models/model-performance.json
    """
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)

    def evaluate_predictions(self, historical_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not historical_predictions:
            return {"total_predictions": 0, "overall_accuracy_pct": 0.0, "horizons": {}}

        df = pd.DataFrame(historical_predictions)

        # Require columns: prediction_date, instrument, horizon, predicted_direction, predicted_return, actual_return
        if "actual_return" not in df.columns or df["actual_return"].isna().all():
            return {"total_predictions": len(df), "evaluable_predictions": 0, "overall_accuracy_pct": 0.0}

        clean_df = df.dropna(subset=["actual_return", "predicted_direction"]).copy()
        if clean_df.empty:
            return {"total_predictions": len(df), "evaluable_predictions": 0, "overall_accuracy_pct": 0.0}

        clean_df["actual_direction"] = (clean_df["actual_return"] > 0).astype(int)
        clean_df["is_correct"] = (clean_df["predicted_direction"] == clean_df["actual_direction"]).astype(int)

        overall_acc = float(clean_df["is_correct"].mean() * 100.0)
        mae = float((clean_df["predicted_return"] - clean_df["actual_return"]).abs().mean()) if "predicted_return" in clean_df else 0.0
        rmse = float(np.sqrt(((clean_df["predicted_return"] - clean_df["actual_return"]) ** 2).mean())) if "predicted_return" in clean_df else 0.0

        horizon_metrics = {}
        for h, group in clean_df.groupby("horizon"):
            acc = float(group["is_correct"].mean() * 100.0)
            h_mae = float((group["predicted_return"] - group["actual_return"]).abs().mean()) if "predicted_return" in group else 0.0
            horizon_metrics[str(h)] = {
                "count": len(group),
                "accuracy_pct": round(acc, 1),
                "mae": round(h_mae, 2)
            }

        report = {
            "evaluation_timestamp": pd.Timestamp.now().isoformat(),
            "total_forecasts_evaluated": len(clean_df),
            "overall_directional_accuracy_pct": round(overall_acc, 1),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "horizon_breakdown": horizon_metrics
        }

        report_path = os.path.join(self.models_dir, "model-performance.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report
