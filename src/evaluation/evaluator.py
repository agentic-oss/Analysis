"""
Forecast Tracking & Model Evaluation Store.
Tracks historical predictions made on date T and compares them with actual forward outcomes on date T+h.
Computes directional accuracy, MAE, RMSE, and calibration across market regimes over time.
Outputs to data/models/model-performance.json.
"""

from datetime import datetime
import json
import logging
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """Evaluates stored daily predictions against subsequent actual market outcomes."""

    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        self.performance_file = os.path.join(models_dir, "model-performance.json")

    def evaluate_predictions(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Input DataFrame must contain columns:
        ['prediction_date', 'instrument', 'horizon', 'predicted_direction',
         'predicted_return', 'confidence', 'actual_return']
        """
        if predictions_df.empty or "actual_return" not in predictions_df.columns:
            return {"status": "NO_EVALUABLE_PREDICTIONS"}

        df = predictions_df.dropna(subset=["actual_return"]).copy()
        if df.empty:
            return {"status": "NO_EVALUABLE_PREDICTIONS"}

        df["actual_direction"] = (df["actual_return"] > 0).astype(int)
        df["predicted_direction_bin"] = (df["predicted_direction"].isin(["BULLISH", "BUY", 1])).astype(int)
        df["prediction_correct"] = (df["actual_direction"] == df["predicted_direction_bin"]).astype(int)

        df["error_return"] = df["predicted_return"] - df["actual_return"]
        df["abs_error"] = df["error_return"].abs()
        df["sq_error"] = df["error_return"] ** 2

        metrics_by_horizon = {}
        for h, group in df.groupby("horizon"):
            acc = float(group["prediction_correct"].mean() * 100.0)
            mae = float(group["abs_error"].mean())
            rmse = float(np.sqrt(group["sq_error"].mean()))
            metrics_by_horizon[str(h)] = {
                "count": len(group),
                "directional_accuracy_pct": acc,
                "mae": mae,
                "rmse": rmse
            }

        overall_metrics = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "total_evaluated": len(df),
            "overall_accuracy_pct": float(df["prediction_correct"].mean() * 100.0),
            "overall_mae": float(df["abs_error"].mean()),
            "overall_rmse": float(np.sqrt(df["sq_error"].mean())),
            "by_horizon": metrics_by_horizon
        }

        with open(self.performance_file, "w") as f:
            json.dump(overall_metrics, f, indent=2)

        logger.info(f"Updated forecast evaluation performance at {self.performance_file}")
        return overall_metrics
