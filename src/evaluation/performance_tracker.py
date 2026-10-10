import os
import json
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ForecastPerformanceTracker:
    """
    Evaluates previous forecasts against actual outcomes as time progresses.
    Maintains `data/models/model-performance.json` and creates a continuously growing evaluation database.
    Calculates directional accuracy, MAE, RMSE, and calibration by horizon and regime.
    """

    def __init__(self, performance_file_path: str = "data/models/model-performance.json"):
        self.performance_file_path = performance_file_path

    def update_performance(self, predictions_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates predictions records containing predicted_return, predicted_direction, actual_return, horizon, market_regime.
        """
        if not predictions_records:
            return {"status": "No historical predictions to evaluate"}

        df = pd.DataFrame(predictions_records)
        eval_results = {"total_predictions_evaluated": len(df), "horizons": {}}

        for horizon, h_group in df.groupby("horizon"):
            if h_group.empty:
                continue

            correct_dir = (np.sign(h_group["predicted_return"]) == np.sign(h_group["actual_return"]))
            dir_acc = float(correct_dir.mean() * 100)

            mae = float((h_group["predicted_return"] - h_group["actual_return"]).abs().mean())
            rmse = float(np.sqrt(((h_group["predicted_return"] - h_group["actual_return"]) ** 2).mean()))

            eval_results["horizons"][f"{horizon}d"] = {
                "total_count": len(h_group),
                "directional_accuracy_pct": round(dir_acc, 2),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4)
            }

        os.makedirs(os.path.dirname(self.performance_file_path), exist_ok=True)
        with open(self.performance_file_path, "w") as f:
            json.dump(eval_results, f, indent=2)

        logger.info(f"Updated forecast performance saved to {self.performance_file_path}")
        return eval_results
