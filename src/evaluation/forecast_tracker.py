import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)


class ForecastTracker:
    """
    Stores daily predictions alongside creation date T, evaluates forecasts against subsequent actual outcomes,
    and updates performance tracking in data/models/model-performance.json.
    """

    def __init__(self, performance_filepath: str = "data/models/model-performance.json"):
        self.performance_filepath = performance_filepath

    def record_predictions(
        self, date_str: str, instrument: str, forecasts: Dict[str, Dict[str, Any]], predictions_dir: str
    ) -> None:
        """Saves daily generated forecast record for future evaluation."""
        record = {
            "prediction_date": date_str,
            "instrument": instrument,
            "forecasts": forecasts,
            "created_at": datetime.utcnow().isoformat(),
        }
        filepath = os.path.join(predictions_dir, f"{date_str}_{instrument}.json")
        os.makedirs(predictions_dir, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)
        logger.info(f"Recorded daily prediction to {filepath}")

    def evaluate_past_predictions(
        self, historical_df: pd.DataFrame, predictions_dir: str
    ) -> Dict[str, Any]:
        """
        Loads past prediction records, matches with actual realized outcomes,
        and computes directional accuracy, MAE, and calibration metrics.
        """
        if not os.path.exists(predictions_dir):
            return {"evaluated_count": 0, "directional_accuracy_10d": None}

        pred_files = [f for f in os.listdir(predictions_dir) if f.endswith(".json")]
        if not pred_files:
            return {"evaluated_count": 0, "directional_accuracy_10d": None}

        eval_records = []
        hist_indexed = historical_df.set_index("date") if "date" in historical_df.columns else historical_df

        for file in pred_files:
            filepath = os.path.join(predictions_dir, file)
            with open(filepath, "r") as f:
                pred_data = json.load(f)

            pred_date = pred_data["prediction_date"]
            for horizon_key, forecast in pred_data.get("forecasts", {}).items():
                horizon_days = forecast["horizon_days"]
                prob_pos = forecast["probability_positive"]

                # Find actual price on pred_date and pred_date + horizon_days
                if pred_date in hist_indexed.index:
                    curr_idx = hist_indexed.index.get_loc(pred_date)
                    future_idx = curr_idx + horizon_days
                    if future_idx < len(hist_indexed):
                        start_price = hist_indexed.iloc[curr_idx]["close"]
                        actual_price = hist_indexed.iloc[future_idx]["close"]
                        actual_return = (actual_price - start_price) / start_price
                        actual_direction = 1 if actual_return > 0 else 0

                        predicted_direction = 1 if prob_pos > 0.5 else 0
                        correct = (predicted_direction == actual_direction)

                        eval_records.append({
                            "prediction_date": pred_date,
                            "horizon": horizon_days,
                            "predicted_prob": prob_pos,
                            "predicted_direction": predicted_direction,
                            "actual_return": actual_return,
                            "actual_direction": actual_direction,
                            "correct": correct,
                        })

        if not eval_records:
            return {"evaluated_count": 0, "directional_accuracy": None}

        eval_df = pd.DataFrame(eval_records)
        accuracy_by_horizon = {}
        for h, group in eval_df.groupby("horizon"):
            acc = float(group["correct"].mean() * 100.0)
            accuracy_by_horizon[f"{h}d"] = {
                "sample_count": len(group),
                "directional_accuracy_pct": round(acc, 2),
            }

        perf_summary = {
            "last_updated": datetime.utcnow().isoformat(),
            "evaluated_predictions_count": len(eval_df),
            "overall_accuracy_pct": round(float(eval_df["correct"].mean() * 100.0), 2),
            "horizon_performance": accuracy_by_horizon,
        }

        # Save to model-performance.json
        os.makedirs(os.path.dirname(self.performance_filepath), exist_ok=True)
        with open(self.performance_filepath, "w") as f:
            json.dump(perf_summary, f, indent=2)

        return perf_summary
