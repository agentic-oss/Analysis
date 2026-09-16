import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """Tracks prior prediction accuracy against actual realized outcomes over time."""

    def __init__(self, predictions_dir: str = "data/predictions/daily", model_metrics_path: str = "data/models/model-performance.json"):
        self.predictions_dir = predictions_dir
        self.model_metrics_path = model_metrics_path
        os.makedirs(os.path.dirname(self.model_metrics_path), exist_ok=True)

    def log_daily_prediction(
        self,
        date_str: str,
        instrument: str,
        horizon: int,
        predicted_direction: int,
        predicted_return: float,
        confidence: float,
    ) -> None:
        """Stores daily prediction alongside creation date for future evaluation."""
        rec = {
            "prediction_date": date_str,
            "instrument": instrument,
            "horizon": horizon,
            "predicted_direction": predicted_direction,
            "predicted_return": round(predicted_return, 6),
            "confidence": round(confidence, 2),
            "actual_return": None,
            "prediction_correct": None,
            "evaluation_date": None,
        }

        os.makedirs(self.predictions_dir, exist_ok=True)
        path = os.path.join(self.predictions_dir, f"{date_str}_{instrument.lower()}_{horizon}d.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2)

    def evaluate_past_predictions(self, market_data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Scans saved prediction files, checks if horizon has elapsed, and evaluates accuracy."""
        if not os.path.exists(self.predictions_dir):
            return {"evaluated_count": 0, "overall_accuracy": 0.0}

        evaluated = []
        for filename in os.listdir(self.predictions_dir):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.predictions_dir, filename)
            with open(filepath, "r") as f:
                rec = json.load(f)

            if rec.get("actual_return") is not None:
                evaluated.append(rec)
                continue

            # Evaluate un-evaluated predictions
            pred_date = pd.to_datetime(rec["prediction_date"])
            symbol = rec["instrument"]
            horizon = rec["horizon"]

            if symbol in market_data_dict:
                m_df = market_data_dict[symbol].sort_values("date").reset_index(drop=True)
                m_df["date"] = pd.to_datetime(m_df["date"])

                match_idx = m_df[m_df["date"] >= pred_date].index
                if len(match_idx) > 0:
                    start_i = match_idx[0]
                    target_i = start_i + horizon
                    if target_i < len(m_df):
                        start_price = m_df.loc[start_i, "close"]
                        target_price = m_df.loc[target_i, "close"]
                        actual_ret = (target_price - start_price) / start_price
                        actual_dir = 1 if actual_ret > 0 else 0

                        rec["actual_return"] = round(float(actual_ret), 6)
                        rec["prediction_correct"] = bool(rec["predicted_direction"] == actual_dir)
                        rec["evaluation_date"] = datetime.utcnow().strftime("%Y-%m-%d")

                        with open(filepath, "w") as f_out:
                            json.dump(rec, f_out, indent=2)

                        evaluated.append(rec)

        if not evaluated:
            return {"evaluated_count": 0, "overall_accuracy": 0.0}

        correct_count = sum(1 for e in evaluated if e.get("prediction_correct"))
        accuracy = float(correct_count / len(evaluated)) if evaluated else 0.0

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "evaluated_predictions_count": len(evaluated),
            "overall_accuracy": round(accuracy, 4),
            "status": "stable" if accuracy >= 0.50 else "drift_warning",
        }

        with open(self.model_metrics_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
