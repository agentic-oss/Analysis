import logging
import json
import os
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    """Evaluates prediction performance, calculates classification/regression metrics, and maintains forecast history tracking."""

    @staticmethod
    def evaluate_predictions(y_true_dir: List[int], y_pred_dir: List[int], y_true_ret: List[float], y_pred_ret: List[float]) -> Dict[str, Any]:
        if not y_true_dir or len(y_true_dir) == 0:
            return {}

        acc = float(accuracy_score(y_true_dir, y_pred_dir))
        prec = float(precision_score(y_true_dir, y_pred_dir, zero_division=0))
        rec = float(recall_score(y_true_dir, y_pred_dir, zero_division=0))
        f1 = float(f1_score(y_true_dir, y_pred_dir, zero_division=0))

        mae = float(mean_absolute_error(y_true_ret, y_pred_ret)) if len(y_true_ret) > 0 else 0.0
        rmse = float(np.sqrt(mean_squared_error(y_true_ret, y_pred_ret))) if len(y_true_ret) > 0 else 0.0
        r2 = float(r2_score(y_true_ret, y_pred_ret)) if len(y_true_ret) > 1 else 0.0

        return {
            "classification": {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            },
            "regression": {
                "mae": mae,
                "rmse": rmse,
                "r2_score": r2
            }
        }

class ForecastTracker:
    """Tracks stored predictions on date T against subsequent actual outcomes."""

    def __init__(self, tracking_file: str = "data/models/prediction_history.json"):
        self.tracking_file = tracking_file
        os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading prediction history: {e}")
        return []

    def record_prediction(self, prediction_date: str, instrument: str, horizon: int, predicted_dir: str, predicted_ret: float, confidence: float):
        entry = {
            "prediction_date": prediction_date,
            "instrument": instrument,
            "horizon": horizon,
            "predicted_direction": predicted_dir,
            "predicted_return": predicted_ret,
            "confidence": confidence,
            "actual_return": None,
            "prediction_correct": None
        }
        self.history.append(entry)
        self._save_history()

    def update_actual_outcomes(self, current_df: pd.DataFrame, instrument: str):
        if not self.history or current_df.empty:
            return

        date_map = dict(zip(current_df["date"], current_df["close"]))

        updated = False
        for entry in self.history:
            if entry["instrument"] == instrument and entry["actual_return"] is None:
                p_date = entry["prediction_date"]
                horizon = entry["horizon"]

                if p_date in date_map:
                    dates_list = list(current_df["date"])
                    try:
                        p_idx = dates_list.index(p_date)
                        target_idx = p_idx + horizon
                        if target_idx < len(dates_list):
                            p_price = current_df.loc[p_idx, "close"]
                            target_price = current_df.loc[target_idx, "close"]
                            actual_ret = float((target_price - p_price) / p_price)

                            entry["actual_return"] = actual_ret
                            actual_dir = "Bullish" if actual_ret > 0 else "Bearish"
                            entry["prediction_correct"] = (entry["predicted_direction"] == actual_dir)
                            updated = True
                    except Exception:
                        pass

        if updated:
            self._save_history()

    def _save_history(self):
        with open(self.tracking_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.history:
            return {"total_predictions": 0, "evaluated_predictions": 0, "overall_accuracy": 0.0}

        evaluated = [e for e in self.history if e["prediction_correct"] is not None]
        if not evaluated:
            return {"total_predictions": len(self.history), "evaluated_predictions": 0, "overall_accuracy": 0.0}

        correct = sum(1 for e in evaluated if e["prediction_correct"] is True)
        acc = float(correct / len(evaluated))

        return {
            "total_predictions": len(self.history),
            "evaluated_predictions": len(evaluated),
            "overall_accuracy": acc
        }
