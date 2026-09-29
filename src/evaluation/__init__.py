import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger("ForecastEvaluator")


class ForecastEvaluator:
    """
    Evaluates prior predictions saved on date T against actual realized outcomes
    as time progresses. Stores outputs in a continuously growing evaluation database.
    """
    def __init__(self, predictions_dir: str = "data/predictions", models_dir: str = "data/models"):
        self.predictions_dir = os.path.join(predictions_dir, "daily")
        self.models_dir = models_dir
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def evaluate_past_predictions(self, current_df: pd.DataFrame, instrument: str = "GOLD") -> Dict[str, Any]:
        """
        Scans historical prediction JSON files and compares predicted returns/direction
        with realized actual returns in current_df.
        """
        if not os.path.exists(self.predictions_dir):
            return {"evaluated_count": 0}

        pred_files = [f for f in os.listdir(self.predictions_dir) if f.endswith(".json")]
        if not pred_files:
            return {"evaluated_count": 0}

        current_df = current_df.sort_values("date").reset_index(drop=True)
        dates = current_df["date"].tolist()
        prices = dict(zip(current_df["date"], current_df["close"]))

        evaluation_records = []

        for pfile in pred_files:
            filepath = os.path.join(self.predictions_dir, pfile)
            try:
                with open(filepath, "r") as f:
                    pred_data = json.load(f)
            except Exception:
                continue

            pred_date = pred_data.get("date")
            if not pred_date or pred_date not in prices:
                continue

            base_price = prices[pred_date]
            horizons_data = pred_data.get("predictions", {}).get(instrument, {}).get("horizons", {})

            for h_str, h_info in horizons_data.items():
                try:
                    h_days = int(h_str.replace("d", ""))
                except Exception:
                    continue

                if pred_date in dates:
                    pred_idx = dates.index(pred_date)
                    target_idx = pred_idx + h_days
                    if target_idx < len(dates):
                        target_date = dates[target_idx]
                        actual_price = prices[target_date]
                        actual_return = (actual_price - base_price) / base_price

                        exp_return = h_info.get("expected_return", 0.0)
                        pred_direction = 1 if exp_return > 0 else 0
                        actual_direction = 1 if actual_return > 0 else 0
                        is_correct = (pred_direction == actual_direction)

                        evaluation_records.append({
                            "pred_date": pred_date,
                            "target_date": target_date,
                            "instrument": instrument,
                            "horizon_days": h_days,
                            "expected_return": exp_return,
                            "actual_return": round(float(actual_return), 4),
                            "predicted_direction": pred_direction,
                            "actual_direction": actual_direction,
                            "prediction_correct": is_correct,
                            "mae": abs(exp_return - actual_return)
                        })

        if not evaluation_records:
            return {"evaluated_count": 0}

        eval_df = pd.DataFrame(evaluation_records)
        acc_by_horizon = {}
        for h, group in eval_df.groupby("horizon_days"):
            acc_by_horizon[f"{h}d"] = {
                "count": len(group),
                "directional_accuracy": round(float(group["prediction_correct"].mean()), 4),
                "mae": round(float(group["mae"].mean()), 4)
            }

        perf_summary = {
            "total_evaluations": len(eval_df),
            "overall_accuracy": round(float(eval_df["prediction_correct"].mean()), 4),
            "by_horizon": acc_by_horizon,
            "last_updated": pd.Timestamp.now().isoformat()
        }

        # Save model monitoring metrics
        perf_path = os.path.join(self.models_dir, "model-performance.json")
        with open(perf_path, "w") as f:
            json.dump(perf_summary, f, indent=2)

        return perf_summary
