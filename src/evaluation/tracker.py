import os
import json
import logging
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ForecastPerformanceTracker:
    """
    Logs predictions generated on date T and evaluates actual outcomes over time.
    Maintains a continuously growing evaluation database.
    """

    def __init__(self, predictions_dir: str = "data/predictions", eval_file: str = "data/models/forecast_evaluation.json"):
        self.predictions_dir = predictions_dir
        self.eval_file = eval_file
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.eval_file), exist_ok=True)

    def log_daily_predictions(self, date_str: str, predictions_data: Dict[str, Any]) -> str:
        """Saves daily forecasts with date T timestamp."""
        p_path = os.path.join(self.predictions_dir, f"{date_str}.json")
        with open(p_path, "w") as f:
            json.dump(predictions_data, f, indent=2)
        return p_path

    def evaluate_past_predictions(self, historical_prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Scans all past prediction JSON files, compares predicted direction/return against
        actual market outcomes from historical_prices_df, and updates metrics.
        """
        if historical_prices_df.empty:
            return {"status": "no_price_data"}

        price_lookup = historical_prices_df.set_index("date")["close"].to_dict()
        pred_files = [f for f in os.listdir(self.predictions_dir) if f.endswith(".json")]

        eval_records = []

        for p_file in pred_files:
            pred_date = p_file.replace(".json", "")
            if pred_date not in price_lookup:
                continue

            start_price = price_lookup[pred_date]

            with open(os.path.join(self.predictions_dir, p_file), "r") as f:
                data = json.load(f)

            for inst, forecasts in data.items():
                if not isinstance(forecasts, dict) or "horizons" not in forecasts:
                    continue

                for h_str, f_info in forecasts["horizons"].items():
                    h_days = int(h_str.replace("d", ""))
                    # Find date T + h trading days
                    dates_sorted = sorted(price_lookup.keys())
                    if pred_date in dates_sorted:
                        idx = dates_sorted.index(pred_date)
                        if idx + h_days < len(dates_sorted):
                            future_date = dates_sorted[idx + h_days]
                            actual_price = price_lookup[future_date]
                            actual_ret = (actual_price - start_price) / start_price

                            pred_dir = f_info.get("predicted_direction", "Bullish")
                            actual_dir = "Bullish" if actual_ret > 0 else "Bearish"
                            is_correct = (pred_dir == actual_dir)

                            eval_records.append({
                                "prediction_date": pred_date,
                                "target_date": future_date,
                                "instrument": inst,
                                "horizon": h_str,
                                "predicted_direction": pred_dir,
                                "actual_return_pct": round(actual_ret * 100.0, 2),
                                "prediction_correct": is_correct
                            })

        if not eval_records:
            return {"status": "no_evaluable_predictions", "total_evaluated": 0}

        eval_df = pd.DataFrame(eval_records)
        overall_acc = float(eval_df["prediction_correct"].mean() * 100.0)

        summary = {
            "total_evaluated": len(eval_df),
            "overall_accuracy_pct": round(overall_acc, 1),
            "accuracy_by_horizon": eval_df.groupby("horizon")["prediction_correct"].mean().round(3).to_dict(),
            "accuracy_by_instrument": eval_df.groupby("instrument")["prediction_correct"].mean().round(3).to_dict()
        }

        with open(self.eval_file, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
