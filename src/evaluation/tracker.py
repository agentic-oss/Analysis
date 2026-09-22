"""
Forecast performance tracking and model monitoring module.
Maintains prediction logs and compares historical forecasts with actual outcomes over time.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Logs predictions with their creation date and calculates historical performance metrics:
    - Directional accuracy
    - MAE / RMSE
    - Accuracy by horizon and market regime
    """

    def __init__(self, predictions_dir: str = "data/predictions", models_dir: str = "data/models"):
        self.predictions_dir = predictions_dir
        self.models_dir = models_dir

    def log_prediction(self, prediction_data: Dict[str, Any]):
        """Logs a daily forecast entry to data/predictions/daily/YYYY-MM-DD.json"""
        date_str = prediction_data.get("prediction_date")
        if not date_str:
            return

        out_dir = os.path.join(self.predictions_dir, "daily")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"{date_str}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(prediction_data, f, indent=2)

    def evaluate_all_predictions(self, historical_prices_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Loads all past predictions and compares them against actual historical market outcomes.
        Outputs performance metrics report to data/models/model-performance.json.
        """
        daily_dir = os.path.join(self.predictions_dir, "daily")
        if not os.path.exists(daily_dir) or historical_prices_df.empty:
            return {"status": "no_predictions_to_evaluate"}

        price_map = historical_prices_df.set_index("date")["close"].to_dict()
        pred_files = [f for f in os.listdir(daily_dir) if f.endswith(".json")]

        eval_records = []

        for f_name in pred_files:
            try:
                with open(os.path.join(daily_dir, f_name), "r", encoding="utf-8") as f:
                    pred_data = json.load(f)

                p_date = pred_data.get("prediction_date")
                if p_date not in price_map:
                    continue

                base_price = price_map[p_date]
                signals = pred_data.get("horizon_signals", {})

                for h_str, sig in signals.items():
                    h_days = int(sig.get("horizon_days", 5))
                    # Find actual date h_days ahead in historical_prices_df
                    dates_list = historical_prices_df["date"].tolist()
                    if p_date in dates_list:
                        p_idx = dates_list.index(p_date)
                        target_idx = p_idx + h_days
                        if target_idx < len(dates_list):
                            target_date = dates_list[target_idx]
                            actual_price = price_map[target_date]
                            actual_ret = (actual_price - base_price) / base_price
                            predicted_direction = sig.get("research_bias")
                            actual_direction = "Bullish" if actual_ret > 0 else "Bearish"
                            is_correct = (predicted_direction == actual_direction)

                            eval_records.append({
                                "prediction_date": p_date,
                                "target_date": target_date,
                                "horizon_days": h_days,
                                "predicted_direction": predicted_direction,
                                "actual_direction": actual_direction,
                                "is_correct": 1 if is_correct else 0,
                                "predicted_return_pct": sig.get("historical_mean_return_pct", 0.0),
                                "actual_return_pct": round(actual_ret * 100, 2),
                                "error_pct": abs(sig.get("historical_mean_return_pct", 0.0) - (actual_ret * 100)),
                            })
            except Exception as e:
                logger.error(f"Error evaluating prediction file {f_name}: {str(e)}")

        if not eval_records:
            return {"status": "insufficient_matured_predictions"}

        df_eval = pd.DataFrame(eval_records)

        horizons_summary = {}
        for h in [1, 3, 5, 10, 20, 60]:
            h_df = df_eval[df_eval["horizon_days"] == h]
            if not h_df.empty:
                horizons_summary[f"{h}d"] = {
                    "count": len(h_df),
                    "directional_accuracy_pct": round(float(h_df["is_correct"].mean() * 100), 1),
                    "mae_pct": round(float(h_df["error_pct"].mean()), 2),
                    "rmse_pct": round(float(np.sqrt((h_df["error_pct"] ** 2).mean())), 2),
                }

        summary = {
            "total_evaluated_forecasts": len(df_eval),
            "overall_directional_accuracy_pct": round(float(df_eval["is_correct"].mean() * 100), 1),
            "by_horizon": horizons_summary,
        }

        os.makedirs(self.models_dir, exist_ok=True)
        perf_path = os.path.join(self.models_dir, "model-performance.json")
        with open(perf_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary
