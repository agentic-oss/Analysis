"""
Forecast Performance Tracking and Evaluation Module.
Tracks generated forecasts and evaluates accuracy as actual outcomes materialize over time.
"""
from typing import Dict, List, Any
import pandas as pd
import numpy as np


class ForecastTracker:
    """
    Records daily forecasts and evaluates directional accuracy, MAE, RMSE, and calibration over time.
    """

    @staticmethod
    def evaluate_past_predictions(
        predictions_history: List[Dict[str, Any]],
        actual_prices_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compares past predictions against actual realized market prices.
        """
        if not predictions_history or actual_prices_df.empty:
            return {"evaluated_count": 0, "message": "No predictions or price history available."}

        price_dict = actual_prices_df.set_index("date")["close"].to_dict()
        evaluations = []

        for pred in predictions_history:
            pred_date = pred.get("date")
            symbol = pred.get("symbol")
            horizons = pred.get("horizons", {})

            if pred_date not in price_dict:
                continue

            p_base = price_dict[pred_date]

            for h_key, h_data in horizons.items():
                try:
                    h_days = int(h_key.replace("d", ""))
                except ValueError:
                    continue

                # Find future date index
                dates_list = list(price_dict.keys())
                if pred_date not in dates_list:
                    continue

                curr_idx = dates_list.index(pred_date)
                fut_idx = curr_idx + h_days

                if fut_idx < len(dates_list):
                    fut_date = dates_list[fut_idx]
                    p_future = price_dict[fut_date]
                    actual_return = (p_future - p_base) / p_base

                    predicted_pos_prob = h_data.get("probability_positive_return_pct", 50.0)
                    predicted_direction = 1 if predicted_pos_prob > 50.0 else 0
                    actual_direction = 1 if actual_return > 0 else 0

                    is_correct = (predicted_direction == actual_direction)

                    evaluations.append({
                        "date": pred_date,
                        "future_date": fut_date,
                        "symbol": symbol,
                        "horizon": h_key,
                        "predicted_pos_prob": predicted_pos_prob,
                        "predicted_direction": predicted_direction,
                        "actual_return_pct": actual_return * 100.0,
                        "actual_direction": actual_direction,
                        "is_correct": is_correct,
                        "error": abs(actual_return * 100.0 - h_data.get("expected_median_return_pct", 0.0)),
                    })

        if not evaluations:
            return {"evaluated_count": 0, "message": "Pending future horizon maturity."}

        eval_df = pd.DataFrame(evaluations)
        acc = float(eval_df["is_correct"].mean() * 100.0)
        mae = float(eval_df["error"].mean())

        return {
            "total_evaluated_forecasts": len(eval_df),
            "directional_accuracy_pct": round(acc, 2),
            "mean_absolute_error": round(mae, 2),
            "sample_evaluations": evaluations[:10],
        }
