import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ForecastTracker:
    """
    Stores forecasts with creation timestamps and evaluates historical predictions
    against realized market outcomes over time to maintain data/models/model-performance.json.
    """

    def __init__(self, predictions_dir: str = "data/predictions/daily", performance_file: str = "data/models/model-performance.json"):
        self.predictions_dir = predictions_dir
        self.performance_file = performance_file

    def record_forecast(
        self,
        date_str: str,
        instrument: str,
        forecast_data: Dict[str, Any],
        confidence_data: Dict[str, Any],
        current_price: float,
    ):
        """Save daily prediction record to data/predictions/daily/YYYY-MM-DD.json."""
        os.makedirs(self.predictions_dir, exist_ok=True)
        filepath = os.path.join(self.predictions_dir, f"{date_str}_{instrument}.json")

        record = {
            "prediction_date": date_str,
            "instrument": instrument,
            "entry_price": current_price,
            "forecasts": forecast_data.get("forecasts_by_horizon", {}),
            "research_bias": confidence_data.get("research_bias"),
            "confidence_score": confidence_data.get("confidence_score"),
            "created_at": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        logger.info(f"Recorded forecast for {instrument} on {date_str}")

    def evaluate_historical_forecasts(self, historical_price_df: pd.DataFrame, instrument: str) -> Dict[str, Any]:
        """
        Scan saved prediction files, compare predictions with realized historical outcomes,
        and write summary metrics to data/models/model-performance.json.
        """
        if not os.path.exists(self.predictions_dir):
            return {}

        price_map = dict(zip(pd.to_datetime(historical_price_df["Date"]).dt.strftime("%Y-%m-%d"), historical_price_df["Close"]))
        dates_sorted = sorted(price_map.keys())

        evaluation_records = []

        for fname in os.listdir(self.predictions_dir):
            if not fname.endswith(".json") or instrument not in fname:
                continue

            fpath = os.path.join(self.predictions_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                pred = json.load(f)

            p_date = pred.get("prediction_date")
            entry_p = pred.get("entry_price")
            if not p_date or not entry_p or p_date not in price_map:
                continue

            p_idx = dates_sorted.index(p_date) if p_date in dates_sorted else -1
            if p_idx == -1:
                continue

            for h_str, f_data in pred.get("forecasts", {}).items():
                h = f_data.get("horizon_trading_days", 5)
                if p_idx + h < len(dates_sorted):
                    future_date = dates_sorted[p_idx + h]
                    future_price = price_map[future_date]
                    actual_return_pct = ((future_price - entry_p) / entry_p) * 100.0
                    predicted_exp_return = f_data.get("expected_mean_return_pct", 0.0)

                    pred_dir = 1 if predicted_exp_return > 0 else -1
                    actual_dir = 1 if actual_return_pct > 0 else -1

                    evaluation_records.append({
                        "prediction_date": p_date,
                        "instrument": instrument,
                        "horizon": h,
                        "predicted_return_pct": predicted_exp_return,
                        "actual_return_pct": round(actual_return_pct, 2),
                        "directional_correct": int(pred_dir == actual_dir),
                        "error_abs": round(abs(actual_return_pct - predicted_exp_return), 2),
                    })

        if not evaluation_records:
            performance_summary = {
                "last_evaluated": datetime.now().strftime("%Y-%m-%d"),
                "instrument": instrument,
                "total_evaluated_forecasts": 0,
                "overall_directional_accuracy_pct": 50.0,
                "mae": 0.0,
                "note": "No realized forecasts matured yet.",
            }
        else:
            df_eval = pd.DataFrame(evaluation_records)
            accuracy = (df_eval["directional_correct"].mean()) * 100.0
            mae = df_eval["error_abs"].mean()

            performance_summary = {
                "last_evaluated": datetime.now().strftime("%Y-%m-%d"),
                "instrument": instrument,
                "total_evaluated_forecasts": len(df_eval),
                "overall_directional_accuracy_pct": round(float(accuracy), 1),
                "mae": round(float(mae), 2),
                "by_horizon": df_eval.groupby("horizon")["directional_correct"].mean().to_dict(),
            }

        os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
        with open(self.performance_file, "w", encoding="utf-8") as f:
            json.dump(performance_summary, f, indent=2)

        return performance_summary
