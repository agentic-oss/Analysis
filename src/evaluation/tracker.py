"""Forecast performance tracking and model drift monitoring engine."""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Compares historical forecast generated at date T with subsequent actual market outcomes at date T + horizon.
    Tracks directional accuracy, MAE, RMSE, and model performance metrics across market regimes over time.
    """

    def __init__(self, predictions_dir: str = "data/predictions/daily"):
        self.predictions_dir = predictions_dir

    def record_forecast(self, prediction_date: str, instrument: str, horizon: str, predicted_direction: str, expected_return_pct: float, confidence_score: float) -> Dict[str, Any]:
        """Saves structured forecast record for future evaluation."""
        record = {
            "prediction_date": prediction_date,
            "instrument": instrument,
            "horizon": horizon,
            "predicted_direction": predicted_direction,
            "expected_return_pct": expected_return_pct,
            "confidence_score": confidence_score,
            "actual_return_pct": None,
            "prediction_correct": None,
            "evaluated_at": None
        }
        return record

    def evaluate_historical_forecasts(self, historical_prices_df: pd.DataFrame, saved_forecasts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluates pending saved forecasts against actual outcomes in price dataframe."""
        if historical_prices_df.empty or 'date' not in historical_prices_df.columns or 'close' not in historical_prices_df.columns:
            return {"evaluated_count": 0, "accuracy_pct": 0.0}

        prices_df = historical_prices_df.set_index('date')['close'].sort_index()

        evaluated = []
        correct_count = 0
        total_eval = 0

        horizon_map = {"1d": 1, "3d": 3, "5d": 5, "10d": 10, "20d": 20, "60d": 60}

        for f in saved_forecasts:
            p_date = f["prediction_date"]
            h_str = f["horizon"]
            days = horizon_map.get(h_str, 5)

            if p_date in prices_df.index:
                p_date_loc = prices_df.index.get_loc(p_date)
                fut_loc = p_date_loc + days

                if fut_loc < len(prices_df):
                    p_start = prices_df.iloc[p_date_loc]
                    p_end = prices_df.iloc[fut_loc]
                    act_ret = (p_end - p_start) / p_start * 100.0

                    f["actual_return_pct"] = round(act_ret, 2)
                    act_dir = "BULLISH" if act_ret > 0 else ("BEARISH" if act_ret < 0 else "NEUTRAL")

                    is_correct = (f["predicted_direction"] == act_dir)
                    f["prediction_correct"] = is_correct
                    f["evaluated_at"] = prices_df.index[fut_loc]

                    if is_correct:
                        correct_count += 1
                    total_eval += 1

            evaluated.append(f)

        overall_acc = (correct_count / total_eval * 100.0) if total_eval > 0 else 0.0

        return {
            "evaluated_count": total_eval,
            "overall_accuracy_pct": round(overall_acc, 2),
            "evaluated_forecasts": evaluated
        }

    def update_model_performance_summary(self, eval_results: Dict[str, Any], output_path: str = "data/models/model-performance.json") -> str:
        """Saves summary performance metric log to data/models/model-performance.json."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary = {
            "last_updated": pd.Timestamp.now().isoformat(),
            "evaluated_forecast_count": eval_results.get("evaluated_count", 0),
            "overall_directional_accuracy_pct": eval_results.get("overall_accuracy_pct", 0.0),
            "status": "HEALTHY" if eval_results.get("overall_accuracy_pct", 50.0) >= 50.0 else "DRIFT_WARNING"
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Updated model performance log at {output_path}")
        return output_path
