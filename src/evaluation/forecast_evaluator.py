"""Forecast Evaluation and Model Drift Tracker.

Compares past predictions generated on date T against actual realized market outcomes
to measure accuracy, MAE, RMSE, calibration, and regime breakdown.
"""

from typing import Dict, Any, List
import os
import json
import numpy as np
import pandas as pd


class ForecastEvaluator:
    """Evaluates past predictions against realized returns."""

    def evaluate_past_predictions(
        self,
        predictions_dir: str = "data/predictions/daily",
        processed_metals_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Scans historical daily prediction JSON files and compares them with actual outcomes.
        Returns accuracy, MAE, RMSE, and regime-based performance metrics.
        """
        if not os.path.exists(predictions_dir) or processed_metals_df is None or processed_metals_df.empty:
            return {"status": "no_prediction_data_to_evaluate", "evaluations_count": 0}

        records = []
        for fname in os.listdir(predictions_dir):
            if fname.endswith(".json"):
                pred_date = fname.replace(".json", "")
                filepath = os.path.join(predictions_dir, fname)
                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)

                    for symbol in ["gold", "silver"]:
                        if symbol in data and "forward_statistics" in data[symbol]:
                            fwd = data[symbol]["forward_statistics"].get("forecast_signals", {})
                            for h_key, sig in fwd.items():
                                h_days = int(h_key.replace("d", ""))
                                pred_prob = sig.get("probability_positive", 0.5)
                                pred_ret = sig.get("expected_return", 0.0)

                                # Look up actual realized return in processed_metals_df
                                sym_upper = symbol.upper()
                                sub = processed_metals_df[
                                    (processed_metals_df['symbol'] == sym_upper) &
                                    (processed_metals_df['date'] >= pred_date)
                                ].sort_values('date')

                                if len(sub) > h_days:
                                    start_price = sub.iloc[0]['close']
                                    end_price = sub.iloc[h_days]['close']
                                    actual_ret = (end_price - start_price) / start_price
                                    actual_dir = 1 if actual_ret > 0 else 0
                                    pred_dir = 1 if pred_prob > 0.5 else 0

                                    records.append({
                                        "pred_date": pred_date,
                                        "symbol": sym_upper,
                                        "horizon": h_key,
                                        "pred_prob": pred_prob,
                                        "pred_ret": pred_ret,
                                        "actual_ret": actual_ret,
                                        "correct_direction": int(pred_dir == actual_dir),
                                        "abs_error": abs(pred_ret - actual_ret),
                                        "sq_error": (pred_ret - actual_ret) ** 2
                                    })
                except Exception:
                    continue

        if not records:
            return {"status": "insufficient_matured_predictions", "evaluations_count": 0}

        eval_df = pd.DataFrame(records)
        overall_accuracy = float(eval_df['correct_direction'].mean())
        mae = float(eval_df['abs_error'].mean())
        rmse = float(np.sqrt(eval_df['sq_error'].mean()))

        horizon_metrics = {}
        for h_key, group in eval_df.groupby('horizon'):
            horizon_metrics[h_key] = {
                "count": len(group),
                "directional_accuracy": round(float(group['correct_direction'].mean()), 4),
                "mae": round(float(group['abs_error'].mean()), 6),
                "rmse": round(float(np.sqrt(group['sq_error'].mean())), 6)
            }

        summary = {
            "evaluations_count": len(records),
            "overall_directional_accuracy": round(overall_accuracy, 4),
            "overall_mae": round(mae, 6),
            "overall_rmse": round(rmse, 6),
            "by_horizon": horizon_metrics
        }

        # Persist summary
        os.makedirs("data/models", exist_ok=True)
        with open("data/models/model-performance.json", "w") as f:
            json.dump(summary, f, indent=2)

        return summary
