import os
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class ForecastEvaluator:
    """Tracks historical predictions vs actual market outcomes over time, evaluating accuracy across horizons, market regimes, and volatility regimes."""

    def evaluate_historical_forecasts(self, predictions_history: list[dict], actual_prices_df: pd.DataFrame) -> dict:
        """
        Compares saved historical predictions with actual subsequent market prices.
        predictions_history: list of daily prediction dicts with keys (prediction_date, symbol, horizon, direction_prob, expected_return)
        actual_prices_df: DataFrame with 'date', 'symbol', 'close'
        """
        if not predictions_history or actual_prices_df.empty:
            return {
                "total_forecasts_evaluated": 0,
                "overall_directional_accuracy": 0.0,
                "horizon_breakdown": {},
                "model_monitoring": {"status": "INSUFFICIENT_DATA"}
            }

        records = []
        prices_map = actual_prices_df.set_index(["date", "symbol"])["close"].to_dict()

        for pred in predictions_history:
            p_date = pred.get("date")
            sym = pred.get("symbol")
            horizon = pred.get("horizon_days", 10)
            pred_prob = pred.get("positive_return_probability", 0.5)
            exp_ret = pred.get("expected_return_pct", 0.0)

            # Find actual price on prediction date and date + horizon
            try:
                dt_p = pd.to_datetime(p_date)
                future_dates = [d for (d, s) in prices_map.keys() if s == sym and pd.to_datetime(d) > dt_p]
                future_dates.sort()

                if len(future_dates) >= horizon:
                    target_dt = future_dates[horizon - 1]
                    p_start = prices_map.get((p_date, sym))
                    p_end = prices_map.get((target_dt, sym))

                    if p_start and p_end and p_start > 0:
                        actual_ret_pct = ((p_end - p_start) / p_start) * 100.0
                        actual_dir = 1 if actual_ret_pct > 0 else 0
                        predicted_dir = 1 if pred_prob >= 0.5 else 0
                        correct = 1 if predicted_dir == actual_dir else 0

                        records.append({
                            "prediction_date": p_date,
                            "symbol": sym,
                            "horizon_days": horizon,
                            "pred_prob": pred_prob,
                            "predicted_dir": predicted_dir,
                            "actual_dir": actual_dir,
                            "correct": correct,
                            "expected_return_pct": exp_ret,
                            "actual_return_pct": actual_ret_pct,
                            "abs_error": abs(exp_ret - actual_ret_pct)
                        })
            except Exception as e:
                logger.warning(f"Error evaluating prediction for {sym} on {p_date}: {e}")

        if not records:
            return {
                "total_forecasts_evaluated": 0,
                "overall_directional_accuracy": 0.0,
                "horizon_breakdown": {},
                "model_monitoring": {"status": "NO_MATURED_FORECASTS_YET"}
            }

        df_eval = pd.DataFrame(records)
        overall_acc = float(df_eval["correct"].mean() * 100.0)
        overall_mae = float(df_eval["abs_error"].mean())

        horizon_summary = {}
        for h, group in df_eval.groupby("horizon_days"):
            horizon_summary[f"{h}d"] = {
                "count": len(group),
                "directional_accuracy_pct": round(float(group["correct"].mean() * 100.0), 1),
                "mae": round(float(group["abs_error"].mean()), 2)
            }

        # Model degradation / drift check
        recent_acc = float(df_eval.tail(20)["correct"].mean() * 100.0) if len(df_eval) >= 20 else overall_acc
        status = "HEALTHY" if recent_acc >= 50.0 else "PERFORMANCE_DEGRADATION_DETECTED"

        return {
            "total_forecasts_evaluated": len(df_eval),
            "overall_directional_accuracy_pct": round(overall_acc, 1),
            "overall_mae": round(overall_mae, 2),
            "horizon_breakdown": horizon_summary,
            "model_monitoring": {
                "status": status,
                "recent_directional_accuracy_pct": round(recent_acc, 1),
                "baseline_comparison": "Baseline 50.0% uninformative target"
            }
        }
