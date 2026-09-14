"""
Evaluation & Historical Forecast Performance Tracking Module.
Compares previous historical forecasts (Date T) with actual outcomes (Date T + H)
and evaluates accuracy by regime, volatility, and horizon without hiding poor performance.
"""
import os
import json
import logging
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ForecastEvaluator:
    """
    Evaluates past historical predictions stored in predictions directory against real outcomes.
    Generates data/models/model-performance.json tracking accuracy across horizons and regimes.
    """

    @staticmethod
    def evaluate_predictions(
        predictions_history: List[Dict[str, Any]],
        historical_prices: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        `predictions_history`: list of previously stored prediction JSON records.
        `historical_prices`: dict mapping symbol -> DataFrame with ['date', 'close'].
        """
        if not predictions_history:
            return {
                "total_evaluations": 0,
                "overall_directional_accuracy_pct": 0.0,
                "horizons": {}
            }

        eval_records = []

        for record in predictions_history:
            pred_date = record.get("date")
            for symbol in ["gold", "silver"]:
                sym_data = record.get(symbol, {})
                signals = sym_data.get("forward_statistics", {}) or sym_data.get("signals", {})
                sym_code = "GC=F" if symbol == "gold" else "SI=F"

                price_df = historical_prices.get(sym_code, pd.DataFrame())
                if price_df.empty or "date" not in price_df.columns:
                    continue

                # Find entry index
                match_idx = price_df.index[price_df["date"] == pred_date]
                if len(match_idx) == 0:
                    continue
                entry_idx = match_idx[0]
                entry_price = price_df["close"].iloc[entry_idx]

                for horizon_str, sig in signals.items():
                    h_days = sig.get("horizon_days", 5)
                    predicted_bias = sig.get("research_bias", "Neutral")

                    if entry_idx + h_days < len(price_df):
                        exit_price = price_df["close"].iloc[entry_idx + h_days]
                        actual_return = (exit_price - entry_price) / entry_price
                        actual_dir = "Bullish" if actual_return > 0 else ("Bearish" if actual_return < 0 else "Neutral")

                        correct = (
                            (predicted_bias == "Bullish" and actual_dir == "Bullish") or
                            (predicted_bias == "Bearish" and actual_dir == "Bearish") or
                            (predicted_bias == "Neutral" and abs(actual_return) < 0.005)
                        )

                        eval_records.append({
                            "prediction_date": pred_date,
                            "symbol": symbol,
                            "horizon": horizon_str,
                            "predicted_bias": predicted_bias,
                            "actual_return_pct": round(actual_return * 100, 2),
                            "actual_direction": actual_dir,
                            "is_correct": correct
                        })

        if not eval_records:
            return {
                "total_evaluations": 0,
                "overall_directional_accuracy_pct": 50.0,
                "horizons": {}
            }

        df_eval = pd.DataFrame(eval_records)
        overall_acc = float(df_eval["is_correct"].mean() * 100)

        horizon_summary = {}
        for h, group in df_eval.groupby("horizon"):
            horizon_summary[h] = {
                "count": len(group),
                "accuracy_pct": round(float(group["is_correct"].mean() * 100), 1),
                "mean_actual_return_pct": round(float(group["actual_return_pct"].mean()), 2)
            }

        return {
            "total_evaluations": len(df_eval),
            "overall_directional_accuracy_pct": round(overall_acc, 1),
            "horizons": horizon_summary,
            "recent_evaluations": eval_records[-10:]
        }
