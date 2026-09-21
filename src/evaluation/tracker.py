import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ForecastTracker:
    """Tracks historical prediction logs, evaluates predictions as outcomes realize, and records model performance metrics."""

    def __init__(self, predictions_db_path: str = "data/predictions/prediction_history.parquet"):
        self.predictions_db_path = predictions_db_path
        os.makedirs(os.path.dirname(self.predictions_db_path), exist_ok=True)

    def log_predictions(self, predictions: List[Dict[str, Any]]) -> pd.DataFrame:
        """Log new predictions generated on date T."""
        if not predictions:
            return pd.DataFrame()

        new_df = pd.DataFrame(predictions)
        if "actual_return" not in new_df.columns:
            new_df["actual_return"] = np.nan
        if "prediction_correct" not in new_df.columns:
            new_df["prediction_correct"] = np.nan

        if os.path.exists(self.predictions_db_path):
            try:
                existing_df = pd.read_parquet(self.predictions_db_path)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Deduplicate based on prediction_date, instrument, horizon
                combined = combined.drop_duplicates(subset=["prediction_date", "instrument", "horizon"], keep="last")
                df_to_save = combined
            except Exception as e:
                logger.warning(f"Failed to read prediction history: {e}")
                df_to_save = new_df
        else:
            df_to_save = new_df

        df_to_save.to_parquet(self.predictions_db_path, index=False)
        return df_to_save

    def evaluate_realized_outcomes(self, current_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Match realized actual returns on date T + horizon with logged forecasts."""
        if not os.path.exists(self.predictions_db_path):
            return pd.DataFrame()

        df = pd.read_parquet(self.predictions_db_path)
        if df.empty:
            return df

        updated = False
        for idx, row in df.iterrows():
            if not pd.isna(row["actual_return"]):
                continue  # already evaluated

            p_date_str = str(row["prediction_date"])
            symbol = str(row["instrument"])
            horizon = int(row["horizon"])

            if symbol not in current_data or current_data[symbol].empty:
                continue

            metal_df = current_data[symbol].sort_values("date").reset_index(drop=True)
            p_idx_list = metal_df.index[metal_df["date"] == p_date_str].tolist()
            if not p_idx_list:
                continue

            p_idx = p_idx_list[0]
            target_idx = p_idx + horizon

            if target_idx < len(metal_df):
                p_price = float(metal_df.loc[p_idx, "close"])
                actual_price = float(metal_df.loc[target_idx, "close"])
                actual_ret = (actual_price - p_price) / p_price

                pred_dir = 1 if row.get("predicted_direction", "UP") in ["UP", "BULLISH", 1] else -1
                actual_dir = 1 if actual_ret > 0 else -1

                df.loc[idx, "actual_return"] = round(actual_ret * 100.0, 2)
                df.loc[idx, "prediction_correct"] = 1 if (pred_dir == actual_dir) else 0
                updated = True

        if updated:
            df.to_parquet(self.predictions_db_path, index=False)

        return df

    def compute_forecast_performance_summary(self) -> Dict[str, Any]:
        """Generate comprehensive performance analytics across horizons, market regimes, and volatility regimes."""
        if not os.path.exists(self.predictions_db_path):
            return {"status": "no_prediction_history"}

        df = pd.read_parquet(self.predictions_db_path)
        evaluated = df.dropna(subset=["prediction_correct"]).copy()

        if evaluated.empty:
            return {
                "status": "pending_outcomes",
                "total_predictions_logged": len(df),
                "evaluated_predictions": 0
            }

        overall_accuracy = round(float(evaluated["prediction_correct"].mean()) * 100.0, 2)

        by_horizon = {}
        for h, sub in evaluated.groupby("horizon"):
            by_horizon[f"{h}d"] = {
                "total": len(sub),
                "accuracy_pct": round(float(sub["prediction_correct"].mean()) * 100.0, 2),
                "mae_return_pct": round(float((sub["actual_return"] - sub["predicted_return"]).abs().mean()), 2) if "predicted_return" in sub.columns else 0.0
            }

        by_regime = {}
        if "market_regime" in evaluated.columns:
            for reg, sub in evaluated.groupby("market_regime"):
                by_regime[str(reg)] = {
                    "total": len(sub),
                    "accuracy_pct": round(float(sub["prediction_correct"].mean()) * 100.0, 2)
                }

        summary = {
            "status": "success",
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_predictions_logged": len(df),
            "evaluated_predictions_count": len(evaluated),
            "overall_directional_accuracy_pct": overall_accuracy,
            "performance_by_horizon": by_horizon,
            "performance_by_market_regime": by_regime
        }

        # Save model monitoring report
        model_perf_path = "data/models/model-performance.json"
        os.makedirs(os.path.dirname(model_perf_path), exist_ok=True)
        with open(model_perf_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
