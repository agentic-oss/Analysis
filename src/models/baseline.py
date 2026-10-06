import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)

class ForecastingModels:
    """Baseline models for direction, forward return, and volatility prediction using expanding/walk-forward time series validation."""

    def __init__(self):
        self.regressors = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=50, random_state=42)
        }
        self.classifiers = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }

    def train_and_predict(self, feature_df: pd.DataFrame, target_date: str, horizon: int = 5) -> Dict[str, Any]:
        if feature_df.empty or "date" not in feature_df.columns:
            return {"horizon": horizon, "predictions": {}}

        past_df = feature_df[feature_df["date"] < target_date].dropna(subset=[f"future_return_{horizon}d"]).copy()
        target_row = feature_df[feature_df["date"] == target_date]
        if target_row.empty:
            target_row = feature_df[feature_df["date"] <= target_date].tail(1)

        feature_cols = [c for c in ["rsi_14", "macd_hist", "dist_sma_20", "dist_sma_200", "volatility_20d", "return_5d", "return_20d"] if c in past_df.columns]

        if len(past_df) < 30 or target_row.empty or not feature_cols:
            rsi_val = target_row.iloc[-1].get("rsi_14", 50) if not target_row.empty else 50
            return {
                "horizon": horizon,
                "model_agreement": "0/0",
                "predicted_direction": "Bullish" if rsi_val > 50 else "Bearish",
                "predicted_return": 0.005,
                "predictions": {}
            }

        X_train = past_df[feature_cols].fillna(0)
        y_train_reg = past_df[f"future_return_{horizon}d"]
        y_train_cls = past_df[f"future_dir_{horizon}d"]

        X_test = target_row[feature_cols].fillna(0)

        preds = {}
        bullish_votes = 0
        total_votes = 0

        # Regression models
        for name, model in self.regressors.items():
            try:
                model.fit(X_train, y_train_reg)
                pred_val = float(model.predict(X_test)[0])
                preds[f"{name}_return"] = pred_val
            except Exception as e:
                logger.warning(f"Error training regressor {name}: {e}")

        # Classification models
        for name, model in self.classifiers.items():
            try:
                model.fit(X_train, y_train_cls)
                pred_cls = int(model.predict(X_test)[0])
                preds[f"{name}_direction"] = "Bullish" if pred_cls == 1 else "Bearish"
                total_votes += 1
                if pred_cls == 1:
                    bullish_votes += 1
            except Exception as e:
                logger.warning(f"Error training classifier {name}: {e}")

        overall_dir = "Bullish" if bullish_votes >= (total_votes / 2.0) else "Bearish"
        pred_ret = preds.get("random_forest_return", preds.get("linear_regression_return", 0.0))

        return {
            "horizon": horizon,
            "target_date": target_date,
            "model_agreement": f"{bullish_votes}/{total_votes}",
            "predicted_direction": overall_dir,
            "predicted_return": pred_ret,
            "predictions": preds
        }
