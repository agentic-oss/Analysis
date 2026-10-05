"""Baseline statistical and ML forecasting models using walk-forward time-series validation."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier


class MetalsForecastingModels:
    """Trains and evaluates baseline models (Logistic Regression, Ridge, Random Forest)."""

    def __init__(self):
        self.feature_cols = [
            'rsi_14', 'macd', 'atr_14', 'volatility_20d',
            'distance_sma_20', 'distance_sma_50', 'distance_sma_200',
            'return_1d', 'return_5d', 'return_20d'
        ]

    def train_and_predict(
        self,
        features_df: pd.DataFrame,
        current_date: str,
        symbol: str = "GOLD"
    ) -> Dict[str, Any]:
        """
        Trains models on historical data up to current_date (walk-forward split)
        and predicts direction and forward return for current_date.
        """
        inst_df = features_df[features_df['symbol'] == symbol].sort_values('date').copy()
        valid_cols = [c for c in self.feature_cols if c in inst_df.columns]

        if len(inst_df) < 30 or not valid_cols:
            return self._fallback_predictions()

        train_data = inst_df[inst_df['date'] < current_date].dropna(subset=valid_cols)
        if len(train_data) < 20:
            return self._fallback_predictions()

        curr_row = inst_df[inst_df['date'] == current_date]
        if curr_row.empty:
            curr_row = inst_df.iloc[-1:]

        X_curr = curr_row[valid_cols].fillna(0).values

        predictions = {}
        for h in [1, 3, 5, 10, 20, 60]:
            h_key = f"{h}d"
            target_col = f"future_direction_{h}d"
            ret_col = f"future_return_{h}d"

            if target_col in train_data.columns and train_data[target_col].nunique() > 1:
                sub_train = train_data.dropna(subset=[target_col, ret_col])
                X_train = sub_train[valid_cols].fillna(0).values
                y_class = sub_train[target_col].values
                y_reg = sub_train[ret_col].values

                # Classification model (Logistic Regression)
                clf = LogisticRegression(max_iter=200)
                clf.fit(X_train, y_class)
                prob_up = float(clf.predict_proba(X_curr)[0][1]) if len(clf.classes_) > 1 else 0.5

                # Regression model (Ridge)
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_reg)
                exp_ret = float(reg.predict(X_curr)[0])

                predictions[h_key] = {
                    "direction_prob": round(prob_up, 4),
                    "predicted_return": round(exp_ret, 6)
                }
            else:
                predictions[h_key] = {"direction_prob": 0.5, "predicted_return": 0.0}

        return predictions

    def _fallback_predictions(self) -> Dict[str, Any]:
        res = {}
        for h in [1, 3, 5, 10, 20, 60]:
            res[f"{h}d"] = {"direction_prob": 0.5, "predicted_return": 0.0}
        return res
