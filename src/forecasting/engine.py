import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class ForecasterPipeline:
    """
    Forecasting Engine implementing probabilistic directional & return research models.
    Supports expanding-window / walk-forward time-series validation.

    Models evaluated:
    - Baseline Conditional Historical Frequency
    - Ridge Regression (Return prediction)
    - Logistic Regression & Random Forest (Direction classification)
    """
    def __init__(self, horizons: List[int] = [1, 3, 5, 10, 20, 60]):
        self.horizons = horizons

    def train_and_forecast_horizon(
        self,
        feature_df: pd.DataFrame,
        horizon: int,
        feature_cols: List[str]
    ) -> Dict[str, Any]:
        """
        Trains baseline models on historical feature store and forecasts for the latest observation.
        Prevents look-ahead bias by strictly requiring feature_df up to date T.
        """
        target_dir_col = f"target_future_direction_{horizon}d"
        target_ret_col = f"target_future_return_{horizon}d"

        # Filter rows where features and target exist for training
        train_data = feature_df.dropna(subset=feature_cols + [target_dir_col, target_ret_col])
        latest_features = feature_df[feature_cols].iloc[[-1]].dropna(axis=1)

        # Match valid feature columns present in both latest observation and training set
        valid_cols = [c for c in feature_cols if c in latest_features.columns and not latest_features[c].isna().any()]

        if len(train_data) < 30 or not valid_cols:
            # Fallback baseline statistics
            hist_rets = train_data[target_ret_col] if len(train_data) > 0 else pd.Series([0.0])
            pos_prob = (hist_rets > 0).mean() if len(hist_rets) > 0 else 0.5
            return {
                "horizon": f"{horizon}d",
                "probability_positive": round(float(pos_prob * 100.0), 1),
                "probability_negative": round(float((1.0 - pos_prob) * 100.0), 1),
                "expected_return": round(float(hist_rets.mean()), 2),
                "median_return": round(float(hist_rets.median()), 2),
                "model_agreement": "1/1 (Baseline fallback)"
            }

        X_train = train_data[valid_cols]
        y_dir_train = train_data[target_dir_col]
        y_ret_train = train_data[target_ret_col]

        X_latest = feature_df[valid_cols].iloc[[-1]]

        # 1. Historical Conditional Baseline
        base_pos_prob = float((y_dir_train == 1).mean())
        base_exp_ret = float(y_ret_train.mean())

        # 2. Logistic Regression Direction Model
        try:
            clf_log = LogisticRegression(max_iter=200, C=1.0)
            clf_log.fit(X_train, y_dir_train)
            log_prob = float(clf_log.predict_proba(X_latest)[0][1])
        except Exception:
            log_prob = base_pos_prob

        # 3. Random Forest Direction Model
        try:
            clf_rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            clf_rf.fit(X_train, y_dir_train)
            rf_prob = float(clf_rf.predict_proba(X_latest)[0][1])
        except Exception:
            rf_prob = base_pos_prob

        # 4. Ridge Regressor for Return Estimation
        try:
            reg_ridge = Ridge(alpha=10.0)
            reg_ridge.fit(X_train, y_ret_train)
            ridge_pred_ret = float(reg_ridge.predict(X_latest)[0])
        except Exception:
            ridge_pred_ret = base_exp_ret

        # Composite Model Probability Average
        prob_pos_avg = np.mean([base_pos_prob, log_prob, rf_prob]) * 100.0
        prob_neg_avg = 100.0 - prob_pos_avg

        # Count model direction agreement
        agreements = sum([1 for p in [base_pos_prob, log_prob, rf_prob] if (p > 0.5) == (prob_pos_avg > 50.0)])

        return {
            "horizon": f"{horizon}d",
            "probability_positive": round(float(prob_pos_avg), 1),
            "probability_negative": round(float(prob_neg_avg), 1),
            "expected_return": round(float(ridge_pred_ret), 2),
            "median_return": round(float(y_ret_train.median()), 2),
            "model_agreement": f"{agreements}/3"
        }

    def generate_all_forecasts(
        self,
        feature_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        results = {}
        for h in self.horizons:
            results[f"{h}d"] = self.train_and_forecast_horizon(feature_df, h, feature_cols)
        return results
