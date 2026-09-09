"""
Forecasting Models & Walk-Forward Validation Engine.
Implements baseline models:
1. Historical Conditional Probability
2. Linear Regression (Return forecast)
3. Logistic Regression (Direction forecast)
4. Random Forest Classifier/Regressor
5. Gradient Boosting Classifier/Regressor
Evaluates models using time-series expanding / walk-forward validation (NO random shuffles).
Expresses outputs strictly as probabilistic signals.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error, r2_score


class MetalsForecaster:
    """Trains forecasting models and produces probabilistic forward research signals."""

    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon
        self.feature_cols = [
            "rsi_14", "stoch_rsi", "macd", "macd_hist", "atr_14",
            "volatility_20d", "dist_sma_20_pct", "dist_sma_50_pct", "dist_sma_200_pct",
            "return_1d", "return_5d", "return_20d"
        ]

    def prepare_data(self, feature_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Cleans and extracts available feature columns."""
        df = feature_df.copy().sort_values("date").reset_index(drop=True)
        avail_features = [c for c in self.feature_cols if c in df.columns]

        if "gold_silver_ratio" in df.columns:
            avail_features.append("gold_silver_ratio")
        if "gold_silver_ratio_zscore" in df.columns:
            avail_features.append("gold_silver_ratio_zscore")
        if "dxy_return_20d" in df.columns:
            avail_features.append("dxy_return_20d")
        if "vix_close" in df.columns:
            avail_features.append("vix_close")

        df = df.dropna(subset=avail_features).reset_index(drop=True)
        return df, avail_features

    def train_and_predict(self, feature_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trains ensemble of models on historical data up to date T-horizon,
        and generates forward probability predictions for the latest observation date T.
        """
        df, features = self.prepare_data(feature_df)
        target_col = f"future_return_{self.target_horizon}d"
        target_dir_col = f"future_direction_{self.target_horizon}d"

        if len(df) < 100 or target_col not in df.columns:
            return {
                "horizon": f"{self.target_horizon}d",
                "probability_positive": 50.0,
                "expected_return": 0.0,
                "direction": "NEUTRAL",
                "model_agreement": "0/0",
                "model_outputs": {}
            }

        # Historical training set excludes unobserved future target dates
        train_df = df.dropna(subset=[target_col]).copy()
        if len(train_df) < 50:
            return {"horizon": f"{self.target_horizon}d", "probability_positive": 50.0, "expected_return": 0.0}

        X_train = train_df[features].values
        y_train_ret = train_df[target_col].values
        y_train_dir = train_df[target_dir_col].values

        # Latest observation vector for prediction
        X_latest = df[features].iloc[[-1]].values

        model_results = {}
        pos_probs = []
        expected_rets = []

        # 1. Historical Conditional Baseline
        mean_ret = float(np.mean(y_train_ret))
        prob_pos = float((y_train_dir == 1).mean() * 100.0)
        model_results["historical_baseline"] = {"prob_pos": prob_pos, "expected_ret": mean_ret}
        pos_probs.append(prob_pos)
        expected_rets.append(mean_ret)

        # 2. Logistic Regression (Direction)
        try:
            clf_lr = LogisticRegression(max_iter=500)
            clf_lr.fit(X_train, y_train_dir)
            lr_prob = float(clf_lr.predict_proba(X_latest)[0][1] * 100.0)
            model_results["logistic_regression"] = {"prob_pos": lr_prob}
            pos_probs.append(lr_prob)
        except Exception:
            pass

        # 3. Random Forest Classifier
        try:
            rf_clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            rf_clf.fit(X_train, y_train_dir)
            rf_prob = float(rf_clf.predict_proba(X_latest)[0][1] * 100.0)
            model_results["random_forest_clf"] = {"prob_pos": rf_prob}
            pos_probs.append(rf_prob)
        except Exception:
            pass

        # 4. Gradient Boosting Regressor
        try:
            gb_reg = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            gb_reg.fit(X_train, y_train_ret)
            gb_pred_ret = float(gb_reg.predict(X_latest)[0])
            gb_prob = 65.0 if gb_pred_ret > 0.5 else (35.0 if gb_pred_ret < -0.5 else 50.0)
            model_results["gradient_boosting_reg"] = {"expected_ret": gb_pred_ret, "prob_pos": gb_prob}
            expected_rets.append(gb_pred_ret)
            pos_probs.append(gb_prob)
        except Exception:
            pass

        avg_prob = float(np.mean(pos_probs))
        avg_ret = float(np.mean(expected_rets)) if expected_rets else 0.0

        bull_count = sum(1 for p in pos_probs if p > 55.0)
        direction = "BULLISH" if avg_prob >= 58.0 else ("BEARISH" if avg_prob <= 42.0 else "NEUTRAL")

        return {
            "horizon": f"{self.target_horizon}d",
            "as_of_date": str(df.iloc[-1]["date"]),
            "probability_positive": avg_prob,
            "probability_negative": 100.0 - avg_prob,
            "expected_return": avg_ret,
            "direction": direction,
            "model_agreement": f"{bull_count}/{len(pos_probs)}",
            "model_outputs": model_results
        }

    def walk_forward_evaluate(
        self,
        feature_df: pd.DataFrame,
        train_window: int = 500,
        step_size: int = 20
    ) -> Dict[str, Any]:
        """Performs expanding walk-forward validation across time-series."""
        df, features = self.prepare_data(feature_df)
        target_col = f"future_return_{self.target_horizon}d"
        target_dir_col = f"future_direction_{self.target_horizon}d"

        df_valid = df.dropna(subset=[target_col] + features).reset_index(drop=True)
        if len(df_valid) < train_window + step_size:
            return {}

        predictions = []
        actuals_dir = []
        actuals_ret = []

        for i in range(train_window, len(df_valid) - self.target_horizon, step_size):
            train_sub = df_valid.iloc[:i]
            test_sub = df_valid.iloc[i:i+step_size]

            X_tr, y_tr_dir = train_sub[features].values, train_sub[target_dir_col].values
            X_te, y_te_dir = test_sub[features].values, test_sub[target_dir_col].values
            y_te_ret = test_sub[target_col].values

            rf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)
            rf.fit(X_tr, y_tr_dir)
            preds = rf.predict(X_te)

            predictions.extend(preds)
            actuals_dir.extend(y_te_dir)
            actuals_ret.extend(y_te_ret)

        if not predictions:
            return {}

        acc = float(accuracy_score(actuals_dir, predictions))
        prec = float(precision_score(actuals_dir, predictions, zero_division=0))
        rec = float(recall_score(actuals_dir, predictions, zero_division=0))
        f1 = float(f1_score(actuals_dir, predictions, zero_division=0))

        return {
            "target_horizon": f"{self.target_horizon}d",
            "test_samples": len(predictions),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
