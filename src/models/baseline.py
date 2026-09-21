import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class TimeSeriesModelEvaluator:
    """Walk-forward time-series machine learning trainer and evaluator for directional and return forecasting."""

    FEATURE_COLS = [
        "return_1d", "return_5d", "return_20d", "rsi_14", "stoch_rsi",
        "macd_hist", "atr_14", "volatility_20d", "dist_sma_20", "dist_sma_50",
        "dist_sma_200", "gold_silver_ratio_zscore"
    ]

    @staticmethod
    def train_and_evaluate_walk_forward(
        df: pd.DataFrame,
        horizon: int = 5,
        min_train_size: int = 60,
        step_size: int = 20
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            return {"status": "insufficient_data"}

        target_col = f"target_future_direction_{horizon}d"
        ret_col = f"target_future_return_{horizon}d"

        valid_cols = [c for c in TimeSeriesModelEvaluator.FEATURE_COLS if c in df.columns]
        if len(valid_cols) < 3 or target_col not in df.columns:
            return {"status": "missing_features_or_target"}

        clean_df = df.dropna(subset=valid_cols + [target_col, ret_col]).copy().sort_values("date").reset_index(drop=True)
        if len(clean_df) < min_train_size + horizon:
            min_train_size = max(20, len(clean_df) // 2)
            if len(clean_df) < min_train_size + horizon:
                return {"status": "insufficient_clean_data"}

        X = clean_df[valid_cols].values
        y_dir = clean_df[target_col].values
        y_ret = clean_df[ret_col].values

        rf_preds = []
        gb_preds = []
        lr_preds = []
        hist_prob_preds = []
        actual_dirs = []
        actual_rets = []

        # Walk-forward / expanding window loop
        for i in range(min_train_size, len(clean_df) - horizon, step_size):
            X_train, y_train_dir, y_train_ret = X[:i], y_dir[:i], y_ret[:i]
            X_test, y_test_dir, y_test_ret = X[i:i+step_size], y_dir[i:i+step_size], y_ret[i:i+step_size]

            if len(X_test) == 0 or len(np.unique(y_train_dir)) < 2:
                continue

            # 1. Historical conditional probability baseline
            hist_prob = float(np.mean(y_train_dir))
            hist_prob_preds.extend([1 if hist_prob >= 0.5 else 0] * len(X_test))

            # 2. Logistic Regression
            try:
                log_reg = LogisticRegression(max_iter=500)
                log_reg.fit(X_train, y_train_dir)
                lr_preds.extend(log_reg.predict(X_test))
            except Exception:
                lr_preds.extend([1 if hist_prob >= 0.5 else 0] * len(X_test))

            # 3. Random Forest
            try:
                rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
                rf.fit(X_train, y_train_dir)
                rf_preds.extend(rf.predict(X_test))
            except Exception:
                rf_preds.extend([1 if hist_prob >= 0.5 else 0] * len(X_test))

            # 4. Gradient Boosting
            try:
                gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
                gb.fit(X_train, y_train_dir)
                gb_preds.extend(gb.predict(X_test))
            except Exception:
                gb_preds.extend([1 if hist_prob >= 0.5 else 0] * len(X_test))

            actual_dirs.extend(y_test_dir)
            actual_rets.extend(y_test_ret)

        actual_dirs = np.array(actual_dirs)
        if len(actual_dirs) == 0:
            return {"status": "no_test_samples"}

        # Metrics calculation
        def calc_acc(preds, actuals):
            return round(float(np.mean(preds == actuals)) * 100.0, 2)

        metrics = {
            "status": "success",
            "horizon_trading_days": horizon,
            "test_sample_count": len(actual_dirs),
            "baseline_hist_prob_accuracy_pct": calc_acc(np.array(hist_prob_preds), actual_dirs),
            "logistic_regression_accuracy_pct": calc_acc(np.array(lr_preds), actual_dirs),
            "random_forest_accuracy_pct": calc_acc(np.array(rf_preds), actual_dirs),
            "gradient_boosting_accuracy_pct": calc_acc(np.array(gb_preds), actual_dirs)
        }

        return metrics
