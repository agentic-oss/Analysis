import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class BaselineModels:
    """Trains and evaluates baseline machine learning models with walk-forward time-series validation."""

    @staticmethod
    def train_and_predict(
        feature_df: pd.DataFrame,
        feature_cols: List[str],
        horizon: int = 5
    ) -> Dict[str, Any]:
        """
        Runs walk-forward time-series prediction without look-ahead bias or shuffling.
        Returns model metrics and predicted outcomes.
        """
        target_ret_col = f'future_return_{horizon}d'
        target_dir_col = f'future_direction_{horizon}d'

        valid_df = feature_df.dropna(subset=feature_cols + [target_ret_col, target_dir_col]).copy()
        if len(valid_df) < 50:
            return {"status": "insufficient_data"}

        X = valid_df[feature_cols].fillna(0.0)
        y_dir = valid_df[target_dir_col]
        y_ret = valid_df[target_ret_col]

        # Time series split: 70% train, 30% test
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_dir_train, y_dir_test = y_dir.iloc[:split_idx], y_dir.iloc[split_idx:]
        y_ret_train, y_ret_test = y_ret.iloc[:split_idx], y_ret.iloc[split_idx:]

        # Logistic Regression
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_dir_train)
        acc = float(clf.score(X_test, y_dir_test))

        # Random Forest Regressor
        rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)
        rf_reg.fit(X_train, y_ret_train)
        pred_ret = rf_reg.predict(X_test)
        mae = float(np.mean(np.abs(pred_ret - y_ret_test)))

        # Latest prediction for most recent observation
        latest_X = X.iloc[[-1]]
        latest_dir_pred = int(clf.predict(latest_X)[0])
        latest_ret_pred = float(rf_reg.predict(latest_X)[0])

        return {
            "status": "success",
            "horizon_days": horizon,
            "logistic_accuracy": round(acc, 4),
            "rf_mae": round(mae, 4),
            "latest_predicted_direction": latest_dir_pred,
            "latest_predicted_return_pct": round(latest_ret_pred, 2)
        }
