import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class MetalsForecastingModels:
    """Statistical & ML baseline forecasting models evaluated via walk-forward validation."""

    def __init__(self, target_horizon: int = 5):
        self.target_horizon = target_horizon

    def prepare_datasets(
        self, feature_df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, List[str]]:
        """Extracts X (features), y_dir (binary direction), y_ret (continuous return), and dates."""
        valid_cols = [c for c in feature_cols if c in feature_df.columns]
        target_dir_col = f"future_direction_{self.target_horizon}d"
        target_ret_col = f"future_return_{self.target_horizon}d"

        # Drop rows where target is missing (last rows where future isn't known yet)
        clean = feature_df.dropna(subset=[target_dir_col, target_ret_col] + valid_cols).copy()

        X = clean[valid_cols].values
        y_dir = clean[target_dir_col].values
        y_ret = clean[target_ret_col].values
        dates = clean["date"]

        return X, y_dir, y_ret, dates, valid_cols

    def train_walk_forward(
        self, feature_df: pd.DataFrame, feature_cols: List[str], train_window: int = 252, test_window: int = 20
    ) -> Dict[str, Any]:
        """Executes expanding-window / walk-forward time-series validation."""
        X, y_dir, y_ret, dates, valid_cols = self.prepare_datasets(feature_df, feature_cols)
        if len(X) < train_window + test_window:
            return {"status": "insufficient_data", "metrics": {}}

        preds_dir = []
        actuals_dir = []
        preds_ret = []
        actuals_ret = []

        model_clf = RandomForestClassifier(n_estimators=50, random_state=42)
        model_reg = Ridge(alpha=1.0)

        for i in range(train_window, len(X) - test_window + 1, test_window):
            X_train, y_train_dir, y_train_ret = X[:i], y_dir[:i], y_ret[:i]
            X_test, y_test_dir, y_test_ret = X[i:i + test_window], y_dir[i:i + test_window], y_ret[i:i + test_window]

            model_clf.fit(X_train, y_train_dir)
            model_reg.fit(X_train, y_train_ret)

            p_dir = model_clf.predict(X_test)
            p_ret = model_reg.predict(X_test)

            preds_dir.extend(p_dir)
            actuals_dir.extend(y_test_dir)
            preds_ret.extend(p_ret)
            actuals_ret.extend(y_test_ret)

        if not actuals_dir:
            return {"status": "no_predictions", "metrics": {}}

        acc = accuracy_score(actuals_dir, preds_dir)
        prec = precision_score(actuals_dir, preds_dir, zero_division=0)
        rec = recall_score(actuals_dir, preds_dir, zero_division=0)
        f1 = f1_score(actuals_dir, preds_dir, zero_division=0)
        mae = mean_absolute_error(actuals_ret, preds_ret)
        rmse = np.sqrt(mean_squared_error(actuals_ret, preds_ret))

        return {
            "status": "success",
            "samples": len(actuals_dir),
            "metrics": {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1": round(float(f1), 4),
                "mae": round(float(mae), 6),
                "rmse": round(float(rmse), 6),
            },
        }

    def predict_current(
        self, feature_df: pd.DataFrame, feature_cols: List[str]
    ) -> Dict[str, Any]:
        """Fits model on all past historical data and predicts current signal."""
        X, y_dir, y_ret, dates, valid_cols = self.prepare_datasets(feature_df, feature_cols)
        if len(X) < 50:
            return {"direction_pred": 1, "return_pred": 0.0, "prob_positive": 0.5}

        current_row = feature_df[valid_cols].iloc[-1:].fillna(0.0).values

        clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
        clf.fit(X, y_dir)

        prob_pos = float(clf.predict_proba(current_row)[0][1])
        pred_dir = int(clf.predict(current_row)[0])

        reg = Ridge(alpha=1.0)
        reg.fit(X, y_ret)
        pred_ret = float(reg.predict(current_row)[0])

        return {
            "direction_pred": pred_dir,
            "return_pred": round(pred_ret, 6),
            "prob_positive": round(prob_pos, 4),
        }
