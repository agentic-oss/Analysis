import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

class ForecastingModelPipeline:
    """Baseline models and time-series walk-forward validation framework for return and direction prediction."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def train_and_evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        target_col: str = "future_return_5d",
        min_train_size: int = 60,
        step_size: int = 10
    ) -> Dict[str, Any]:
        """Perform time-series expanding-window walk-forward validation without shuffling."""
        valid_df = df.dropna(subset=[target_col] + self.feature_cols).sort_values("date").reset_index(drop=True)

        if len(valid_df) < min_train_size + step_size:
            return {"status": "insufficient_data", "samples": len(valid_df)}

        X = valid_df[self.feature_cols].fillna(0.0)
        y_reg = valid_df[target_col]
        y_clf = (y_reg > 0).astype(int)

        predictions_reg = []
        predictions_clf = []
        actuals_reg = []
        actuals_clf = []

        # Expanding window loop
        for start_idx in range(min_train_size, len(valid_df), step_size):
            X_train, y_train_reg, y_train_clf = X.iloc[:start_idx], y_reg.iloc[:start_idx], y_clf.iloc[:start_idx]
            end_test_idx = min(start_idx + step_size, len(valid_df))
            X_test, y_test_reg, y_test_clf = X.iloc[start_idx:end_test_idx], y_reg.iloc[start_idx:end_test_idx], y_clf.iloc[start_idx:end_test_idx]

            # Fit Regressor (Random Forest)
            model_reg = RandomForestRegressor(n_estimators=30, random_state=42)
            model_reg.fit(X_train, y_train_reg)
            pred_reg = model_reg.predict(X_test)

            # Fit Classifier (Logistic Regression)
            model_clf = LogisticRegression(max_iter=200)
            model_clf.fit(X_train, y_train_clf)
            pred_clf = model_clf.predict_proba(X_test)[:, 1]

            predictions_reg.extend(pred_reg)
            actuals_reg.extend(y_test_reg)
            predictions_clf.extend(pred_clf)
            actuals_clf.extend(y_test_clf)

        # Evaluation metrics
        preds_reg_arr = np.array(predictions_reg)
        acts_reg_arr = np.array(actuals_reg)
        preds_clf_arr = np.array(predictions_clf)
        acts_clf_arr = np.array(actuals_clf)

        mae = float(np.mean(np.abs(preds_reg_arr - acts_reg_arr)))
        rmse = float(np.sqrt(np.mean((preds_reg_arr - acts_reg_arr) ** 2)))
        dir_acc = float(np.mean((preds_reg_arr > 0) == (acts_reg_arr > 0)))

        return {
            "status": "success",
            "samples_evaluated": len(preds_reg_arr),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "directional_accuracy": round(dir_acc, 4)
        }
