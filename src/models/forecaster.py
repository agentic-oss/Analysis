import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class TimeSeriesModelEvaluator:
    """
    Evaluates baseline ML and statistical models using expanding-window / walk-forward validation.
    Strictly avoids random shuffling to prevent temporal data leakage.
    """

    def __init__(
        self,
        features: List[str],
        target_horizon: int = 5,
        min_train_size: int = 120,
        step_size: int = 20
    ):
        self.features = features
        self.target_horizon = target_horizon
        self.min_train_size = min_train_size
        self.step_size = step_size

    def walk_forward_validation(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Runs expanding window walk-forward validation for direction and return regression."""

        target_col_ret = f"future_return_{self.target_horizon}d"
        target_col_dir = f"future_direction_{self.target_horizon}d"

        clean_df = df.dropna(subset=self.features + [target_col_ret, target_col_dir]).reset_index(drop=True)
        if len(clean_df) < self.min_train_size + 20:
            return {"status": "insufficient_data", "samples": len(clean_df)}

        X = clean_df[self.features].values
        y_ret = clean_df[target_col_ret].values
        y_dir = clean_df[target_col_dir].values

        rf_reg_preds = []
        rf_clf_preds = []
        actual_rets = []
        actual_dirs = []

        # Walk-forward loop
        for start_idx in range(self.min_train_size, len(clean_df) - self.step_size, self.step_size):
            X_train, y_train_ret, y_train_dir = X[:start_idx], y_ret[:start_idx], y_dir[:start_idx]
            X_test = X[start_idx : start_idx + self.step_size]
            y_test_ret = y_ret[start_idx : start_idx + self.step_size]
            y_test_dir = y_dir[start_idx : start_idx + self.step_size]

            # Train models
            reg = RandomForestRegressor(n_estimators=30, max_depth=4, random_state=42)
            clf = RandomForestClassifier(n_estimators=30, max_depth=4, random_state=42)

            reg.fit(X_train, y_train_ret)
            clf.fit(X_train, y_train_dir)

            p_reg = reg.predict(X_test)
            p_clf = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test)

            rf_reg_preds.extend(p_reg)
            rf_clf_preds.extend(p_clf)
            actual_rets.extend(y_test_ret)
            actual_dirs.extend(y_test_dir)

        rf_reg_preds = np.array(rf_reg_preds)
        rf_clf_preds = np.array(rf_clf_preds)
        actual_rets = np.array(actual_rets)
        actual_dirs = np.array(actual_dirs)

        # Classification metrics
        binary_preds = (rf_clf_preds > 0.5).astype(int)
        accuracy = float(np.mean(binary_preds == actual_dirs))

        # Regression metrics
        mae = float(np.mean(np.abs(rf_reg_preds - actual_rets)))
        rmse = float(np.sqrt(np.mean((rf_reg_preds - actual_rets) ** 2)))
        directional_acc = float(np.mean((rf_reg_preds > 0) == (actual_rets > 0)))

        return {
            "status": "success",
            "eval_samples": len(actual_rets),
            "classification": {
                "accuracy": round(accuracy, 4),
                "directional_accuracy": round(directional_acc, 4)
            },
            "regression": {
                "mae": round(mae, 6),
                "rmse": round(rmse, 6)
            }
        }
