import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier


class BaselineModels:
    """
    Implements baseline statistical and machine learning models
    with expanding-window and walk-forward time-series validation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def train_evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        horizon: int = 5,
        min_train_size: int = 100,
        step_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Executes expanding-window walk-forward validation for directional prediction and return regression.
        Strictly prevents look-ahead bias and random shuffle.
        """
        if df is None or df.empty or len(df) < min_train_size + horizon + step_size:
            return {"error": "Insufficient data for walk-forward validation"}

        clean_df = df.dropna(subset=feature_cols + [target_col]).reset_index(drop=True)
        if len(clean_df) < min_train_size + step_size:
            return {"error": "Not enough clean rows after dropping NAs"}

        y_true_dir = []
        y_pred_dir_prob = []
        y_pred_dir = []

        y_true_ret = []
        y_pred_ret = []

        n_samples = len(clean_df)

        for train_end in range(min_train_size, n_samples - step_size, step_size):
            train_data = clean_df.iloc[:train_end]
            test_data = clean_df.iloc[train_end : train_end + step_size]

            X_train = train_data[feature_cols]
            y_train_ret = train_data[target_col]
            y_train_dir = (y_train_ret > 0).astype(int)

            X_test = test_data[feature_cols]
            y_test_ret = test_data[target_col]
            y_test_dir = (y_test_ret > 0).astype(int)

            # Fit Random Forest Classifier
            clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=self.random_state)
            clf.fit(X_train, y_train_dir)
            preds_dir = clf.predict(X_test)
            probs_dir = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds_dir

            # Fit Linear Regression for return magnitude
            reg = LinearRegression()
            reg.fit(X_train, y_train_ret)
            preds_ret = reg.predict(X_test)

            y_true_dir.extend(y_test_dir.tolist())
            y_pred_dir.extend(preds_dir.tolist())
            y_pred_dir_prob.extend(probs_dir.tolist())

            y_true_ret.extend(y_test_ret.tolist())
            y_pred_ret.extend(preds_ret.tolist())

        # Performance evaluation metrics
        y_true_dir = np.array(y_true_dir)
        y_pred_dir = np.array(y_pred_dir)
        y_true_ret = np.array(y_true_ret)
        y_pred_ret = np.array(y_pred_ret)

        accuracy = float((y_true_dir == y_pred_dir).mean())
        mae = float(np.abs(y_true_ret - y_pred_ret).mean())
        rmse = float(np.sqrt(((y_true_ret - y_pred_ret) ** 2).mean()))

        return {
            "model_type": "RandomForest + LinearRegression",
            "validation_method": "expanding_window_walk_forward",
            "total_test_samples": len(y_true_dir),
            "directional_accuracy": round(accuracy, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
        }
