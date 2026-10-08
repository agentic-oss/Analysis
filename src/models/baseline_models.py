"""
Baseline Machine Learning Models for Precious Metals Forecasting.
Uses expanding-window / walk-forward time-series validation without shuffling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


class BaselineModels:
    """
    Evaluates baseline directional and return prediction models using walk-forward validation.
    """

    @staticmethod
    def train_evaluate_walk_forward(
        df: pd.DataFrame,
        feature_cols: list,
        target_col: str = "future_direction_5d",
        model_type: str = "random_forest",
        train_window_min: int = 252,
        step_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Executes expanding-window walk-forward validation on time-series dataset.
        """
        clean_df = df.dropna(subset=feature_cols + [target_col]).sort_values("date").reset_index(drop=True)
        if len(clean_df) < train_window_min + step_size:
            return {"error": "Insufficient dataset length for walk-forward validation"}

        X = clean_df[feature_cols]
        y = clean_df[target_col]

        y_true = []
        y_pred = []

        for start_idx in range(train_window_min, len(clean_df), step_size):
            X_train, y_train = X.iloc[:start_idx], y.iloc[:start_idx]
            X_test = X.iloc[start_idx : start_idx + step_size]
            y_test = y.iloc[start_idx : start_idx + step_size]

            if X_test.empty:
                break

            if model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            elif model_type == "logistic_regression":
                model = LogisticRegression(max_iter=500)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            y_true.extend(y_test.values)
            y_pred.extend(preds)

        if not y_true:
            return {}

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        return {
            "model_type": model_type,
            "target": target_col,
            "total_test_samples": len(y_true),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        }
