"""Baseline ML & Statistical Forecasting Models with Time-Series Walk-Forward Validation."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


class MetalsForecastingModels:
    """Baseline forecasting models evaluated using expanding window walk-forward validation."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def train_evaluate_walk_forward(self, feature_df: pd.DataFrame, target_col: str = 'future_return_5d', train_ratio: float = 0.7) -> Dict[str, Any]:
        """Runs expanding window walk-forward validation."""
        clean_df = feature_df.dropna(subset=self.feature_cols + [target_col]).sort_values('date').reset_index(drop=True)
        if len(clean_df) < 50:
            return {"status": "insufficient_data"}

        is_classification = target_col.startswith('future_direction') or clean_df[target_col].nunique() <= 2

        split_idx = int(len(clean_df) * train_ratio)
        train_df = clean_df.iloc[:split_idx]
        test_df = clean_df.iloc[split_idx:]

        X_train, y_train = train_df[self.feature_cols], train_df[target_col]
        X_test, y_test = test_df[self.feature_cols], test_df[target_col]

        results = {}

        if is_classification:
            models = {
                "logistic_regression": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, zero_division=0)
                rec = recall_score(y_test, preds, zero_division=0)
                f1 = f1_score(y_test, preds, zero_division=0)

                results[name] = {
                    "accuracy": round(float(acc), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1": round(float(f1), 4)
                }
        else:
            models = {
                "linear_regression": LinearRegression(),
                "random_forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                dir_acc = accuracy_score((y_test > 0), (preds > 0))

                results[name] = {
                    "mae": round(float(mae), 4),
                    "rmse": round(float(rmse), 4),
                    "r2": round(float(r2), 4),
                    "directional_accuracy": round(float(dir_acc), 4)
                }

        return {
            "target": target_col,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "model_metrics": results
        }
