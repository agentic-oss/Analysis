import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, root_mean_squared_error

logger = logging.getLogger(__name__)

class MetalsForecastingModels:
    """Baseline, ML models and walk-forward time-series validation for metals return forecasting."""

    FEATURE_COLS = [
        'return_1d', 'return_5d', 'return_20d', 'rsi_14', 'macd', 'stoch_rsi',
        'atr', 'hist_vol_20', 'dist_sma_20', 'dist_sma_50', 'dist_sma_200',
        'dxy_return_20d', 'us10y_change_20d', 'sp500_return_20d'
    ]

    @staticmethod
    def prepare_data(feature_df: pd.DataFrame, target_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        target_col = f'future_return_{target_horizon}d'
        dir_target_col = f'future_direction_{target_horizon}d'

        df = feature_df.dropna(subset=[target_col]).copy()
        for col in MetalsForecastingModels.FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        X = df[MetalsForecastingModels.FEATURE_COLS]
        y_reg = df[target_col]
        y_clf = df[dir_target_col]
        return X, y_reg, y_clf

    @classmethod
    def walk_forward_validation(
        cls,
        feature_df: pd.DataFrame,
        target_horizon: int = 5,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """Performs expanding-window walk-forward time series cross validation."""
        X, y_reg, y_clf = cls.prepare_data(feature_df, target_horizon)
        if len(X) < 100:
            return {"status": "insufficient_data", "samples": len(X)}

        split_size = len(X) // (n_splits + 1)
        clf_metrics = {"rf": [], "gb": [], "lr": [], "baseline": []}

        for i in range(1, n_splits + 1):
            train_idx = range(0, i * split_size)
            test_idx = range(i * split_size, (i + 1) * split_size)

            X_train, y_train = X.iloc[train_idx], y_clf.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y_clf.iloc[test_idx]

            # 1. Baseline (majority class)
            majority = int(y_train.mode()[0])
            base_preds = [majority] * len(y_test)
            clf_metrics["baseline"].append(accuracy_score(y_test, base_preds))

            # 2. Logistic Regression
            lr = LogisticRegression(max_iter=500)
            lr.fit(X_train, y_train)
            clf_metrics["lr"].append(accuracy_score(y_test, lr.predict(X_test)))

            # 3. Random Forest
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            clf_metrics["rf"].append(accuracy_score(y_test, rf.predict(X_test)))

            # 4. Gradient Boosting
            gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
            gb.fit(X_train, y_train)
            clf_metrics["gb"].append(accuracy_score(y_test, gb.predict(X_test)))

        results = {
            "target_horizon": f"{target_horizon}d",
            "samples": len(X),
            "walk_forward_accuracy": {
                "baseline": float(np.mean(clf_metrics["baseline"])),
                "logistic_regression": float(np.mean(clf_metrics["lr"])),
                "random_forest": float(np.mean(clf_metrics["rf"])),
                "gradient_boosting": float(np.mean(clf_metrics["gb"]))
            }
        }
        return results
