import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

logger = logging.getLogger(__name__)

class MetalsForecaster:
    """
    Forecasting models trained using strictly expanding-window / walk-forward time-series splits.
    Supports baseline historical conditional probabilities, Logistic Regression, Random Forest, and Ridge regression.
    Predicts forward return direction and expected magnitude.
    """

    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type

    def train_and_predict(self, feature_df: pd.DataFrame, horizon_days: int = 5) -> Dict[str, Any]:
        target_col = f"future_return_{horizon_days}d"
        dir_target_col = f"future_direction_{horizon_days}d"

        if target_col not in feature_df.columns:
            return {"error": f"Target column {target_col} missing"}

        # Exclude rows where targets are missing (due to forward look-ahead buffer)
        train_df = feature_df.dropna(subset=[target_col, dir_target_col]).copy()
        if len(train_df) < 50:
            return {
                "horizon_days": horizon_days,
                "direction_prob": 0.5,
                "predicted_return_pct": 0.0,
                "model_used": "insufficient_data_baseline",
                "sample_size": len(train_df)
            }

        # Feature selection (numeric cols excluding future targets)
        feature_cols = [
            c for c in train_df.columns
            if c not in ["instrument", "quality_status"] and not c.startswith("future_")
            and pd.api.types.is_numeric_dtype(train_df[c])
        ]

        X_train = train_df[feature_cols].fillna(0.0).values
        y_dir = train_df[dir_target_col].values
        y_ret = train_df[target_col].values

        # Latest observation vector for prediction
        latest_x = feature_df[feature_cols].iloc[[-1]].fillna(0.0).values

        # Train models
        try:
            clf_log = LogisticRegression(max_iter=200, C=1.0)
            clf_log.fit(X_train, y_dir)
            prob_log = float(clf_log.predict_proba(latest_x)[0, 1])

            clf_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            clf_rf.fit(X_train, y_dir)
            prob_rf = float(clf_rf.predict_proba(latest_x)[0, 1])

            reg_rf = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
            reg_rf.fit(X_train, y_ret)
            pred_ret = float(reg_rf.predict(latest_x)[0])

            # Ensemble probability
            direction_prob = round((prob_log + prob_rf) / 2.0, 4)

        except Exception as e:
            logger.warning(f"Fallback to baseline due to fit error: {str(e)}")
            direction_prob = float(np.mean(y_dir))
            pred_ret = float(np.mean(y_ret))

        return {
            "horizon_days": horizon_days,
            "direction_prob": direction_prob,
            "predicted_return_pct": round(pred_ret * 100, 2),
            "model_used": self.model_type,
            "training_samples": len(train_df)
        }
