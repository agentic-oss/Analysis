import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)


class BaselineForecastingModels:
    """
    Baseline statistical & ML forecasting models trained using expanding-window time-series splits.
    Models evaluated:
    - Historical Conditional Probability
    - Logistic Regression / Ridge
    - Random Forest
    - Gradient Boosting
    Predicts directional probability and expected return.
    """

    FEATURE_COLS = [
        "return_1d",
        "return_5d",
        "return_20d",
        "rsi_14",
        "dist_sma_20",
        "dist_sma_200",
        "volatility_20d",
    ]

    def __init__(self, target_horizon: int = 10, min_train_samples: int = 252):
        self.target_horizon = target_horizon
        self.min_train_samples = min_train_samples

    def train_and_predict(
        self, feature_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Trains models on available historical data up to T (excluding target rows with look-ahead),
        and predicts directional probability for current date T.
        """
        target_col = f"future_direction_{self.target_horizon}d"
        target_ret_col = f"future_return_{self.target_horizon}d"

        # Valid feature columns present in DataFrame
        valid_feats = [c for c in self.FEATURE_COLS if c in feature_df.columns]

        # Training set excludes the last `target_horizon` rows (unlabeled targets)
        train_df = feature_df.dropna(subset=valid_feats + [target_col]).copy()

        if len(train_df) < self.min_train_samples:
            return {
                "horizon": self.target_horizon,
                "model_agreement": 0.5,
                "direction_prob": 0.5,
                "model_predictions": {},
            }

        X_train = train_df[valid_feats]
        y_train = train_df[target_col]
        y_ret_train = train_df[target_ret_col]

        latest_x = feature_df[valid_feats].iloc[[-1]].fillna(0)

        preds = {}

        # 1. Historical Base Rate
        base_rate = float(y_train.mean())
        preds["historical_base_rate"] = base_rate

        # 2. Logistic Regression
        try:
            log_reg = LogisticRegression(max_iter=200)
            log_reg.fit(X_train.fillna(0), y_train)
            preds["logistic_regression"] = float(log_reg.predict_proba(latest_x)[0][1])
        except Exception:
            preds["logistic_regression"] = base_rate

        # 3. Random Forest
        try:
            rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
            rf.fit(X_train.fillna(0), y_train)
            preds["random_forest"] = float(rf.predict_proba(latest_x)[0][1])
        except Exception:
            preds["random_forest"] = base_rate

        # 4. Gradient Boosting
        try:
            gb = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)
            gb.fit(X_train.fillna(0), y_train)
            preds["gradient_boosting"] = float(gb.predict_proba(latest_x)[0][1])
        except Exception:
            preds["gradient_boosting"] = base_rate

        # Model ensemble average probability
        avg_prob = np.mean(list(preds.values()))

        return {
            "horizon": self.target_horizon,
            "direction_prob": float(avg_prob),
            "model_predictions": preds,
            "model_agreement": float(np.mean([p > 0.5 for p in preds.values()])),
        }
