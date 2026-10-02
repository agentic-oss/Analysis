import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

class BaselineForecastingModels:
    """Baseline models (Historical Conditional Probability, Logistic Regression, Random Forest, Ridge Regression) with time-series walk-forward validation."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.feature_cols = [
            "rsi_14", "volatility_20d", "dist_sma_200", "dist_sma_20",
            "gold_silver_ratio_zscore", "return_1d", "return_5d", "return_20d"
        ]

    def train_predict_walk_forward(
        self,
        feature_df: pd.DataFrame,
        horizon_days: int = 10,
        min_train_size: int = 30
    ) -> dict:
        """
        Executes strictly sequential time-series walk-forward split (NO random shuffling).
        Predicts directional probability and expected return for date T (the last row of feature_df).
        """
        target_col = f"future_return_{horizon_days}d"
        dir_col = f"future_direction_{horizon_days}d"

        if feature_df.empty:
            return {"direction_probability": 0.50, "expected_return_pct": 0.0, "model_agreement": "1/1", "sample_size": 0}

        df_sorted = feature_df.sort_values("date").reset_index(drop=True)
        avail_features = [c for c in self.feature_cols if c in df_sorted.columns]

        if not avail_features:
            return {"direction_probability": 0.50, "expected_return_pct": 0.0, "model_agreement": "1/1", "sample_size": 0}

        # Date T is the latest row in feature_df
        test_row = df_sorted.iloc[[-1]].copy()
        # Fill any missing feature values in test set with historical median
        X_test = test_row[avail_features].apply(pd.to_numeric, errors="coerce").fillna(df_sorted[avail_features].median())

        # Training set: historical rows where target horizon has matured (non-null targets)
        train_df = df_sorted.dropna(subset=avail_features + [target_col, dir_col])

        if len(train_df) < min_train_size:
            # Fallback to historical unconditional mean if available
            hist_dir_prob = float((df_sorted[dir_col].dropna() > 0).mean()) if len(df_sorted[dir_col].dropna()) > 0 else 0.50
            hist_exp_ret = float(df_sorted[target_col].dropna().mean()) if len(df_sorted[target_col].dropna()) > 0 else 0.0
            return {
                "direction_probability": round(hist_dir_prob, 3),
                "expected_return_pct": round(hist_exp_ret * 100, 2),
                "model_agreement": "1/1 (Historical Baseline)",
                "sample_size": len(train_df)
            }

        X_train = train_df[avail_features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y_dir_train = train_df[dir_col].astype(int)
        y_ret_train = train_df[target_col].astype(float)

        # Model 1: Logistic Regression
        try:
            lr = LogisticRegression(random_state=self.random_seed, max_iter=200)
            lr.fit(X_train, y_dir_train)
            lr_prob = float(lr.predict_proba(X_test)[0][1])
        except Exception:
            lr_prob = float((y_dir_train > 0).mean())

        # Model 2: Random Forest Classifier
        try:
            rf = RandomForestClassifier(n_estimators=50, random_state=self.random_seed)
            rf.fit(X_train, y_dir_train)
            rf_prob = float(rf.predict_proba(X_test)[0][1])
        except Exception:
            rf_prob = float((y_dir_train > 0).mean())

        # Model 3: Historical Baseline
        hist_prob = float((y_dir_train > 0).mean())

        # Model 4: Ridge Regression for return estimate
        try:
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_ret_train)
            pred_return = float(ridge.predict(X_test)[0])
        except Exception:
            pred_return = float(y_ret_train.mean())

        # Ensembled Directional Probability
        ensemble_prob = (lr_prob * 0.40) + (rf_prob * 0.40) + (hist_prob * 0.20)

        # Model Agreement Calculation
        votes_bull = sum([lr_prob > 0.50, rf_prob > 0.50, hist_prob > 0.50])
        agreement_str = f"{votes_bull}/3"

        return {
            "direction_probability": round(ensemble_prob, 3),
            "expected_return_pct": round(pred_return * 100, 2),
            "model_agreement": agreement_str,
            "models": {
                "logistic_regression_prob": round(lr_prob, 3),
                "random_forest_prob": round(rf_prob, 3),
                "historical_baseline_prob": round(hist_prob, 3)
            },
            "sample_size": len(train_df)
        }
