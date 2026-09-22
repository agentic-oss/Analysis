"""
Baseline statistical and machine learning forecasting models for precious metals returns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class MetalsForecastingModels:
    """
    Time-series machine learning models with walk-forward validation:
    - Baseline Conditional Probability
    - Logistic Regression Classifier
    - Linear Regression Model
    - Random Forest Classifier
    - Gradient Boosting Classifier
    """

    def __init__(self, target_horizon_days: int = 5):
        self.target_horizon_days = target_horizon_days
        self.models = {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        }

    def prepare_data(
        self,
        features_df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Extracts feature matrix X and target y (1 for positive return, 0 otherwise)
        filtering out rows with NaNs in features or target.
        """
        df = features_df.dropna(subset=feature_cols + [target_col]).copy()
        if df.empty:
            return np.array([]), np.array([]), df

        X = df[feature_cols].values
        y = (df[target_col] > 0).astype(int).values
        return X, y, df

    def train_and_predict(
        self,
        features_df: pd.DataFrame,
        feature_cols: list,
        target_col: str,
        current_date: str,
    ) -> Dict[str, Any]:
        """
        Trains models strictly on historical data prior to current_date and generates predictions for current_date.
        Never shuffles time-series data.
        """
        df = features_df.sort_values("date").reset_index(drop=True)

        if current_date not in df["date"].values:
            current_idx = len(df) - 1
        else:
            current_idx = df[df["date"] == current_date].index[0]

        # Historical split: target_horizon_days gap to prevent overlap
        historical_df = df.iloc[: max(0, current_idx - self.target_horizon_days)]
        current_row_df = df.iloc[[current_idx]]

        X_train, y_train, train_df = self.prepare_data(historical_df, feature_cols, target_col)

        if len(X_train) < 30 or len(np.unique(y_train)) < 2:
            # Fallback to historical baseline probability
            prob_pos = float(np.mean(y_train)) if len(y_train) > 0 else 0.5
            return {
                "ensemble_positive_probability": prob_pos,
                "model_predictions": {},
                "model_agreement_ratio": 1.0,
                "trained_samples": len(X_train),
            }

        X_current = current_row_df[feature_cols].fillna(0.0).values

        model_preds = {}
        probs = []

        for name, model in self.models.items():
            try:
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_current)[0][1]
                model_preds[name] = round(float(prob), 4)
                probs.append(prob)
            except Exception as e:
                model_preds[name] = 0.5

        ensemble_prob = float(np.mean(probs))
        agreed_count = sum(1 for p in probs if (p > 0.5) == (ensemble_prob > 0.5))
        agreement_ratio = agreed_count / len(probs) if probs else 1.0

        return {
            "ensemble_positive_probability": round(ensemble_prob, 4),
            "model_predictions": model_preds,
            "model_agreement_ratio": round(agreement_ratio, 2),
            "trained_samples": len(X_train),
        }
