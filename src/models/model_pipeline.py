import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelPipeline:
    """Manages training and evaluation of baseline statistical and ML models using walk-forward validation."""

    def __init__(self, feature_cols: List[str], target_col: str, horizon_gap: int = 5):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.horizon_gap = horizon_gap
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=500),
            "random_forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
        }

    def walk_forward_evaluate(
        self,
        df: pd.DataFrame,
        min_train_size: int = 100,
        step_size: int = 20
    ) -> Dict[str, Any]:
        """
        Executes expanding-window walk-forward validation to evaluate models without temporal leakage.
        Purges the last `horizon_gap` rows from train split to avoid target overlap with test split.
        """
        clean_df = df.dropna(subset=self.feature_cols + [self.target_col]).reset_index(drop=True)
        if len(clean_df) < min_train_size + step_size + self.horizon_gap:
            return {"error": "Insufficient clean data for walk-forward evaluation", "results": {}}

        results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in self.models.keys()}

        for start in range(min_train_size, len(clean_df) - step_size, step_size):
            # Purge last horizon_gap observations from training set to avoid target overlap into test set
            train_end = max(0, start - self.horizon_gap)
            if train_end < 20:
                continue

            train_sub = clean_df.iloc[:train_end]
            test_sub = clean_df.iloc[start : start + step_size]

            X_train = train_sub[self.feature_cols]
            y_train = train_sub[self.target_col]
            X_test = test_sub[self.feature_cols]
            y_test = test_sub[self.target_col]

            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

                    results[name]["y_true"].extend(y_test.tolist())
                    results[name]["y_pred"].extend(preds.tolist())
                    results[name]["y_prob"].extend(probs.tolist())
                except Exception:
                    pass

        metrics_summary = {}
        for name, data in results.items():
            if data["y_true"]:
                y_true = np.array(data["y_true"])
                y_pred = np.array(data["y_pred"])
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)

                metrics_summary[name] = {
                    "accuracy": float(acc),
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1_score": float(f1),
                    "sample_count": len(y_true)
                }

        return metrics_summary
