import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score


class BaselineModelsManager:
    def __init__(self):
        self.classification_models = {
            "logistic_regression": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=50, random_state=42)
        }

    def train_and_evaluate_walk_forward(
        self,
        feature_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "future_direction_5d",
        min_train_size: int = 252,
        step_size: int = 20
    ) -> Dict[str, Any]:
        """
        Walk-forward time-series validation for classification/directional prediction.
        """
        df = feature_df.dropna(subset=feature_cols + [target_col]).copy().reset_index(drop=True)
        if len(df) < min_train_size + step_size:
            return {"error": "Insufficient observations for walk-forward validation"}

        results = {name: {"y_true": [], "y_pred": [], "y_prob": []} for name in self.classification_models.keys()}

        n_samples = len(df)
        for start_idx in range(min_train_size, n_samples, step_size):
            train_data = df.iloc[:start_idx]
            test_data = df.iloc[start_idx:min(start_idx + step_size, n_samples)]

            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values.astype(int)

            X_test = test_data[feature_cols].values
            y_test = test_data[target_col].values.astype(int)

            if len(np.unique(y_train)) < 2:
                continue

            for name, model in self.classification_models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds

                    results[name]["y_true"].extend(y_test)
                    results[name]["y_pred"].extend(preds)
                    results[name]["y_prob"].extend(probs)
                except Exception as e:
                    pass

        eval_summary = {}
        for name, data in results.items():
            if len(data["y_true"]) > 0:
                yt = np.array(data["y_true"])
                yp = np.array(data["y_pred"])
                yprob = np.array(data["y_prob"])

                acc = float(accuracy_score(yt, yp))
                prec = float(precision_score(yt, yp, zero_division=0))
                rec = float(recall_score(yt, yp, zero_division=0))
                f1 = float(f1_score(yt, yp, zero_division=0))
                try:
                    auc = float(roc_auc_score(yt, yprob))
                except Exception:
                    auc = 0.5

                eval_summary[name] = {
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "roc_auc": round(auc, 4),
                    "samples_evaluated": len(yt)
                }

        return eval_summary
