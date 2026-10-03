import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


class ModelEvaluator:
    """Evaluates classification, regression, and trading metrics."""

    @staticmethod
    def evaluate_classification(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        if not y_true or not y_pred:
            return {}
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        }

    @staticmethod
    def evaluate_regression(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
        if not y_true or not y_pred:
            return {}
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        dir_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())
        return {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "directional_accuracy": round(dir_acc, 4),
        }
