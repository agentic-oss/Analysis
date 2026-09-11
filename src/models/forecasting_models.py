"""
Statistical & Machine Learning Forecasting Models and Backtesting Engine.
Baseline Random Forest / Linear Models, Walk-Forward Validation,
Execution Simulation with Costs & Slippage, and Forecast Tracking.
"""

from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


class MetalsForecastingModel:
    """Baseline Statistical & Machine Learning Model for Metals Return & Direction Prediction."""

    def __init__(self, model_type: str = "rf", horizon: int = 5):
        self.model_type = model_type
        self.horizon = horizon
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42) if model_type == "rf" else LogisticRegression()
        self.reg = RandomForestRegressor(n_estimators=100, random_state=42) if model_type == "rf" else LinearRegression()
        self.feature_cols = [
            "rsi_14", "macd", "macd_hist", "volatility_20d", "dist_sma_20_pct",
            "dist_sma_50_pct", "dist_sma_200_pct", "return_1d", "return_5d", "return_20d"
        ]

    def train_and_evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        min_train_size: int = 120,
        step_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Executes expanding-window walk-forward validation without look-ahead bias.
        """
        target_dir = f"future_direction_{self.horizon}d"
        target_ret = f"future_return_{self.horizon}d"

        clean_df = df.dropna(subset=self.feature_cols + [target_dir, target_ret]).copy().reset_index(drop=True)
        if len(clean_df) < min_train_size + step_size:
            return {"accuracy": 0.5, "mae": 0.0, "predictions": []}

        predictions = []

        for i in range(min_train_size, len(clean_df) - self.horizon, step_size):
            train_df = clean_df.iloc[:i]
            test_df = clean_df.iloc[i : i + step_size]

            X_train, y_train_dir, y_train_ret = (
                train_df[self.feature_cols],
                train_df[target_dir],
                train_df[target_ret],
            )
            X_test, y_test_dir, y_test_ret = (
                test_df[self.feature_cols],
                test_df[target_dir],
                test_df[target_ret],
            )

            self.clf.fit(X_train, y_train_dir)
            self.reg.fit(X_train, y_train_ret)

            pred_dir = self.clf.predict(X_test)
            pred_prob = self.clf.predict_proba(X_test)[:, 1] if hasattr(self.clf, "predict_proba") else pred_dir
            pred_ret = self.reg.predict(X_test)

            for idx in range(len(test_df)):
                predictions.append({
                    "date": test_df.iloc[idx]["date"],
                    "actual_dir": int(y_test_dir.iloc[idx]),
                    "pred_dir": int(pred_dir[idx]),
                    "pred_prob": float(pred_prob[idx]),
                    "actual_ret": float(y_test_ret.iloc[idx]),
                    "pred_ret": float(pred_ret[idx]),
                })

        pred_df = pd.DataFrame(predictions)
        if pred_df.empty:
            return {"accuracy": 0.5, "mae": 0.0, "predictions": []}

        acc = accuracy_score(pred_df["actual_dir"], pred_df["pred_dir"])
        mae = mean_absolute_error(pred_df["actual_ret"], pred_df["pred_ret"])

        return {
            "model_type": self.model_type,
            "horizon": self.horizon,
            "accuracy": round(float(acc), 4),
            "mae": round(float(mae), 4),
            "predictions_count": len(pred_df),
            "predictions": predictions,
        }


class BacktestEngine:
    """Reusable Backtesting Engine with transaction costs, slippage, and position sizing."""

    def __init__(self, initial_capital: float = 100000.0, transaction_cost_bps: float = 10.0, slippage_bps: float = 5.0):
        self.initial_capital = initial_capital
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_signal_backtest(self, df: pd.DataFrame, signal_col: str) -> Dict[str, Any]:
        """
        Simulates trading based on signal_col (+1 Long, -1 Short / 0 Cash).
        Prevents look-ahead bias by taking position at close on T, executing return on T+1.
        """
        df = df.copy().sort_values("date").reset_index(drop=True)
        if signal_col not in df.columns or "return_1d" not in df.columns:
            return {}

        df["position"] = df[signal_col].shift(1).fillna(0)  # Position taken on previous close
        df["turnover"] = (df["position"] - df["position"].shift(1).fillna(0)).abs()
        df["costs"] = df["turnover"] * self.cost_pct

        df["strategy_return_raw"] = df["position"] * df["return_1d"]
        df["strategy_return_net"] = df["strategy_return_raw"] - df["costs"]

        df["equity_curve"] = self.initial_capital * (1.0 + df["strategy_return_net"]).cumprod()

        cumulative_return = (df["equity_curve"].iloc[-1] / self.initial_capital) - 1.0
        n_days = len(df)
        annualized_return = ((1.0 + cumulative_return) ** (252.0 / max(n_days, 1))) - 1.0

        # Drawdown calculation
        df["peak"] = df["equity_curve"].cummax()
        df["drawdown"] = (df["equity_curve"] - df["peak"]) / df["peak"]
        max_drawdown = float(df["drawdown"].min())

        # Sharpe ratio
        daily_mean = df["strategy_return_net"].mean()
        daily_std = df["strategy_return_net"].std()
        sharpe_ratio = float((daily_mean / (daily_std + 1e-9)) * np.sqrt(252)) if daily_std > 0 else 0.0

        win_rate = float((df["strategy_return_net"] > 0).mean())

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(float(df["equity_curve"].iloc[-1]), 2),
            "cumulative_return_pct": round(cumulative_return * 100.0, 2),
            "annualized_return_pct": round(annualized_return * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "win_rate_pct": round(win_rate * 100.0, 2),
            "transaction_cost_bps": self.cost_pct * 10000.0,
        }


class ForecastEvaluationTracker:
    """Tracks historical predictions against actual market outcomes over time."""

    def evaluate_past_predictions(self, historical_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates accuracy, MAE, RMSE, and regime calibration metrics across past forecasts."""
        if not historical_predictions:
            return {"status": "No historical predictions recorded yet."}

        df = pd.DataFrame(historical_predictions)
        if "actual_return" not in df.columns or df["actual_return"].isnull().all():
            return {"status": "Pending actual outcome data."}

        completed = df.dropna(subset=["actual_return"]).copy()
        completed["actual_direction"] = (completed["actual_return"] > 0).astype(int)
        completed["predicted_direction"] = (completed["predicted_return"] > 0).astype(int)

        accuracy = accuracy_score(completed["actual_direction"], completed["predicted_direction"])
        mae = mean_absolute_error(completed["actual_return"], completed["predicted_return"])
        rmse = np.sqrt(mean_squared_error(completed["actual_return"], completed["predicted_return"]))

        return {
            "total_forecasts_evaluated": len(completed),
            "directional_accuracy_pct": round(float(accuracy) * 100.0, 2),
            "mae": round(float(mae), 6),
            "rmse": round(float(rmse), 6),
            "last_updated": datetime.utcnow().isoformat(),
        }
