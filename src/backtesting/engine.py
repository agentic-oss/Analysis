import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

class ForecastingModelPipeline:
    """Baseline and ML forecasting models with walk-forward time-series validation."""

    def __init__(self, feature_cols: List[str]):
        self.feature_cols = feature_cols

    def evaluate_walk_forward(
        self,
        df: pd.DataFrame,
        target_col: str = "future_return_5d",
        min_train_size: int = 100,
        step_size: int = 20
    ) -> Dict[str, Any]:
        """Walk-forward time-series validation for directional classification and return regression."""
        cols = [c for c in self.feature_cols if c in df.columns]
        valid_df = df.dropna(subset=cols + [target_col]).sort_values("date").reset_index(drop=True)

        if len(valid_df) < min_train_size + step_size:
            return {"status": "insufficient_data", "samples": len(valid_df)}

        X = valid_df[cols].astype(float)
        y_ret = valid_df[target_col].astype(float)
        y_dir = (y_ret > 0).astype(int)

        class_preds = []
        class_actuals = []
        reg_preds = []
        reg_actuals = []

        n_samples = len(valid_df)
        for i in range(min_train_size, n_samples, step_size):
            X_train, y_train_dir, y_train_ret = X.iloc[:i], y_dir.iloc[:i], y_ret.iloc[:i]
            X_test, y_test_dir, y_test_ret = X.iloc[i:i+step_size], y_dir.iloc[i:i+step_size], y_ret.iloc[i:i+step_size]

            if len(X_test) == 0:
                break

            # Logistic Regression Classifier
            clf = LogisticRegression(max_iter=500)
            clf.fit(X_train, y_train_dir)
            p_dir = clf.predict(X_test)

            # Linear Regression Model
            reg = LinearRegression()
            reg.fit(X_train, y_train_ret)
            p_ret = reg.predict(X_test)

            class_preds.extend(p_dir)
            class_actuals.extend(y_test_dir)
            reg_preds.extend(p_ret)
            reg_actuals.extend(y_test_ret)

        acc = accuracy_score(class_actuals, class_preds) if class_preds else 0.0
        prec = precision_score(class_actuals, class_preds, zero_division=0) if class_preds else 0.0
        rec = recall_score(class_actuals, class_preds, zero_division=0) if class_preds else 0.0
        f1 = f1_score(class_actuals, class_preds, zero_division=0) if class_preds else 0.0

        mae = mean_absolute_error(reg_actuals, reg_preds) if reg_preds else 0.0
        rmse = np.sqrt(mean_squared_error(reg_actuals, reg_preds)) if reg_preds else 0.0
        r2 = r2_score(reg_actuals, reg_preds) if reg_preds else 0.0

        return {
            "status": "success",
            "samples": len(class_preds),
            "classification_metrics": {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1)
            },
            "regression_metrics": {
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2)
            }
        }


class BacktestEngine:
    """Reusable backtesting framework with transaction costs, slippage, and position sizing."""

    def __init__(self, transaction_cost_bps: float = 5.0, slippage_bps: float = 2.0):
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close"
    ) -> Dict[str, Any]:
        """Runs daily signal backtest. Signal should be 1 (Long), 0 (Cash), -1 (Short)."""
        if df.empty or signal_col not in df.columns or price_col not in df.columns:
            return {}

        df = df.copy().sort_values("date").reset_index(drop=True)
        prices = df[price_col]
        signals = df[signal_col].shift(1).fillna(0) # Execution on next day open/close

        daily_returns = prices.pct_change(1).fillna(0)
        position_changes = signals.diff().abs().fillna(0)

        strategy_returns = (signals * daily_returns) - (position_changes * self.cost_pct)
        cum_returns = (1.0 + strategy_returns).cumprod()

        total_return = float(cum_returns.iloc[-1] - 1.0) if len(cum_returns) > 0 else 0.0
        ann_return = float((1.0 + total_return) ** (252.0 / max(len(df), 1)) - 1.0)

        std_dev = float(strategy_returns.std() * np.sqrt(252))
        sharpe = float(ann_return / std_dev) if std_dev > 0 else 0.0

        # Drawdown calculation
        rolling_max = cum_returns.cummax()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        trades = (position_changes > 0).sum()
        winning_days = (strategy_returns > 0).sum()
        total_days = len(strategy_returns)
        win_rate = float(winning_days / total_days) if total_days > 0 else 0.0

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "annualized_volatility": std_dev,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": int(trades),
            "transaction_costs_applied_bps": self.cost_pct * 10000.0
        }
