import pandas as pd
import numpy as np
from typing import Dict, Any, List


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct

    def run_signal_backtest(self, df: pd.DataFrame, signal_col: str = "signal") -> Dict[str, Any]:
        """
        Runs backtest on DataFrame where signal_col is in {-1, 0, 1} or boolean/probabilities.
        """
        if df is None or df.empty or "close" not in df.columns or signal_col not in df.columns:
            return {"error": "Invalid input dataframe for backtest"}

        data = df.copy().sort_values("date").reset_index(drop=True)
        prices = data["close"].values
        signals = data[signal_col].values

        capital = self.initial_capital
        position = 0  # 1 for long, -1 for short, 0 for cash
        equity_curve = [capital]
        trades = []

        total_costs = self.transaction_cost_pct + self.slippage_pct

        for i in range(len(prices) - 1):
            curr_price = prices[i]
            next_price = prices[i + 1]
            desired_pos = 1 if signals[i] > 0.5 else (-1 if signals[i] < -0.5 else 0)

            # Rebalance if position changes
            if desired_pos != position:
                # Close old position
                if position != 0:
                    trade_cost = capital * total_costs
                    capital -= trade_cost

                position = desired_pos

                # Open new position
                if position != 0:
                    trade_cost = capital * total_costs
                    capital -= trade_cost

            # Calculate daily return
            if position != 0:
                ret = (next_price - curr_price) / curr_price
                daily_pnl = capital * ret * position
                capital += daily_pnl

            equity_curve.append(capital)

        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()

        cum_return = (capital - self.initial_capital) / self.initial_capital
        ann_return = (1 + cum_return) ** (252 / max(len(prices), 1)) - 1 if len(prices) > 0 else 0.0

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_drawdown = float(drawdown.min())

        std_dev = daily_returns.std()
        sharpe = float(np.sqrt(252) * daily_returns.mean() / (std_dev if std_dev != 0 else 1.0))

        downside_std = daily_returns[daily_returns < 0].std()
        sortino = float(np.sqrt(252) * daily_returns.mean() / (downside_std if downside_std != 0 and not np.isnan(downside_std) else 1.0))

        pos_returns = daily_returns[daily_returns > 0]
        neg_returns = daily_returns[daily_returns < 0]
        win_rate = float(len(pos_returns) / len(daily_returns)) if len(daily_returns) > 0 else 0.0
        profit_factor = float(pos_returns.sum() / abs(neg_returns.sum())) if len(neg_returns) > 0 and neg_returns.sum() != 0 else 1.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(float(capital), 2),
            "cumulative_return": round(float(cum_return), 4),
            "annualized_return": round(float(ann_return), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "trading_days": len(prices)
        }
