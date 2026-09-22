"""
Backtesting engine for evaluating trading strategies based on probabilistic signals without look-ahead bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


class StrategyBacktester:
    """
    Simulates trading strategies accounting for transaction costs, slippage, and position sizing.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0005,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_backtest(
        self,
        prices_df: pd.DataFrame,
        signal_series: pd.Series,
        price_col: str = "close",
    ) -> Dict[str, Any]:
        """
        Runs strategy simulation where signal_series is 1 (Long), -1 (Short), or 0 (Cash).
        Signals on day t dictate positions taken on open/close of day t+1.
        """
        if prices_df.empty or price_col not in prices_df.columns:
            return {}

        df = prices_df.copy().sort_values("date").reset_index(drop=True)
        df["signal"] = signal_series.shift(1).fillna(0)  # Shift by 1 day to prevent look-ahead bias

        df["raw_return"] = df[price_col].pct_change().fillna(0)

        # Apply position signal
        df["strategy_raw_return"] = df["signal"] * df["raw_return"]

        # Transaction costs occur whenever signal changes
        df["trade_occurred"] = (df["signal"] != df["signal"].shift(1)).astype(int)
        df.loc[0, "trade_occurred"] = 0
        total_friction_pct = self.transaction_cost_pct + self.slippage_pct

        df["friction"] = df["trade_occurred"] * total_friction_pct
        df["strategy_net_return"] = df["strategy_raw_return"] - df["friction"]

        # Cumulative equity curve
        df["equity"] = self.initial_capital * (1.0 + df["strategy_net_return"]).cumprod()

        cum_return = (df["equity"].iloc[-1] - self.initial_capital) / self.initial_capital
        ann_return = ((1.0 + cum_return) ** (252.0 / max(1, len(df)))) - 1.0 if len(df) > 0 else 0.0

        # Drawdown calculation
        df["peak"] = df["equity"].cummax()
        df["drawdown"] = (df["equity"] - df["peak"]) / df["peak"]
        max_drawdown = float(df["drawdown"].min())

        # Sharpe Ratio
        mean_daily = df["strategy_net_return"].mean()
        std_daily = df["strategy_net_return"].std()
        sharpe_ratio = float((mean_daily / std_daily) * np.sqrt(252)) if std_daily > 0 else 0.0

        # Win Rate & Profit Factor
        winning_trades = df[df["strategy_net_return"] > 0]["strategy_net_return"]
        losing_trades = df[df["strategy_net_return"] < 0]["strategy_net_return"]
        win_rate = len(winning_trades) / max(1, len(winning_trades) + len(losing_trades))

        gross_profit = winning_trades.sum()
        gross_loss = abs(losing_trades.sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 999.0

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(float(df["equity"].iloc[-1]), 2),
            "cumulative_return_pct": round(cum_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "profit_factor": round(profit_factor, 2),
            "total_trades": int(df["trade_occurred"].sum()),
        }
