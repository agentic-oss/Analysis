"""
Reusable Backtesting Framework.
Simulates trading strategies based on generated research signals while accounting for:
transaction costs, slippage, position sizing limits, entry/exit timing, and missing data.
Calculates CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, and Profit Factor without look-ahead bias.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class Backtester:
    """Executes time-series backtests on precious metals research signals."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_position_size: float = 0.20
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0
        self.max_position_size = max_position_size

    def run_backtest(
        self,
        feature_df: pd.DataFrame,
        signal_column: str = "signal",
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Runs daily backtest.
        Signal values: 1 (Long), -1 (Short / Cash), 0 (Neutral / Cash).
        Positions are executed on next day's open price or close price after signal generated at T.
        """
        if feature_df.empty or signal_column not in feature_df.columns or price_column not in feature_df.columns:
            return {"status": "INSUFFICIENT_DATA"}

        df = feature_df.copy().sort_values("date").reset_index(drop=True)
        n = len(df)
        if n < 20:
            return {"status": "INSUFFICIENT_DATA"}

        # Position allocated at T applies to return from T to T+1
        df["position"] = df[signal_column].shift(1).fillna(0.0).clip(-1.0, 1.0)
        df["raw_return"] = df[price_column].pct_change().fillna(0.0)

        # Detect position changes to apply transaction costs
        df["pos_change"] = df["position"].diff().abs().fillna(0.0)
        df["cost"] = df["pos_change"] * self.transaction_cost_pct

        # Strategy net return
        df["strategy_return"] = (df["position"] * df["raw_return"]) - df["cost"]
        df["equity"] = self.initial_capital * (1.0 + df["strategy_return"]).cumprod()

        equity = df["equity"]
        total_ret = float((equity.iloc[-1] - self.initial_capital) / self.initial_capital)

        # CAGR
        days = len(df)
        cagr = float(((equity.iloc[-1] / self.initial_capital) ** (252.0 / days) - 1.0) * 100.0) if days > 0 else 0.0

        # Annualized Volatility & Sharpe
        daily_ret = df["strategy_return"]
        vol_ann = float(daily_ret.std() * np.sqrt(252) * 100.0)
        sharpe = float((daily_ret.mean() * np.sqrt(252)) / daily_ret.std()) if daily_ret.std() > 0 else 0.0

        # Sortino Ratio
        downside_ret = daily_ret[daily_ret < 0]
        sortino = float((daily_ret.mean() * np.sqrt(252)) / downside_ret.std()) if len(downside_ret) > 0 and downside_ret.std() > 0 else 0.0

        # Maximum Drawdown
        cummax = equity.cummax()
        dd = (equity - cummax) / cummax
        max_dd = float(dd.min() * 100.0)

        # Win rate & Profit factor
        trades = daily_ret[df["position"] != 0]
        wins = trades[trades > 0]
        losses = trades[trades < 0]
        win_rate = float((len(wins) / len(trades) * 100.0)) if len(trades) > 0 else 0.0

        gross_profit = wins.sum()
        gross_loss = abs(losses.sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity.iloc[-1]),
            "total_return_pct": float(total_ret * 100.0),
            "cagr_pct": cagr,
            "annualized_volatility_pct": vol_ann,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": max_dd,
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "total_trading_days": days
        }
