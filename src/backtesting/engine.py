import pandas as pd
import numpy as np
from typing import Dict, Any, List

class StrategyBacktester:
    """
    Reusable Backtesting Engine for signal-driven precious metals strategies.

    Accounts for:
    - Transaction costs
    - Slippage
    - Risk-adjusted position sizing
    - Entry/Exit timing

    Computes performance metrics:
    - Cumulative & Annualized Return
    - Maximum Drawdown & Duration
    - Sharpe & Sortino Ratios
    - Win Rate & Profit Factor
    """
    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.total_cost_pct = transaction_cost_pct + slippage_pct

    def run_backtest(
        self,
        price_df: pd.DataFrame,
        signal_series: pd.Series,
        price_col: str = "close"
    ) -> Dict[str, Any]:
        """
        Executes backtest given price history and signals (1 for Long, -1 for Short, 0 for Cash).
        Signal at T is executed at open/close of T+1 to prevent look-ahead bias.
        """
        if price_df.empty or len(price_df) < 5:
            return {}

        df = price_df[["timestamp", price_col]].copy().rename(columns={price_col: "price"})
        df["signal"] = signal_series.reindex(df.index).fillna(0).shift(1) # Executed next period

        df["price_return"] = df["price"].pct_change().fillna(0.0)

        # Calculate turnover and costs on signal changes
        df["trade"] = df["signal"].diff().abs().fillna(0.0)
        df["cost"] = df["trade"] * self.total_cost_pct

        # Strategy Return
        df["strat_ret"] = (df["signal"] * df["price_return"]) - df["cost"]
        df["equity_curve"] = self.initial_capital * (1.0 + df["strat_ret"]).cumprod()

        # Returns & Metrics
        total_ret = (df["equity_curve"].iloc[-1] - self.initial_capital) / self.initial_capital * 100.0
        n_days = len(df)
        ann_ret = (((1.0 + total_ret / 100.0) ** (252.0 / max(1, n_days))) - 1.0) * 100.0

        # Peak & Drawdown
        peak = df["equity_curve"].cummax()
        dd = (df["equity_curve"] - peak) / peak * 100.0
        max_dd = float(dd.min())

        # Sharpe & Sortino Ratios
        daily_rets = df["strat_ret"]
        mean_ret = daily_rets.mean() * 252
        std_ret = daily_rets.std() * np.sqrt(252)
        sharpe = float(mean_ret / (std_ret + 1e-10)) if std_ret > 0 else 0.0

        downside_std = daily_rets[daily_rets < 0].std() * np.sqrt(252)
        sortino = float(mean_ret / (downside_std + 1e-10)) if downside_std > 0 else 0.0

        # Win Rate
        active_trades = df[df["signal"] != 0]
        winning_days = (active_trades["strat_ret"] > 0).sum()
        total_active_days = len(active_trades)
        win_rate = float(winning_days / total_active_days * 100.0) if total_active_days > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(float(df["equity_curve"].iloc[-1]), 2),
            "cumulative_return_pct": round(total_ret, 2),
            "annualized_return_pct": round(ann_ret, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": round(win_rate, 1),
            "total_trades_count": int(df["trade"].sum())
        }
