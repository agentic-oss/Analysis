"""Reusable Backtesting Engine for metals forecasting research signals."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class BacktestEngine:
    """
    Backtests trading strategies based on research signals accounting for transaction costs,
    slippage, entry/exit timing, position sizing, Sharpe/Sortino/Drawdown/Win-rate metrics.
    """

    def __init__(self, initial_capital: float = 100000.0, transaction_cost_pct: float = 0.0005, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_backtest(self, df: pd.DataFrame, signal_col: str, price_col: str = 'close') -> Dict[str, Any]:
        """
        Executes backtest on signal series (+1 long, -1 short, 0 cash).
        """
        if df.empty or signal_col not in df.columns or price_col not in df.columns:
            return {"error": "Invalid input dataframe or missing required columns"}

        data = df[['date', price_col, signal_col]].copy().dropna().sort_values('date').reset_index(drop=True)
        if len(data) < 2:
            return {"error": "Insufficient data for backtest"}

        # Lag signal by 1 day to execute on next day close (prevent look-ahead bias)
        data['position'] = data[signal_col].shift(1).fillna(0.0)
        data['price_return'] = data[price_col].pct_change().fillna(0.0)

        # Position changes and trading costs
        data['position_change'] = data['position'].diff().abs().fillna(0.0)
        total_cost_pct = self.transaction_cost_pct + self.slippage_pct
        data['costs'] = data['position_change'] * total_cost_pct

        # Strategy daily return
        data['strategy_return'] = (data['position'] * data['price_return']) - data['costs']
        data['equity_curve'] = self.initial_capital * (1.0 + data['strategy_return']).cumprod()

        # Cumulative metrics
        cum_return = (data['equity_curve'].iloc[-1] - self.initial_capital) / self.initial_capital
        total_days = len(data)
        ann_return = ((1.0 + cum_return) ** (252.0 / max(total_days, 1))) - 1.0

        # Drawdown calculation
        peak = data['equity_curve'].cummax()
        drawdown = (data['equity_curve'] - peak) / peak
        max_drawdown = float(drawdown.min())

        # Sharpe & Sortino
        daily_rets = data['strategy_return']
        std_ret = daily_rets.std()
        sharpe = float((daily_rets.mean() / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        downside_std = daily_rets[daily_rets < 0].std()
        sortino = float((daily_rets.mean() / downside_std) * np.sqrt(252)) if downside_std > 0 else 0.0

        # Win rate
        trade_rets = daily_rets[daily_rets != 0]
        win_rate = float((trade_rets > 0).mean()) if len(trade_rets) > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": round(float(data['equity_curve'].iloc[-1]), 2),
            "cumulative_return_pct": round(cum_return * 100, 2),
            "annualized_return_pct": round(ann_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": round(win_rate * 100, 2),
            "total_trades": int((data['position_change'] > 0).sum())
        }
