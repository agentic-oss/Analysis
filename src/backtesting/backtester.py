import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """Reusable backtester accounting for transaction costs, slippage, and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_backtest(
        self,
        prices: pd.Series,
        signals: pd.Series
    ) -> Dict[str, Any]:
        """
        prices: Series of close prices indexed by Date.
        signals: Series of target position weights (-1.0 to +1.0) indexed by Date.
        """
        df = pd.DataFrame({'price': prices, 'signal': signals}).dropna()
        if len(df) < 2:
            return {"error": "Insufficient data for backtest."}

        df['signal'] = df['signal'].shift(1).fillna(0.0)  # avoid look-ahead bias (trade on next open/close)
        df['price_return'] = df['price'].pct_change().fillna(0.0)

        # Position changes for transaction costs
        pos_change = df['signal'].diff().abs().fillna(0.0)
        costs = pos_change * (self.transaction_cost_pct + self.slippage_pct)

        df['strategy_return'] = (df['signal'] * df['price_return']) - costs
        df['equity_curve'] = self.initial_capital * (1.0 + df['strategy_return']).cumprod()

        # Performance Metrics
        cum_return = (df['equity_curve'].iloc[-1] - self.initial_capital) / self.initial_capital
        ann_return = ((1.0 + cum_return) ** (252.0 / len(df))) - 1.0 if len(df) > 0 else 0.0

        daily_std = df['strategy_return'].std()
        sharpe = (df['strategy_return'].mean() / (daily_std + 1e-10)) * np.sqrt(252)

        downside_std = df[df['strategy_return'] < 0]['strategy_return'].std()
        sortino = (df['strategy_return'].mean() / (downside_std + 1e-10)) * np.sqrt(252)

        # Drawdown
        peak = df['equity_curve'].cummax()
        drawdown = (df['equity_curve'] - peak) / peak
        max_drawdown = float(drawdown.min())

        winning_days = (df['strategy_return'] > 0).sum()
        total_days = len(df[df['signal'] != 0])
        win_rate = float(winning_days / total_days) if total_days > 0 else 0.0

        gross_profit = df[df['strategy_return'] > 0]['strategy_return'].sum()
        gross_loss = abs(df[df['strategy_return'] < 0]['strategy_return'].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else 1.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(df['equity_curve'].iloc[-1]),
            "cumulative_return": float(cum_return),
            "annualized_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": int(pos_change[pos_change > 0].count())
        }
