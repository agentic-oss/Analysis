import pandas as pd
import numpy as np
from typing import Dict, Any

class BacktestEngine:
    """Reusable strategy backtesting framework with transaction costs, slippage, and performance metrics."""

    @staticmethod
    def run_signal_backtest(
        price_df: pd.DataFrame,
        signals_series: pd.Series,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0
    ) -> Dict[str, Any]:
        """
        Executes signal-based strategy backtest.
        signals_series: 1 for long, -1 for short, 0 for cash.
        """
        if price_df.empty or len(price_df) < 10:
            return {}

        df = price_df.copy().sort_values('date').reset_index(drop=True)
        close = df['close'].astype(float)
        returns = close.pct_change().fillna(0.0)

        # Shift signals by 1 day to prevent look-ahead bias
        pos = signals_series.shift(1).fillna(0.0)

        # Cost factor per position change
        cost_factor = (transaction_cost_bps + slippage_bps) / 10000.0
        pos_change = pos.diff().abs().fillna(0.0)

        strategy_returns = (pos * returns) - (pos_change * cost_factor)
        equity_curve = initial_capital * (1 + strategy_returns).cumprod()

        total_return_pct = float((equity_curve.iloc[-1] - initial_capital) / initial_capital * 100)

        # Risk & Drawdown
        cum_max = equity_curve.cummax()
        drawdown = (equity_curve - cum_max) / cum_max * 100
        max_drawdown_pct = float(drawdown.min())

        std_dev = float(strategy_returns.std() * np.sqrt(252))
        ann_return = float(strategy_returns.mean() * 252)
        sharpe = round(ann_return / std_dev, 2) if std_dev > 0 else 0.0

        win_rate = float((strategy_returns > 0).sum() / (pos != 0).sum() * 100) if (pos != 0).sum() > 0 else 0.0

        return {
            "initial_capital": initial_capital,
            "final_capital": round(float(equity_curve.iloc[-1]), 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": sharpe,
            "win_rate_pct": round(win_rate, 1),
            "trades_count": int(pos_change.sum())
        }
