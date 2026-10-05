"""Reusable strategy backtesting framework with transaction costs and slippage."""

from typing import Dict, Any
import numpy as np
import pandas as pd


class BacktestEngine:
    """Backtests signals with transaction costs, slippage, and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0
    ):
        self.initial_capital = initial_capital
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        symbol: str = "GOLD"
    ) -> Dict[str, Any]:
        """
        Runs backtest where signal is +1 (Long), -1 (Short), or 0 (Cash).
        """
        if df.empty or signal_col not in df.columns:
            return {}

        df = df.sort_values('date').copy()
        prices = df['close'].values
        signals = df[signal_col].values
        dates = df['date'].values

        n = len(df)
        equity = np.zeros(n)
        equity[0] = self.initial_capital
        position = 0

        trades = 0
        wins = 0

        for i in range(1, n):
            target_pos = signals[i - 1] # Signal on T enters on T+1 open/close
            price_change = (prices[i] - prices[i - 1]) / prices[i - 1]

            # Check trade execution cost if position changes
            cost = 0.0
            if target_pos != position:
                cost = self.cost_pct
                trades += 1

            ret = (target_pos * price_change) - cost
            equity[i] = equity[i - 1] * (1.0 + ret)

            if ret > 0:
                wins += 1

            position = target_pos

        equity_series = pd.Series(equity)
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        daily_returns = equity_series.pct_change().dropna()

        ann_return = float(daily_returns.mean() * 252)
        ann_vol = float(daily_returns.std() * np.sqrt(252))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0

        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = float(drawdown.min())

        win_rate = float(wins / trades) if trades > 0 else 0.0

        return {
            "symbol": symbol,
            "initial_capital": self.initial_capital,
            "final_capital": round(float(equity[-1]), 2),
            "total_return_pct": round(total_return * 100.0, 2),
            "annualized_return_pct": round(ann_return * 100.0, 2),
            "annualized_volatility_pct": round(ann_vol * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "total_trades": trades,
            "win_rate": round(win_rate, 4)
        }
