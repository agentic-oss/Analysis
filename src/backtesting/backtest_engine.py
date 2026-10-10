import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for research signals on precious metals.
    Accounts for transaction costs, slippage, and position sizing.
    Avoids look-ahead bias and survivorship bias.
    """

    def __init__(self, initial_capital: float = 100000.0, transaction_cost_pct: float = 0.0005, slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_backtest(self, price_series: pd.Series, signals: pd.Series) -> Dict[str, Any]:
        """
        Executes backtest given price_series and signals (-1: Short, 0: Cash, 1: Long).
        """
        if price_series.empty or len(price_series) != len(signals):
            return {"error": "Invalid price or signal series"}

        df = pd.DataFrame({"price": price_series, "signal": signals.shift(1).fillna(0)}).dropna()
        if df.empty:
            return {"error": "Empty backtest period"}

        returns = df["price"].pct_change().fillna(0.0)

        # Signal changes incur transaction costs + slippage
        trades = df["signal"].diff().fillna(0.0).abs()
        cost_penalty = trades * (self.transaction_cost_pct + self.slippage_pct)

        strategy_returns = (df["signal"] * returns) - cost_penalty
        equity_curve = self.initial_capital * (1.0 + strategy_returns).cumprod()

        total_return = float((equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital)

        # Calculate Drawdowns
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = float(drawdown.min())

        # Sharpe Ratio (annualized)
        mean_ret = strategy_returns.mean()
        std_ret = strategy_returns.std()
        sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 0 else 0.0

        # Win rate
        active_returns = strategy_returns[df["signal"] != 0]
        win_rate = float((active_returns > 0).mean()) if len(active_returns) > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return_pct": round(total_return * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "total_trades": int((trades > 0).sum())
        }
