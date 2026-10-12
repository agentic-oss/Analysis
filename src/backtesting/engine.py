import numpy as np
import pandas as pd
from typing import Dict, Any, List

class BacktestEngine:
    """
    Event-driven backtesting engine for signal-based trading strategies.
    Accounts for transaction costs, slippage, and position sizing.
    Calculates Sharpe, Sortino, max drawdown, win rate, and profit factor.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0
    ):
        self.initial_capital = initial_capital
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_backtest(
        self,
        df: pd.DataFrame,
        signal_column: str,
        price_column: str = "close"
    ) -> Dict[str, Any]:
        """
        Executes backtest where signal_column is 1 (long), -1 (short), or 0 (cash).
        Calculates realistic portfolio equity curve including trading costs.
        """
        if df.empty or signal_column not in df.columns:
            return {"status": "error", "message": "Invalid dataframe or missing signal column"}

        data = df.copy().reset_index(drop=True)
        prices = data[price_column].values
        signals = data[signal_column].shift(1).fillna(0).values # Execution on open/next close (no look-ahead)

        capital = self.initial_capital
        equity_curve = [capital]
        positions = [0]
        trades = []

        current_pos = 0

        for i in range(1, len(data)):
            sig = signals[i]
            p_prev = prices[i - 1]
            p_curr = prices[i]

            # Rebalance if position changed
            if sig != current_pos:
                # Cost incurred on rebalance
                cost = capital * self.cost_pct
                capital -= cost
                trades.append({
                    "date": str(data.loc[i, "date"]),
                    "from_pos": current_pos,
                    "to_pos": sig,
                    "cost": cost
                })
                current_pos = sig

            # Calculate daily return from holding current_pos
            if current_pos != 0:
                ret = (p_curr - p_prev) / p_prev
                capital += capital * (current_pos * ret)

            equity_curve.append(capital)
            positions.append(current_pos)

        eq_series = pd.Series(equity_curve)
        daily_rets = eq_series.pct_change().dropna()

        # Metrics calculation
        cum_return = (capital - self.initial_capital) / self.initial_capital
        n_days = len(data)
        ann_return = ((1 + cum_return) ** (252.0 / max(1, n_days))) - 1.0 if cum_return > -1 else -1.0

        std_dev = daily_rets.std() * np.sqrt(252)
        sharpe = (ann_return / std_dev) if std_dev > 0 else 0.0

        downside_std = daily_rets[daily_rets < 0].std() * np.sqrt(252)
        sortino = (ann_return / downside_std) if downside_std > 0 else 0.0

        # Drawdown
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        max_dd = float(dd.min())

        # Trade stats
        winning_days = (daily_rets > 0).sum()
        total_active_days = (daily_rets != 0).sum()
        win_rate = (winning_days / total_active_days * 100.0) if total_active_days > 0 else 0.0

        gross_gains = daily_rets[daily_rets > 0].sum()
        gross_losses = abs(daily_rets[daily_rets < 0].sum())
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else 1.0

        return {
            "status": "success",
            "initial_capital": self.initial_capital,
            "final_capital": round(capital, 2),
            "cumulative_return_pct": round(cum_return * 100.0, 2),
            "annualized_return_pct": round(ann_return * 100.0, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "win_rate_pct": round(float(win_rate), 1),
            "profit_factor": round(float(profit_factor), 2),
            "total_trades": len(trades)
        }
