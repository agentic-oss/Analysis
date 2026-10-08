"""
Reusable Backtesting Engine for Research Signals.
Includes transaction costs, slippage, and position sizing.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List


class BacktestEngine:
    """
    Simulates trading strategy performance based on generated research signals.
    """

    @staticmethod
    def run_signal_backtest(
        df: pd.DataFrame,
        signal_col: str = "composite_score",
        buy_threshold: float = 60.0,
        sell_threshold: float = 40.0,
        initial_capital: float = 100000.0,
        commission_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Executes daily backtest on price history with signal rules.
        Trades execute on signal close; position held earns returns on subsequent days.
        """
        if df.empty or "close" not in df.columns or signal_col not in df.columns:
            return {}

        data = df.sort_values("date").reset_index(drop=True).copy()
        capital = initial_capital
        position = 0.0  # 1.0 = long, 0.0 = cash
        equity_curve = [capital]
        trades = []

        cost_factor = (commission_bps + slippage_bps) / 10000.0

        for i in range(len(data)):
            row = data.iloc[i]
            close = float(row["close"])
            sig = float(row[signal_col]) if not pd.isna(row[signal_col]) else 50.0

            # 1. Earn return for holding position from i-1 to i
            if i > 0:
                prev_close = float(data.iloc[i - 1]["close"])
                daily_ret = (close / prev_close - 1.0) if position == 1.0 else 0.0
                capital *= (1.0 + daily_ret)

            # 2. Process signal on day i close to update position for next day
            if sig >= buy_threshold and position == 0.0:
                position = 1.0
                capital -= capital * cost_factor
                trades.append({"date": row["date"], "type": "BUY", "price": close})
            elif sig <= sell_threshold and position == 1.0:
                position = 0.0
                capital -= capital * cost_factor
                trades.append({"date": row["date"], "type": "SELL", "price": close})

            if i > 0:
                equity_curve.append(capital)

        data["equity"] = equity_curve
        returns = pd.Series(equity_curve).pct_change().dropna()

        cum_return = (capital - initial_capital) / initial_capital
        ann_return = ((1.0 + cum_return) ** (252.0 / len(data))) - 1.0 if len(data) > 0 else 0.0

        eq_series = pd.Series(equity_curve)
        cummax = eq_series.cummax()
        dd = (eq_series - cummax) / cummax
        max_dd = float(dd.min())

        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252)) if returns.std() > 0 else 0.0

        return {
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "cumulative_return_pct": round(cum_return * 100.0, 2),
            "annualized_return_pct": round(ann_return * 100.0, 2),
            "max_drawdown_pct": round(max_dd * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": len(trades),
        }
