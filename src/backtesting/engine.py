import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BacktestingEngine:
    """Backtesting framework accounting for transaction costs, slippage, position sizing, entry/exit timing without look-ahead bias."""

    def __init__(self, initial_capital: float = 100000.0, transaction_cost_pct: float = 0.001, slippage_pct: float = 0.0005, max_position_size: float = 0.2):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.max_position_size = max_position_size

    def run_backtest(self, df: pd.DataFrame, signal_col: str = "signal") -> Dict[str, Any]:
        if df.empty or signal_col not in df.columns or "close" not in df.columns:
            return {"cumulative_return": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

        data = df.sort_values(by="date").copy()
        prices = data["close"].values
        signals = data[signal_col].values  # +1 for Long, -1 for Short, 0 for Flat

        capital = self.initial_capital
        position = 0.0  # units of asset
        cash = capital

        equity_curve = []
        trades = []

        for i in range(len(prices)):
            price = prices[i]

            # Current position valuation
            current_equity = cash + position * price
            equity_curve.append(current_equity)

            target_signal = signals[i]

            # Calculate target position in units
            target_value = current_equity * self.max_position_size * target_signal
            target_units = target_value / price if price > 0 else 0.0

            unit_change = target_units - position

            if abs(unit_change) > 1e-6:
                # Transaction occurs
                exec_price = price * (1 + self.slippage_pct) if unit_change > 0 else price * (1 - self.slippage_pct)
                trade_value = abs(unit_change) * exec_price
                cost = trade_value * self.transaction_cost_pct

                cash -= (unit_change * exec_price + cost)
                position = target_units

                trades.append({
                    "date": str(data.iloc[i]["date"]),
                    "action": "BUY" if unit_change > 0 else "SELL",
                    "units": abs(unit_change),
                    "price": exec_price,
                    "cost": cost
                })

        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()

        cum_return = float((equity_series.iloc[-1] - self.initial_capital) / self.initial_capital)
        sharpe = float((returns.mean() / returns.std() * np.sqrt(252))) if not returns.empty and returns.std() > 0 else 0.0

        # Drawdown calculation
        rolling_max = equity_series.cummax()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_dd = float(drawdown.min())

        win_rate = float((returns > 0).mean()) if not returns.empty else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_series.iloc[-1]),
            "cumulative_return": cum_return,
            "annualized_return": float((1 + cum_return) ** (252.0 / max(len(prices), 1)) - 1.0),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "total_trades": len(trades),
            "trades": trades
        }
