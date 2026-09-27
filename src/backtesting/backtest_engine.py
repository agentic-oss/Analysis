import pandas as pd
import numpy as np
from typing import Dict, Any, List

class BacktestingEngine:
    """Event-driven signal backtester accounting for transaction costs, slippage, and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str,
        price_col: str = "GOLD_close"
    ) -> Dict[str, Any]:
        """
        Executes backtest based on position signals (-1 = Short, 0 = Cash, 1 = Long).
        Tracks portfolio equity, trades, drawdowns, and returns performance statistics.
        Rebalancing trades are executed only when signal changes.
        """
        data = df.dropna(subset=[price_col, signal_col]).copy().reset_index(drop=True)
        if len(data) < 2:
            return {"error": "Insufficient data for backtest"}

        capital = self.initial_capital
        position = 0.0  # current position units
        current_signal = 0
        equity_curve = []
        trades = []

        prices = data[price_col].astype(float).values
        signals = data[signal_col].astype(int).values
        dates = data["date"].values

        cash = capital

        for i in range(len(data)):
            p = prices[i]
            sig = signals[i]
            d = dates[i]

            # Rebalance only when signal changes
            if sig != current_signal or i == 0:
                current_pos_val = position * p
                total_portfolio_val = cash + current_pos_val

                if sig == 1:
                    target_pos_val = total_portfolio_val
                elif sig == -1:
                    target_pos_val = -total_portfolio_val
                else:
                    target_pos_val = 0.0

                val_diff = target_pos_val - current_pos_val

                if abs(val_diff) > 1e-4:
                    units_to_trade = val_diff / p
                    cost = abs(val_diff) * (self.transaction_cost_pct + self.slippage_pct)
                    cash -= (units_to_trade * p + cost)
                    position += units_to_trade
                    current_signal = sig
                    trades.append({
                        "date": str(d),
                        "price": float(p),
                        "trade_units": float(units_to_trade),
                        "cost": float(cost)
                    })

            total_equity = cash + position * p
            equity_curve.append(total_equity)

        equity_arr = np.array(equity_curve)
        daily_returns = np.diff(equity_arr) / equity_arr[:-1]

        cum_return = (equity_arr[-1] - self.initial_capital) / self.initial_capital
        ann_return = ((1 + cum_return) ** (252.0 / len(equity_arr))) - 1.0 if len(equity_arr) > 0 else 0.0
        ann_vol = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 0 else 0.0

        sharpe = (ann_return - 0.04) / ann_vol if ann_vol > 0 else 0.0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - peak) / peak
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        win_trades = sum(1 for t in trades if t["trade_units"] * (prices[-1] - t["price"]) > 0)
        win_rate = win_trades / len(trades) if trades else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_equity": float(equity_arr[-1]),
            "cumulative_return": float(cum_return),
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": max_drawdown,
            "total_trades": len(trades),
            "win_rate": float(win_rate)
        }
