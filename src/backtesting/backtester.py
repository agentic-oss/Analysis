import numpy as np
import pandas as pd
from typing import Dict, Any, List


class SignalBacktester:
    """
    Backtests research signal trading strategies accounting for transaction costs,
    slippage, and position sizing.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.total_cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str,
        holding_period: int = 5,
    ) -> Dict[str, Any]:
        """
        Runs backtest given a DataFrame containing prices and signals (-1, 0, 1 or continuous score).
        """
        if df is None or df.empty or signal_col not in df.columns or "Close" not in df.columns:
            return {"error": "Invalid input DataFrame for backtest"}

        df = df.copy().reset_index(drop=True)
        prices = df["Close"].values
        signals = df[signal_col].values
        n = len(prices)

        portfolio_value = np.zeros(n)
        portfolio_value[0] = self.initial_capital
        cash = self.initial_capital
        position = 0.0 # units held
        trades_count = 0
        total_costs_paid = 0.0
        trade_returns = []

        entry_price = 0.0

        for i in range(1, n):
            sig = signals[i - 1] # Signal from previous day close (no lookahead)
            curr_price = prices[i]

            # Standardize signal to direction (-1, 0, +1)
            if sig > 60.0:
                target_dir = 1.0
            elif sig < 40.0:
                target_dir = -1.0
            else:
                target_dir = 0.0

            curr_dir = np.sign(position)

            if target_dir != curr_dir:
                # Close existing position
                if position != 0.0:
                    trade_val = abs(position * curr_price)
                    cost = trade_val * self.total_cost_pct
                    total_costs_paid += cost
                    cash += (position * curr_price) - cost
                    ret = (curr_price - entry_price) / entry_price if position > 0 else (entry_price - curr_price) / entry_price
                    trade_returns.append(ret)
                    position = 0.0
                    trades_count += 1

                # Open new position
                if target_dir != 0.0 and cash > 0:
                    alloc_cash = cash * 0.95 # 95% capital allocation
                    trade_val = alloc_cash
                    cost = trade_val * self.total_cost_pct
                    total_costs_paid += cost
                    position = (alloc_cash - cost) / curr_price if target_dir > 0 else -(alloc_cash - cost) / curr_price
                    cash -= alloc_cash
                    entry_price = curr_price
                    trades_count += 1

            portfolio_value[i] = max(0.0, cash + (position * curr_price))

        # Performance summary
        daily_returns = np.diff(portfolio_value) / np.maximum(1e-9, portfolio_value[:-1])
        cumulative_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital

        if 1.0 + cumulative_return > 0:
            ann_return = ((1.0 + cumulative_return) ** (252.0 / max(1, n))) - 1.0
        else:
            ann_return = -1.0

        # Drawdown calculation
        peak = np.maximum.accumulate(portfolio_value)
        peak = np.maximum(1e-9, peak)
        drawdown = (portfolio_value - peak) / peak
        max_drawdown = float(np.min(drawdown))

        # Sharpe ratio
        std_ret = np.std(daily_returns)
        sharpe = (np.mean(daily_returns) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0

        # Sortino ratio
        downside_std = np.std(daily_returns[daily_returns < 0])
        sortino = (np.mean(daily_returns) / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0

        # Win rate & profit factor
        if trade_returns:
            pos_trades = [r for r in trade_returns if r > 0]
            neg_trades = [abs(r) for r in trade_returns if r < 0]
            win_rate = len(pos_trades) / len(trade_returns)
            profit_factor = (sum(pos_trades) / sum(neg_trades)) if sum(neg_trades) > 0 else 999.0
        else:
            win_rate = 0.0
            profit_factor = 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(float(portfolio_value[-1]), 2),
            "cumulative_return_pct": round(float(cumulative_return * 100), 2),
            "annualized_return_pct": round(float(ann_return * 100), 2),
            "max_drawdown_pct": round(float(max_drawdown * 100), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "win_rate_pct": round(float(win_rate * 100), 1),
            "profit_factor": round(float(profit_factor), 2),
            "total_trades": trades_count,
            "total_costs_paid": round(float(total_costs_paid), 2),
        }
