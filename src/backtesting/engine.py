import pandas as pd
import numpy as np
from typing import Dict, Any, List

class BacktestEngine:
    """Reusable backtesting framework accounting for transaction costs, slippage, and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_position_size: float = 0.25
    ):
        self.initial_capital = initial_capital
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0
        self.max_position_size = max_position_size

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = "composite_score",
        buy_threshold: float = 60.0,
        sell_threshold: float = 40.0
    ) -> Dict[str, Any]:
        """Run backtest on daily price series using a score or signal column."""
        if df.empty or len(df) < 5 or signal_col not in df.columns:
            return {"status": "insufficient_data"}

        b_df = df.sort_values("date").reset_index(drop=True).copy()
        prices = b_df["price"].values if "price" in b_df.columns else b_df["close"].values
        signals = b_df[signal_col].values

        cash = self.initial_capital
        position_units = 0.0
        portfolio_values = []
        trades = []

        for i in range(len(b_df)):
            p = prices[i]
            s = signals[i]

            # Signal logic
            if s >= buy_threshold and position_units == 0.0:
                # Buy entry
                total_portfolio = cash
                alloc = total_portfolio * self.max_position_size
                units = (alloc * (1.0 - self.cost_pct)) / p
                cash_spent = units * p * (1.0 + self.cost_pct)
                cash -= cash_spent
                position_units = units
                trades.append({"date": b_df.iloc[i]["date"], "type": "BUY", "price": float(p), "units": float(units)})

            elif s <= sell_threshold and position_units > 0.0:
                # Sell exit
                proceeds = position_units * p * (1.0 - self.cost_pct)
                cash += proceeds
                trades.append({"date": b_df.iloc[i]["date"], "type": "SELL", "price": float(p), "proceeds": float(proceeds)})
                position_units = 0.0

            # Portfolio total valuation at close of bar
            current_portfolio_val = cash + (position_units * p)
            portfolio_values.append(current_portfolio_val)

        pv = np.array(portfolio_values)
        pv_returns = np.diff(pv) / pv[:-1]

        total_return = (pv[-1] - self.initial_capital) / self.initial_capital
        cum_max = np.maximum.accumulate(pv)
        drawdowns = (pv - cum_max) / cum_max
        max_drawdown = float(np.min(drawdowns))

        sharpe = float((np.mean(pv_returns) / np.std(pv_returns)) * np.sqrt(252)) if len(pv_returns) > 1 and np.std(pv_returns) > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(float(pv[-1]), 2),
            "total_return_pct": round(total_return * 100.0, 2),
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
            "sharpe_ratio": round(sharpe, 2),
            "trade_count": len(trades),
            "trades": trades
        }
