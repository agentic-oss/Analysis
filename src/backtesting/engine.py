import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple


class BacktestEngine:
    """Reusable event-driven and vector backtesting engine accounting for trading friction and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 5.0,
        slippage_bps: float = 2.0,
        position_size_pct: float = 0.20
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_bps / 10000.0
        self.slippage_pct = slippage_bps / 10000.0
        self.total_friction_pct = self.transaction_cost_pct + self.slippage_pct
        self.position_size_pct = position_size_pct

    def run_signal_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = "composite_score",
        buy_threshold: float = 60.0,
        sell_threshold: float = 40.0
    ) -> Dict[str, Any]:
        if df is None or df.empty or len(df) < 10 or signal_col not in df.columns:
            return {"status": "error_insufficient_data"}

        df = df.sort_values("date").reset_index(drop=True)
        prices = df["close"].values
        signals = df[signal_col].values
        dates = df["date"].values

        capital = self.initial_capital
        position = 0  # 0: cash, 1: long
        equity_curve = [capital]
        trades = []
        entry_price = 0.0

        for i in range(1, len(df)):
            prev_sig = signals[i-1]
            curr_price = prices[i]
            prev_price = prices[i-1]

            # Strategy decision based on previous day's signal (no look-ahead)
            target_position = position
            if prev_sig >= buy_threshold:
                target_position = 1
            elif prev_sig <= sell_threshold:
                target_position = 0

            # Execute rebalance if position changes
            if target_position != position:
                # Apply transaction cost + slippage
                capital -= (capital * self.position_size_pct * self.total_friction_pct)
                if target_position == 1:
                    entry_price = curr_price
                    trades.append({"entry_date": dates[i], "entry_price": curr_price})
                elif target_position == 0 and len(trades) > 0 and "exit_price" not in trades[-1]:
                    trades[-1]["exit_date"] = dates[i]
                    trades[-1]["exit_price"] = curr_price
                    trades[-1]["pnl_pct"] = ((curr_price - entry_price) / entry_price) - self.total_friction_pct

                position = target_position

            # Mark to market return
            if position == 1:
                asset_ret = (curr_price - prev_price) / prev_price
                strat_ret = asset_ret * self.position_size_pct
                capital *= (1.0 + strat_ret)

            equity_curve.append(capital)

        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()

        # Cumulative and Annualized Return
        cum_return_pct = ((equity_curve[-1] - self.initial_capital) / self.initial_capital) * 100.0
        n_days = len(df)
        ann_return_pct = (((1.0 + cum_return_pct/100.0) ** (252.0 / max(n_days, 1))) - 1.0) * 100.0

        # Drawdown
        cum_max = equity_series.cummax()
        drawdown = (equity_series - cum_max) / cum_max
        max_drawdown_pct = float(drawdown.min()) * 100.0

        # Sharpe and Sortino
        std_ret = daily_returns.std()
        sharpe_ratio = float((daily_returns.mean() / (std_ret + 1e-10)) * np.sqrt(252)) if std_ret > 0 else 0.0

        downside_std = daily_returns[daily_returns < 0].std()
        sortino_ratio = float((daily_returns.mean() / (downside_std + 1e-10)) * np.sqrt(252)) if downside_std > 0 else 0.0

        # Trade metrics
        completed_trades = [t for t in trades if "pnl_pct" in t]
        if completed_trades:
            pnls = [t["pnl_pct"] for t in completed_trades]
            win_rate_pct = float(np.mean([1 if p > 0 else 0 for p in pnls])) * 100.0
            gross_gains = sum([p for p in pnls if p > 0])
            gross_losses = abs(sum([p for p in pnls if p < 0]))
            profit_factor = float(gross_gains / (gross_losses + 1e-10))
        else:
            win_rate_pct = 0.0
            profit_factor = 0.0

        return {
            "status": "success",
            "initial_capital": self.initial_capital,
            "final_equity": round(equity_curve[-1], 2),
            "cumulative_return_pct": round(cum_return_pct, 2),
            "annualized_return_pct": round(ann_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate_pct": round(win_rate_pct, 1),
            "profit_factor": round(profit_factor, 2),
            "total_trades_executed": len(completed_trades)
        }
