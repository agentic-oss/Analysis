import numpy as np
import pandas as pd
from typing import Dict, Any


class BacktestEngine:
    """Simulates trading strategy performance from signals while accounting for costs and slippage."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.cost_pct = (transaction_cost_bps + slippage_bps) / 10000.0

    def run_backtest(
        self,
        df: pd.DataFrame,
        signal_col: str = "signal",
        price_col: str = "close",
    ) -> Dict[str, Any]:
        """Runs daily backtest execution."""
        if df.empty or signal_col not in df.columns or price_col not in df.columns:
            return {"status": "error", "message": "Missing required columns"}

        data = df.copy().sort_values("date").reset_index(drop=True)
        data["asset_return"] = data[price_col].pct_change().fillna(0.0)

        position = 0.0
        equity = [self.initial_capital]
        trades = []
        turnover = 0.0

        for i in range(1, len(data)):
            target_pos = float(data.loc[i - 1, signal_col])  # Position entered at previous close
            current_pos = position

            # Position change
            pos_change = abs(target_pos - current_pos)
            if pos_change > 0:
                turnover += pos_change

            cost = pos_change * self.cost_pct
            ret = target_pos * data.loc[i, "asset_return"] - cost

            new_equity = equity[-1] * (1.0 + ret)
            equity.append(new_equity)

            if pos_change > 0:
                trades.append({
                    "date": data.loc[i, "date"],
                    "position": target_pos,
                    "price": data.loc[i, price_col],
                    "cost": cost,
                })

            position = target_pos

        data["equity"] = equity
        data["strategy_return"] = data["equity"].pct_change().fillna(0.0)

        # Performance metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        n_days = len(data)
        ann_return = ((1.0 + total_return) ** (252.0 / max(1, n_days))) - 1.0

        daily_std = data["strategy_return"].std()
        sharpe = (data["strategy_return"].mean() / (daily_std + 1e-8)) * np.sqrt(252)

        downside_std = data[data["strategy_return"] < 0]["strategy_return"].std()
        sortino = (data["strategy_return"].mean() / (downside_std + 1e-8)) * np.sqrt(252)

        peak = data["equity"].cummax()
        dd = (data["equity"] - peak) / peak
        max_dd = float(dd.min())

        win_returns = data[data["strategy_return"] > 0]["strategy_return"]
        loss_returns = data[data["strategy_return"] < 0]["strategy_return"]

        win_rate = float(len(win_returns) / max(1, len(win_returns) + len(loss_returns)))
        profit_factor = float(win_returns.sum() / abs(loss_returns.sum()) if abs(loss_returns.sum()) > 0 else 1.0)

        return {
            "status": "success",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity[-1], 2),
            "total_return": round(total_return, 4),
            "annualized_return": round(ann_return, 4),
            "sharpe_ratio": round(float(sharpe), 2),
            "sortino_ratio": round(float(sortino), 2),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2),
            "trades_count": len(trades),
        }
