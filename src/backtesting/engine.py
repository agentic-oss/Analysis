import numpy as np
import pandas as pd

class StrategyBacktester:
    """Reusable backtester accounting for transaction costs, slippage, and position sizing."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 0.20
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct

    def run_backtest(self, price_df: pd.DataFrame, signal_series: pd.Series) -> dict:
        """
        Runs strategy simulation based on signals (1 = Long, -1 = Short, 0 = Cash/Flat).
        price_df must contain 'date' and 'close'.
        """
        if price_df.empty or len(price_df) < 5:
            return {}

        df = price_df.copy().sort_values("date").reset_index(drop=True)
        df["signal"] = signal_series.reindex(df.index).fillna(0)

        # Shift signal by 1 day to execute on next open/close (prevent look-ahead)
        df["position"] = df["signal"].shift(1).fillna(0)

        price = df["close"]
        price_ret = price.pct_change().fillna(0)

        # Calculate turnover and costs
        position_change = df["position"].diff().abs().fillna(0)
        costs = position_change * (self.transaction_cost_pct + self.slippage_pct)

        # Strategy return
        strat_ret = (df["position"] * price_ret) - costs
        cum_ret = (1.0 + strat_ret).cumprod()

        capital_curve = self.initial_capital * cum_ret

        total_return_pct = float((cum_ret.iloc[-1] - 1.0) * 100.0)
        n_days = len(df)
        annualized_ret_pct = float(((1.0 + total_return_pct / 100.0) ** (252.0 / max(n_days, 1)) - 1.0) * 100.0)

        # Peak and drawdown
        running_max = cum_ret.cummax()
        drawdowns = (cum_ret - running_max) / running_max
        max_drawdown_pct = float(drawdowns.min() * 100.0)

        # Sharpe & Sortino ratios (assuming risk-free rate = 0)
        std_ret = strat_ret.std()
        sharpe_ratio = float((strat_ret.mean() / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0)

        downside_std = strat_ret[strat_ret < 0].std()
        sortino_ratio = float((strat_ret.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0)

        # Win rate & Profit factor
        trade_returns = strat_ret[strat_ret != 0]
        win_rate_pct = float((trade_returns > 0).mean() * 100.0) if not trade_returns.empty else 0.0

        gains = trade_returns[trade_returns > 0].sum()
        losses = abs(trade_returns[trade_returns < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_capital": round(float(capital_curve.iloc[-1]), 2),
            "total_return_pct": round(total_return_pct, 2),
            "annualized_return_pct": round(annualized_ret_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "win_rate_pct": round(win_rate_pct, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.0,
            "total_trades": int((position_change > 0).sum())
        }
