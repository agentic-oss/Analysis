import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Backtesting engine for research strategies with transaction costs, slippage, position sizing, and performance metrics.
    Prevents look-ahead bias and calculates equity curve, Sharpe, Sortino, Max Drawdown, CAGR, and Win Rate.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost_pct: float = 0.0005,
        slippage_pct: float = 0.0002,
        risk_free_rate: float = 0.04,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct
        self.slippage_pct = slippage_pct
        self.risk_free_rate = risk_free_rate

    def run_signal_backtest(
        self, df: pd.DataFrame, signal_col: str, price_col: str = "close"
    ) -> Dict[str, Any]:
        """
        Runs signal-driven strategy backtest.
        Signal values: +1 (Long), 0 (Cash), -1 (Short)
        """
        if df.empty or len(df) < 10 or signal_col not in df.columns:
            return {"error": "Insufficient data or missing signal column."}

        bt_df = df[["date", price_col, signal_col]].copy().sort_values("date").reset_index(drop=True)
        # Shift signal by 1 day to execute on next day's open/close (avoiding look-ahead bias)
        bt_df["position"] = bt_df[signal_col].shift(1).fillna(0)

        # Calculate raw asset return
        bt_df["raw_return"] = bt_df[price_col].pct_change().fillna(0)

        # Detect position changes for transaction costs & slippage
        bt_df["pos_change"] = bt_df["position"].diff().abs().fillna(0)
        total_cost_pct = self.transaction_cost_pct + self.slippage_pct
        bt_df["costs"] = bt_df["pos_change"] * total_cost_pct

        # Strategy return
        bt_df["strategy_return"] = (bt_df["position"] * bt_df["raw_return"]) - bt_df["costs"]
        bt_df["equity_curve"] = self.initial_capital * (1.0 + bt_df["strategy_return"]).cumprod()

        # Performance summary metrics
        total_days = len(bt_df)
        total_return_pct = (bt_df["equity_curve"].iloc[-1] / self.initial_capital - 1.0) * 100.0
        cagr = ((bt_df["equity_curve"].iloc[-1] / self.initial_capital) ** (252.0 / max(total_days, 1)) - 1.0) * 100.0

        # Drawdown
        peak = bt_df["equity_curve"].cummax()
        drawdown = (bt_df["equity_curve"] - peak) / peak
        max_drawdown_pct = float(drawdown.min() * 100.0)

        # Sharpe & Sortino ratios
        daily_rf = self.risk_free_rate / 252.0
        excess_returns = bt_df["strategy_return"] - daily_rf
        std_ret = bt_df["strategy_return"].std()
        sharpe = float((excess_returns.mean() / std_ret * np.sqrt(252))) if std_ret > 0 else 0.0

        downside_std = bt_df.loc[bt_df["strategy_return"] < 0, "strategy_return"].std()
        sortino = float((excess_returns.mean() / downside_std * np.sqrt(252))) if downside_std > 0 else 0.0

        # Win rate
        active_trades = bt_df[bt_df["position"] != 0]
        win_rate = float((active_trades["strategy_return"] > 0).mean() * 100.0) if not active_trades.empty else 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_capital": float(bt_df["equity_curve"].iloc[-1]),
            "total_return_pct": round(total_return_pct, 2),
            "cagr_pct": round(cagr, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "win_rate_pct": round(win_rate, 2),
            "total_trades_count": int((bt_df["pos_change"] > 0).sum()),
        }
