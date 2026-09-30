# Backtesting Engine

`src/backtesting/backtest_engine.py` provides event-driven research backtesting with:

- Transaction costs (default 5 bps)
- Slippage model (default 2 bps)
- Position execution on next-day open/close (1-day signal lag)
- Metrics: CAGR, Maximum Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, Profit Factor.
