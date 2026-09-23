# Backtesting Engine

The backtester executes trading strategies on signal inputs:
- Shifts signals by 1 period to prevent look-ahead execution.
- Accounts for transaction costs and slippage (configurable, default 5 bps each).
- Computes cumulative return, annualized return, max drawdown, Sharpe ratio, Sortino ratio, and win rate.
