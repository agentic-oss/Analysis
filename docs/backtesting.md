# Backtesting Framework

The `BacktestEngine` (`src/models/forecasting_models.py`) simulates trading strategies based on generated signals with strict look-ahead protection:
* Signals generated at close $T$ execute position on close $T$, earning return on trading day $T+1$.
* Incorporates configurable transaction costs (default: 10 bps) and slippage (default: 5 bps).
* Tracks cumulative return, annualized return, maximum drawdown, Sharpe ratio, and win rate.
