# Backtesting Engine

The event-driven backtester (`src/backtesting/engine.py`) models portfolio equity curves under custom signal strategies.
Key features:
* Execution delay (signals generated on close $T$ execute on open/close $T+1$ to avoid look-ahead bias).
* Friction modeling: Configurable transaction costs (basis points) and slippage.
* Performance metrics: Annualized return, Sharpe ratio, Sortino ratio, max drawdown, win rate, and profit factor.
