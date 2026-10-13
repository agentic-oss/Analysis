# Backtesting Framework Specification

## Strategy Execution Assumptions
- **Transaction Costs:** Default 0.05% (5 bps) per trade.
- **Slippage:** Default 0.05% (5 bps) per trade.
- **Signal Execution:** Position signals lagged by 1 day to execute at next close price.
- **Performance Metrics:** Cumulative return, annualized return, max drawdown, Sharpe ratio, Sortino ratio, win rate.
