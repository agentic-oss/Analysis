# Backtesting Methodology

## Backtest Assumptions & Rules

- **Execution Timing**: Signals generated at close of Day T are executed at Open/Close of Day T+1.
- **Transaction Costs & Slippage**: Configured by default at 5 bps (0.05%) transaction fee + 5 bps slippage per trade.
- **Performance Metrics**: Annualized Return, Cumulative Equity, Maximum Drawdown, Sharpe Ratio, Win Rate.
