# Backtesting Engine Specifications

The backtesting framework in `src/backtesting/`:
- **Costs**: Configurable transaction costs (0.05%) and slippage (0.05%).
- **Execution Rules**: Entry/exit timing without look-ahead bias.
- **Metrics Calculated**: Cumulative Return, Annualized Return, Max Drawdown, Sharpe Ratio, Sortino Ratio, Win Rate, Profit Factor.
