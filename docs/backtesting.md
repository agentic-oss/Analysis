# Backtesting Framework

## Strategy Execution Rules
- Signals are generated at the end of trading day $T$ and executed on day $T+1$ to prevent look-ahead bias.
- Accounts for trading costs (default 10 bps transaction fee + 5 bps slippage).

## Performance Metrics
- **Total Return %**
- **Annualized Return %**
- **Maximum Drawdown %**
- **Sharpe Ratio**
- **Win Rate %**
- **Trade Count & Turnover**
