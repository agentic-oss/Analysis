# Backtesting Framework

The platform includes a reusable signal backtester to evaluate trading strategy rules derived from multi-factor research scores.

## Key Assumptions & Realistic Frictions
- **Transaction Costs**: 5 basis points (0.05%) per trade execution.
- **Slippage**: 2 basis points (0.02%) execution slippage.
- **Position Sizing**: Configurable fraction of portfolio (default 20%).
- **Execution Timing**: Signals generated on date $T$ execute on date $T+1$ open/close to eliminate look-ahead bias.

## Calculated Performance Metrics
- **Cumulative Return (%)**
- **Annualized Return (%)**
- **Maximum Drawdown (%)**
- **Sharpe Ratio** (Annualized risk-adjusted return)
- **Sortino Ratio** (Downside risk-adjusted return)
- **Win Rate (%)**
- **Profit Factor** (Gross gains / Gross losses)
