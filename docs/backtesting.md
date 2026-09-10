# Backtesting Framework

## Backtest Engine Specifications

The backtest engine evaluates trading strategies based on generated research signals:

- **Signal Alignment:** Signal on date $T$ is executed on date $T+1$ (next open or close) to avoid look-ahead bias.
- **Transaction Costs & Slippage:** Configurable cost matrix in basis points (default 5.0 bps transaction cost + 2.0 bps slippage = 7.0 bps total cost per trade).
- **Position Sizing:** Position weights range in $[-1.0, 1.0]$ (Long, Short, Cash).

## Evaluated Metrics

- Cumulative Return & Annualized Return
- Annualized Volatility ($\sigma_{daily} \times \sqrt{252}$)
- Sharpe Ratio ($\frac{R_{ann}}{\sigma_{ann}}$)
- Maximum Drawdown ($\min \frac{V_t - \max_{s \le t} V_s}{\max_{s \le t} V_s}$)
- Win Rate & Trade Count
