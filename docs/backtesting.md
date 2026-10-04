# Backtesting Engine Framework

## Simulation Specification

The backtesting framework (`src/backtesting/engine.py`) allows quantitative strategy testing on historical research signals or composite scores:

* **Transaction Costs**: Configurable transaction cost in bps (e.g., 10 bps)
* **Slippage**: Configurable slippage per order in bps (e.g., 5 bps)
* **Position Sizing**: Maximum position size cap (e.g., 25% of capital)
* **Performance Metrics**: Cumulative Return, Max Drawdown, Sharpe Ratio, Trade Count, Trade Logs
