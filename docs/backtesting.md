# Backtesting Framework

The system includes a modular backtesting engine (`src/backtesting/engine.py`) for evaluating trading strategies derived from probabilistic signals.

## Key Features

- **Realistic Frictions**: Incorporates configurable transaction costs (default `0.1%`) and execution slippage (default `0.05%`).
- **Position Sizing Rules**: Enforces maximum position allocation constraints (e.g., max 20% capital per position).
- **Execution Timing**: Prevents look-ahead bias by evaluating signals at date T close and executing at date T execution price or date T+1 open.

## Performance Metrics Calculated

- **Cumulative Return**: Total percentage gain/loss over initial capital.
- **Annualized Return**: Compounded annual growth rate (CAGR).
- **Sharpe Ratio**: Risk-adjusted excess return per unit of volatility:
  $$\text{Sharpe} = \frac{\mu_{ret}}{\sigma_{ret}} \times \sqrt{252}$$
- **Maximum Drawdown**: Largest peak-to-trough decline during the backtest period.
- **Win Rate**: Percentage of profitable trades.
