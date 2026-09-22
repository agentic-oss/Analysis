# Backtesting & Model Evaluation Methodology

## Backtesting Engine
The strategy backtester simulates positions based on research signals:
- **Signal Shift:** Signals from date T are applied to positions on date T+1 (eliminating look-ahead bias).
- **Friction:** Applies transaction costs (0.05%) and slippage (0.05%) per trade turn.
- **Metrics Calculated:** Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor, Annualized Return.

## Forecast Performance Tracking
Every prediction logged on date T is tracked in `data/predictions/daily/YYYY-MM-DD.json`.
When actual prices mature on date T+H, outcomes are evaluated:
- Directional accuracy
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Performance by market regime and horizon
Output persisted in `data/models/model-performance.json`.
