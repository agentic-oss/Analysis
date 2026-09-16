# Backtesting & Performance Evaluation

## Backtest Engine
Simulates trading strategy performance given research signals. Features:
- Models transaction costs (default: 10 bps) and slippage (default: 5 bps).
- Computes Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Win Rate, and Profit Factor.

## Forecast Evaluation & Model Monitoring
- Logs daily predictions to `data/predictions/daily/`.
- Evaluates actual outcomes once prediction horizons elapse.
- Tracks directional accuracy, MAE, and RMSE over time, logging metrics to `data/models/model-performance.json`.
