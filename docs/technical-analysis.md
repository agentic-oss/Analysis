# Technical Analysis Suite

The platform computes a standardized suite of technical indicators for Gold and Silver:

## Moving Averages
- **Simple Moving Averages (SMA)**: 5, 10, 20, 50, 100, 200 days.
- **Exponential Moving Averages (EMA)**: 9, 20, 50, 200 days.

## Momentum Indicators
- **Relative Strength Index (RSI 14)**: 14-day gain/loss ratio.
- **Stochastic RSI**: 14-day normalized position of RSI between its 14-day min and max.
- **MACD**: 12-day EMA minus 26-day EMA; Signal line (9-day EMA); Histogram.
- **Rate of Change (ROC 12)**: 12-day percentage price change.
- **Momentum 10**: Price minus 10-day prior price.

## Volatility
- **Average True Range (ATR 14)**: 14-day rolling true range.
- **Historical Volatility**: 20d, 60d, 252d annualized log-return standard deviation.
- **Bollinger Bands**: 20-day SMA +/- 2.0 standard deviations; Bollinger Band Width.

## Volume & Market Structure
- **Relative Volume**: Daily volume divided by 20-day volume SMA.
- **On-Balance Volume (OBV)**: Cumulative directional volume flow.
- **Distances**: Percentage distance from 20d, 50d, 200d MAs, 52-week High, 52-week Low.
- **Structural Signals**: Golden Cross, Death Cross, Breakout, Breakdown, 52-week Highs/Lows, Volatility Expansion/Contraction.
