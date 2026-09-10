# Technical Analysis Specifications

## Technical Indicators

- **Simple Moving Averages (SMA):** 5, 10, 20, 50, 100, 200 days
- **Exponential Moving Averages (EMA):** 9, 20, 50, 200 days
- **Relative Strength Index (RSI):** 14-period standard wilder RSI
- **Stochastic RSI:** 14-period normalized RSI range $[0, 1]$
- **Moving Average Convergence Divergence (MACD):** Fast=12, Slow=26, Signal=9
- **Average True Range (ATR):** 14-period true range average
- **Historical Volatility:** 20-day and 60-day annualized standard deviation of daily log/pct returns ($\sigma_{20d} \times \sqrt{252}$)
- **Bollinger Bands:** 20-period SMA $\pm 2 \sigma$, along with Band Width
- **On-Balance Volume (OBV):** Cumulative volume directional sum
- **Market Structure Signals:** Golden Cross (SMA 50 crossing above SMA 200), Death Cross (SMA 50 crossing below SMA 200), 20D Breakouts/Breakdowns, and 52-week High/Low distance percentages.
