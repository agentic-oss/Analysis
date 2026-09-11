# Technical Analysis Specifications

Calculated technical metrics (`src/indicators/technical.py`):
* **Moving Averages:**
  * SMA: 5, 10, 20, 50, 100, 200 days.
  * EMA: 9, 20, 50, 200 days.
* **Momentum:**
  * RSI (14)
  * Stochastic RSI (14)
  * MACD (12, 26, 9)
  * Rate of Change (ROC 12)
* **Volatility:**
  * ATR (14)
  * 20-Day Annualized Historical Volatility
  * Bollinger Bands (20-day, 2.0 std dev) & Band Width
* **Price Structure:**
  * Percentage distances to 20, 50, 200 SMAs and 52-week High/Low.
  * Golden Cross & Death Cross flags.
  * 20-day Breakout and Breakdown flags.
