# Technical Analysis Methodology

Technical indicators calculated in `src/indicators/`:
- **Moving Averages**: SMA (5, 10, 20, 50, 100, 200) and EMA (9, 20, 50, 200).
- **Momentum**: RSI (14), Stochastic RSI, MACD (12, 26, 9), Rate of Change (ROC 12).
- **Volatility**: ATR (14), Historical Volatility (20D annualized), Bollinger Bands (20, 2 std dev).
- **Structure**: Distance from 20/50/200 SMAs, 52-week highs and lows, gap %, intraday range %, trend strength.
- **Pattern Signals**: Golden cross, death cross, 20-day breakouts/breakdowns.
