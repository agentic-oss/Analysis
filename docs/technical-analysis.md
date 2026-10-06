# Technical Indicators Specification

The system calculates a comprehensive suite of technical indicators defined in `src/indicators/technical.py`.

## Calculated Metrics

### 1. Moving Averages
- **Simple Moving Averages (SMA)**: 5, 10, 20, 50, 100, 200 periods.
- **Exponential Moving Averages (EMA)**: 9, 20, 50, 200 periods.

### 2. Momentum & Oscillators
- **Relative Strength Index (RSI 14)**:
  $$RS = \frac{\text{EMA}(\text{Gain}, 14)}{\text{EMA}(\text{Loss}, 14)}, \quad \text{RSI} = 100 - \left(\frac{100}{1 + RS}\right)$$
- **Stochastic RSI**:
  $$\text{Stoch RSI} = \frac{\text{RSI}_{14} - \min(\text{RSI}_{14})}{\max(\text{RSI}_{14}) - \min(\text{RSI}_{14})}$$
- **Moving Average Convergence Divergence (MACD)**:
  $$\text{MACD} = \text{EMA}(12) - \text{EMA}(26), \quad \text{Signal} = \text{EMA}(\text{MACD}, 9), \quad \text{Hist} = \text{MACD} - \text{Signal}$$

### 3. Volatility Metrics
- **Average True Range (ATR 14)**: Rolling 14-day mean of True Range ($\max(H-L, |H-C_1|, |L-C_1|)$).
- **Annualized Historical Volatility (20D)**:
  $$\sigma_{20d} = \text{Std}\left(\ln\left(\frac{P_t}{P_{t-1}}\right)\right)_{20} \times \sqrt{252}$$
- **Bollinger Bands (20D, 2 Std)**: Upper, Lower, and Width.

### 4. Technical Triggers & Patterns
- **Golden Cross**: SMA 50 crosses above SMA 200.
- **Death Cross**: SMA 50 crosses below SMA 200.
- **52-Week Distance**: Percentage distance to rolling 252-day high and low.
