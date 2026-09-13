# Methodology Documentation

## 1. Indian-Market Price Conversion Methodology
- **Gold**: USD/oz converted to INR/10g = `(Price_USD / 31.1034768) * USDINR_Rate * 10`
- **Silver**: USD/oz converted to INR/kg = `(Price_USD / 31.1034768) * USDINR_Rate * 1000`

## 2. Historical Analogue Search Methodology
- Standardizes feature vector $V_T = [RSI, MACD, Dist20, Dist50, Dist200, Vol, RatioZ, DXY20, US10Y20, VIX]$.
- Calculates similarity score $S_t = \frac{1}{1 + \|V_T - V_t\|}$ for all $t \le T - 65$ business days.
- Extracts forward return distribution ($1D, 3D, 5D, 10D, 20D, 60D$) across top $N$ analogues.

## 3. Look-Ahead Bias Prevention
- Features at time $T$ are constructed using information available up to and including $T$.
- Analogue candidates are restricted to historical observations prior to $T - H$, ensuring future data is never used during matching.
- Backtesting trades are executed on $T+1$ based on signals at $T$.

## 4. Probabilistic Research Signals
- All predictions are expressed as conditional historical probabilities $P(\text{Return} > 0 \mid \text{Analogue State})$, rather than deterministic price claims.
