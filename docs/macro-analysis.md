# Macro Analysis & Regime Framework

## Macro Regime Engine (`src/regimes/macro_regime.py`)

Inputs evaluated over 20-day changes:
* **S&P 500 & Volatility (VIX):** Risk-On vs. Risk-Off classification.
* **10Y Treasury Yield & TIPS Real Yield:** Monetary Tightening vs. Easing.
* **Crude Oil & DXY:** Inflationary, Disinflationary, or Stagflationary environment.

## Market Regimes (`src/regimes/macro_regime.py`)

Asset-specific classification into:
* **Trend Regimes:** Strong Bullish, Bullish, Bearish, Strong Bearish, Range.
* **Volatility Regimes:** High Volatility (>22% annualized), Normal Volatility, Low Volatility (<10%).
