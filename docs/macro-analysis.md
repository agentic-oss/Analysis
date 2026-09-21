# Macro Analysis & Cross-Market Drivers

Precious metals are heavily influenced by global macroeconomic conditions, monetary policy, currency movements, and equity volatility.

## Inter-Market Drivers Tracked

1. **US Dollar Index (DXY)**: Inversely correlated with USD-denominated gold/silver.
2. **US 10Y Treasury Yield (`^TNX`) & 2Y Yield (`^IRX`)**: Opportunity cost of holding non-yielding metals.
3. **Crude Oil (WTI & Brent)**: Energy inflation benchmark.
4. **Equities (S&P 500 & NIFTY 50)**: Benchmark equity performance and risk appetite.
5. **Volatility (VIX & India VIX)**: Safe-haven demand proxy during equity panics.

## Rolling Correlations

The system calculates rolling correlations across 20, 60, 120, and 252-day windows.
Sign-flips or significant correlation shifts (> 0.4 change in 60d correlation) trigger regime shift notifications.

## Macro Regime Rules

- **Inflationary**: Commodity strength + persistent dollar weakness.
- **Disinflationary**: Easing oil/commodity prices + moderating yield pressures.
- **Risk-off**: High VIX (> 22) or equity drawdowns (> 3% in 20d).
- **Risk-on**: Low VIX (< 16) + rising equity indexes.
- **Stagflationary**: High oil/inflation + tightening rates + negative equity returns.
