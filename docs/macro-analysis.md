# Macroeconomic Analysis Specification

## Multi-Factor Macro Regimes
The macro engine evaluates joint trends across DXY, 10-Year Treasury Yields, Crude Oil, Equities, and CBOE VIX:

- **Risk-Off:** VIX > 25.0 or S&P 500 20d return < -5% with high volatility.
- **Inflationary:** Crude Oil 20d return > +8% and US10Y Yield 20d change > +5%.
- **Stagflationary:** Crude Oil 20d return > +8% and S&P 500 20d return < -3%.
- **Deflationary:** Crude Oil 20d return < -8% and US10Y Yield 20d change < -5%.
- **Risk-On:** S&P 500 20d return > +3% with VIX < 18.0.
- **Tightening:** US10Y Yield 20d change > +5% and DXY > +2%.
- **Easing:** US10Y Yield 20d change < -5% and DXY < -2%.
