# Macro Analysis & Regime Classification

## Macro Cross-Market Relationships

Precious metals pricing is deeply tied to cross-market dynamics:

1. **US Dollar Index (DXY)**: Inversely correlated with dollar-denominated gold/silver.
2. **Real Yields (TIP / US 10Y)**: Opportunity cost driver for holding non-yielding precious metals.
3. **Crude Oil**: Energy driver influencing headline inflation expectations.
4. **Equity Volatility (VIX / India VIX)**: Risk-off haven demand trigger.

## Macro Regime Framework

Rule-based composite macro regime detector evaluates 20-day returns and changes across key assets:

* **Stagflationary**: Oil returns > +5% AND Equity returns < -2%
* **Inflationary**: Oil returns > +5% AND Equity returns >= 0%
* **Risk-Off**: VIX > 25 OR Equity returns < -4%
* **Risk-On**: Equity returns > +3% AND VIX < 18
* **Tightening**: 10Y Yields rise > +0.3 pts
* **Easing**: 10Y Yields fall < -0.3 pts
* **Disinflationary**: Oil returns < -5% AND Equity returns < 0%
* **Neutral**: Default baseline
