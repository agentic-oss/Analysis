# Macro Analysis & Regime Detection

## Macro Components Tracked
- **DXY**: US Dollar Index level and % change.
- **US Treasury Yields**: 10-year yield (`^TNX`) and 2-year yield (`^IRX`).
- **Yield Curve**: Inversion check (`US10Y - US2Y < 0`).
- **Energy**: WTI Crude Oil (`CL=F`) daily % move.
- **Equities**: S&P 500 (`^GSPC`) and NIFTY 50 (`^NSEI`).
- **Volatility**: VIX (`^VIX`) and India VIX (`^INDIAVIX`).

## Macro Regime Model
Classifies macro state into:
- **Inflationary**: High oil price momentum & rising yields.
- **Risk-Off**: Elevated VIX (>22), equity market declines, rising DXY.
- **Risk-On**: Subdued VIX (<16), equity gains, weakening DXY.
- **Stagflationary**: High oil gains with declining equities and rising yields.
- **Tightening / Easing / Disinflationary / Neutral**.
