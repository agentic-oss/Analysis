# Data Sources

Primary data provider interface: `MarketDataProvider`.

Supported Providers:
1. **YFinanceProvider**: Uses Yahoo Finance for historical spot, futures, fx, index, rate, commodity, equity, and volatility data.
2. **SyntheticProvider**: Deterministic simulated data provider for offline testing and environments without external network access.

## Supported Instruments

Configured in `config/instruments.json`:
- **Precious Metals**: Spot Gold (`GC=F`), Spot Silver (`SI=F`), Gold Futures, Silver Futures, Indian Gold INR (10g), Indian Silver INR (kg), MCX Gold (`GOLDBEES.NS`), MCX Silver (`SILVERBEES.NS`).
- **Currency**: USD/INR (`USDINR=X`), US Dollar Index DXY (`DX-Y.NYB`).
- **Rates**: US 10Y Yield (`^TNX`), US 2Y Yield (`2YY=F`), Real Yield (`TIP`).
- **Energy**: Crude Oil (`CL=F`).
- **Equities**: S&P 500 (`^GSPC`), NIFTY 50 (`^NSEI`).
- **Volatility**: VIX (`^VIX`), India VIX (`^INDIAVIX`).
