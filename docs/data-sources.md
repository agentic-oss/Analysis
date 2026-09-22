# Data Sources & Ingestion Abstraction

The pipeline utilizes the `MarketDataProvider` abstraction layer (`YFinanceDataProvider`) with:
- Retry logic and exponential backoff
- Timeout and rate-limit error handling
- Auditing and validation logging

## Ingested Instruments
- **Gold Futures & Spot:** COMEX Gold Futures (`GC=F`)
- **Silver Futures & Spot:** COMEX Silver Futures (`SI=F`)
- **Currencies:** USD/INR (`USDINR=X`), ICE US Dollar Index (`DX-Y.NYB`)
- **Interest Rates:** US 10-Year Treasury Yield (`^TNX`), US 2-Year Treasury Yield (`^IRX`)
- **Commodities:** Crude Oil (`CL=F`)
- **Equities:** S&P 500 (`^GSPC`), NIFTY 50 (`^NSEI`)
- **Indian ETFs:** MCX Gold ETF (`GOLDBEES.NS`), MCX Silver ETF (`SILVERBEES.NS`)
- **Volatility Indices:** CBOE VIX (`^VIX`), India VIX (`^INDIAVIX`)
