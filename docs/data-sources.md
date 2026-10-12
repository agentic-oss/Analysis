# Data Sources & Providers

* **Yahoo Finance (`YahooMarketDataProvider`)**: Primary provider for historical OHLCV series for Gold futures (`GC=F`), Silver futures (`SI=F`), USD/INR (`INR=X`), DXY (`DX-Y.NYB`), US10Y (`^TNX`), Crude Oil (`CL=F`), S&P 500 (`^GSPC`), NIFTY 50 (`^NSEI`), and VIX (`^VIX`).
* **Synthetic Fallback Provider (`SyntheticMarketDataProvider`)**: Deterministic synthetic generator used for offline backfilling, unit testing, and isolated CI runner environments.
