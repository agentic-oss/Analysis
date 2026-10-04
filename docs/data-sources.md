# Data Sources & Ingestion

## Supported Data Providers

The pipeline uses the `MarketDataProvider` abstraction, allowing seamless switching between data sources:

1. **Yahoo Finance (`yfinance`)**: Primary market data provider for global precious metals futures/spot, currency exchange rates, Treasury yields, indices, commodities, and volatility indicators.
2. **Mock Data Provider (`MockDataProvider`)**: Deterministic synthetic data generator used for testing, offline execution, and sandbox validation.

## Supported Universe

Configured via `config/instruments.json`:

* **Precious Metals**: Gold Futures (`GC=F`), Spot Gold (`XAUUSD=X`), Silver Futures (`SI=F`), Spot Silver (`XAGUSD=X`), Indian Gold ETF Proxy (`GOLDBEES.NS`), Indian Silver ETF Proxy (`SILVERBEES.NS`).
* **Currencies**: `USDINR=X`, `DX-Y.NYB` (DXY).
* **Rates**: `^TNX` (US 10Y Yield), `2YY=F` (US 2Y Yield), `TIP` (TIPS ETF / Real Yield Proxy).
* **Energy & Commodities**: `CL=F` (WTI Crude), `BZ=F` (Brent Crude), `HG=F` (Copper Futures).
* **Equities**: `^GSPC` (S&P 500), `^IXIC` (Nasdaq), `^NSEI` (NIFTY 50).
* **Volatility**: `^VIX` (CBOE VIX), `^INDIAVIX` (India VIX).

## Retry & Resilience Mechanisms

Every ingestion batch implements:
* Configurable retry counts (`retries: 3`)
* Exponential backoff delays (`backoff_factor: 1.5`)
* Timeouts (`timeout_seconds: 10`)
* Structured logging of failed retrievals
