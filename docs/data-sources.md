# Data Sources & Providers

## Overview
The system collects market data across precious metals, currencies, interest rates, energy commodities, equity indices, and volatility metrics using `MarketDataProvider`.

## Instrument Mapping & Providers

| Instrument Symbol | Name | Market / Exchange | Currency | Unit | Provider Ticker |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `GOLD` | Spot Gold | COMEX / Spot | USD | oz | `GC=F` |
| `SILVER` | Spot Silver | COMEX / Spot | USD | oz | `SI=F` |
| `GOLD_FUT` | Gold Futures | COMEX Futures | USD | oz | `GC=F` |
| `SILVER_FUT` | Silver Futures | COMEX Futures | USD | oz | `SI=F` |
| `GOLD_INR` | Indian Gold Spot | Spot (India) / BEES | INR | 10g | `GOLDBEES.NS` |
| `SILVER_INR` | Indian Silver Spot | Spot (India) / BEES | INR | kg | `SILVERBEES.NS` |
| `USDINR` | USD to INR Rate | FX Market | INR | rate | `INR=X` |
| `DXY` | US Dollar Index | ICE / NYBOT | USD | index | `DX-Y.NYB` |
| `US10Y` | US 10-Yr Yield | Treasury Bond | USD | percent | `^TNX` |
| `US2Y` | US 2-Yr Yield | Treasury Bill | USD | percent | `^IRX` |
| `WTI` | WTI Crude Oil | NYMEX Futures | USD | barrel | `CL=F` |
| `SP500` | S&P 500 Index | US Equity | USD | index | `^GSPC` |
| `NIFTY50` | NIFTY 50 Index | NSE India | INR | index | `^NSEI` |
| `VIX` | Volatility Index | CBOE | USD | index | `^VIX` |
| `INDIAVIX` | India Volatility Index | NSE India | INR | index | `^INDIAVIX` |

## Resilience & Retry Strategy
All provider downloads utilize:
- Max Retries: 3 attempts
- Exponential Backoff: Base multiplier of 1.0s (1s, 2s, 4s)
- Timeout: 10 seconds per request
- Deterministic Offline Fallback: If live feeds fail or run offline in test environments, synthetic series are generated based on deterministic seed parameters to guarantee reproducibility.
