# Data Sources & Providers

## Supported Instruments

Configured in `config/instruments.json`:

| Symbol | Market | Currency | Unit | Provider Ticker |
|---|---|---|---|---|
| GOLD | Spot / Futures | USD | oz | `GC=F` |
| SILVER | Spot / Futures | USD | oz | `SI=F` |
| GOLD_INR | Domestic Indian | INR | 10g | `GOLDBEES.NS` |
| SILVER_INR | Domestic Indian | INR | kg | `SILVERBEES.NS` |
| USDINR | Currency FX | INR | rate | `USDINR=X` |
| DXY | Index | USD | index | `DX-Y.NYB` |
| US10Y | Rates | USD | percent | `^TNX` |
| US2Y | Rates | USD | percent | `2YY=F` |
| CRUDE_OIL | Energy | USD | barrel | `CL=F` |
| SP500 | Equity Index | USD | index | `^GSPC` |
| NIFTY50 | Equity Index | INR | index | `^NSEI` |
| VIX | Volatility | USD | index | `^VIX` |
| INDIA_VIX | Volatility | INR | index | `^INDIAVIX` |

## Retry & Error Handling Strategy

The `YFinanceMarketDataProvider` implements exponential backoff retries with configurable timeouts (3 retries, backoff factor 1.5). On persistent failure, the pipeline logs errors and falls back to deterministic mock providers to ensure pipeline continuity.
