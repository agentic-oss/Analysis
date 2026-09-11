# Data Sources & Ingestion Specifications

## Overview

The ingestion layer uses a resilient `MarketDataProvider` abstraction interface (`src/ingestion/provider.py`) with primary Yahoo Finance provider (`YFinanceProvider`) and deterministic fallback provider (`SyntheticProvider`).

## Instruments Monitored

| Symbol | Category | Market / Ticker | Unit | Currency |
| :--- | :--- | :--- | :--- | :--- |
| **GOLD** | Precious Metal | Spot / `GC=F` | USD/oz | USD |
| **SILVER** | Precious Metal | Spot / `SI=F` | USD/oz | USD |
| **GOLD_FUTURES** | Precious Metal | COMEX Futures / `GC=F` | USD/oz | USD |
| **SILVER_FUTURES** | Precious Metal | COMEX Futures / `SI=F` | USD/oz | USD |
| **GOLD_INR** | Precious Metal | Spot INR Derived | INR/10g | INR |
| **SILVER_INR** | Precious Metal | Spot INR Derived | INR/kg | INR |
| **MCX_GOLD** | Precious Metal | MCX Futures / Gold BEES (`GOLDBEES.NS`) | INR/10g | INR |
| **MCX_SILVER** | Precious Metal | MCX Futures / Silver BEES (`SILVERBEES.NS`) | INR/kg | INR |
| **USDINR** | Currency | FX / `USDINR=X` | FX Rate | INR |
| **DXY** | Currency | Index / `DX-Y.NYB` | Index | USD |
| **US10Y** | Rates | 10Y Yield / `^TNX` | % Yield | USD |
| **US2Y** | Rates | 2Y Yield Proxy / `^IRX` | % Yield | USD |
| **REAL_YIELD** | Rates | Real Yield Proxy / `TIP` | ETF Price/Yield | USD |
| **CRUDE_OIL** | Commodities | WTI Crude / `CL=F` | USD/bbl | USD |
| **SP500** | Equities | S&P 500 / `^GSPC` | Index | USD |
| **NIFTY50** | Equities | NIFTY 50 / `^NSEI` | Index | INR |
| **VIX** | Volatility | CBOE VIX / `^VIX` | Index | USD |
| **INDIA_VIX** | Volatility | India VIX / `^INDIAVIX` | Index | INR |

## Fault Tolerance & Metadata Record

Every observation record stores metadata fields:
* `provider`
* `retrieved_at` (ISO 8601 UTC timestamp)
* `symbol`
* `currency`
* `unit`
* `market`
