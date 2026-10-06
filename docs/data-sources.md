# Data Sources & Ingestion Layer

The system uses a modular data ingestion abstraction layer (`src/ingestion/provider.py`) that decouples downstream analytics from specific market data providers.

## Abstraction Layer (`MarketDataProvider`)

All data providers inherit from `MarketDataProvider` and implement standard interfaces:

- `get_spot_prices(symbol, start_date, end_date)`
- `get_historical_prices(ticker, start_date, end_date)`
- `get_futures_prices(symbol, start_date, end_date)`
- `get_currency_prices(currency_pair, start_date, end_date)`
- `get_macro_data(series_id, start_date, end_date)`
- `get_economic_indicators(start_date, end_date)`

## Ingested Categories & Symbols

| Category | Instrument Symbol | Exchange / Source | Unit | Currency |
|---|---|---|---|---|
| Spot Gold | `GOLD` | LBMA / Spot | oz | USD |
| Spot Silver | `SILVER` | LBMA / Spot | oz | USD |
| Gold Futures | `GOLD_FUT` | COMEX | oz | USD |
| Silver Futures | `SILVER_FUT` | COMEX | oz | USD |
| Indian Gold Spot | `GOLD_INR` | IBJA | 10g | INR |
| Indian Silver Spot | `SILVER_INR` | IBJA | kg | INR |
| MCX Gold | `MCX_GOLD` | MCX | 10g | INR |
| MCX Silver | `MCX_SILVER` | MCX | kg | INR |
| Currency Pair | `USD/INR` | FX Market | ratio | INR |
| Dollar Index | `DXY` | ICE | index | USD |
| Interest Rates | `US10Y`, `US2Y` | US Treasury | pct | USD |
| Commodities | `OIL_WTI`, `OIL_BRENT` | NYMEX / ICE | bbl | USD |
| Equities | `SP500`, `NASDAQ`, `NIFTY50` | S&P, Nasdaq, NSE | index | USD / INR |
| Volatility | `VIX`, `INDIA_VIX` | CBOE, NSE | index | USD / INR |

## Metadata Enrichment

Every observation ingested by `DataIngestionPipeline` is enriched with standard provenance fields:

- `provider`: Data source name (e.g. `yfinance`)
- `retrieval_timestamp`: ISO-8601 UTC timestamp
- `instrument`: Standardized instrument identifier
- `market`: Market classification (`spot`, `futures`, `fx`, `rates`, `equity`, `volatility`)
- `currency`: Price currency (`USD`, `INR`)
- `unit`: Quantity unit (`oz`, `10g`, `kg`, `pct`, `index`)
- `exchange`: Originating exchange or benchmark provider
- `timezone`: Time zone standard (`UTC`)
