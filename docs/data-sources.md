# Data Sources & Instruments

The system collects data across multiple asset classes with explicit metadata tagging:

## Instruments Supported

| Symbol | Market Category | Currency | Unit | Description |
|---|---|---|---|---|
| GOLD | Precious Metals | USD | oz | Gold Spot/Futures |
| SILVER | Precious Metals | USD | oz | Silver Spot/Futures |
| GOLD_INR | Precious Metals (Indian) | INR | 10g | Indian Gold Representation |
| SILVER_INR | Precious Metals (Indian) | INR | kg | Indian Silver Representation |
| USDINR | Currency | INR | rate | USD / INR Exchange Rate |
| DXY | Currency | USD | index | US Dollar Index |
| US10Y | Rates | USD | percent | US 10-Year Treasury Yield |
| US2Y | Rates | USD | percent | US 2-Year Treasury Yield |
| CRUDE_OIL | Energy | USD | barrel | WTI Crude Oil |
| SP500 | Equities | USD | index | S&P 500 Index |
| NIFTY50 | Equities | INR | index | NIFTY 50 Index |
| VIX | Volatility | USD | index | CBOE Volatility Index |
| INDIA_VIX | Volatility | INR | index | India Volatility Index |

## Metadata Metadata Standard
Every recorded observation stores:
- `provider`: Data provider name
- `retrieval_timestamp`: ISO-8601 UTC timestamp
- `symbol`: Instrument identifier
- `currency`: Instrument currency
- `unit`: Measurement unit (e.g. `oz`, `10g`, `kg`)
- `market`: Market classification
