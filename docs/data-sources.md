# Data Sources & Instruments

The system ingests multi-asset price and macro data, normalizing across currencies and units.

## Supported Instruments (`config/instruments.json`)

### Precious Metals
- **GOLD**: Spot Gold (COMEX Spot, USD/oz)
- **SILVER**: Spot Silver (COMEX Spot, USD/oz)
- **GOLD_FUTURES**: Gold Futures (COMEX, USD/oz, `GC=F`)
- **SILVER_FUTURES**: Silver Futures (COMEX, USD/oz, `SI=F`)
- **GOLD_INR**: Indian Gold Retail/Spot Proxy (INR/10g, `GOLDBEES.NS`)
- **SILVER_INR**: Indian Silver Retail/Spot Proxy (INR/kg, `SILVERBEES.NS`)
- **MCX_GOLD**: MCX Gold Futures Proxy (INR/10g)
- **MCX_SILVER**: MCX Silver Futures Proxy (INR/kg)

### Currencies & Macro Drivers
- **USDINR**: USD / INR Exchange Rate (`USDINR=X`)
- **DXY**: US Dollar Index (`DX-Y.NYB`)
- **US10Y**: US 10-Year Treasury Yield (`^TNX`)
- **US2Y**: US 2-Year Treasury Yield (`^IRX`)
- **CRUDE_OIL**: WTI Crude Oil (`CL=F`)
- **BRENT_CRUDE**: Brent Crude (`BZ=F`)
- **SP500**: S&P 500 Index (`^GSPC`)
- **NIFTY50**: NIFTY 50 Index (`^NSEI`)
- **VIX**: CBOE Volatility Index (`^VIX`)
- **INDIA_VIX**: India Volatility Index (`^INDIAVIX`)

## Unit Conversions

- **Gold**: 1 Troy Ounce = 31.1034768 grams. Price per 10 grams in INR = `(USD_Price / 31.1034768) * 10 * USDINR_Rate`.
- **Silver**: 1 Troy Ounce = 0.0311034768 kilograms. Price per kilogram in INR = `(USD_Price / 0.0311034768) * USDINR_Rate`.

Both raw market prices and converted INR prices are preserved with conversion timestamps and FX rates used.
