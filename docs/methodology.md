# Methodology

## Research Pipeline Flow

1. **Ingest & Convert:** Ingest latest prices across metals, macro, equities, and FX. Convert spot gold/silver prices to INR measurements (INR/10g and INR/kg) using live USD/INR conversion rates.
2. **Validate:** Check for duplicate dates, missing values, non-positive OHLC values, invalid high/low/close relationships, and price spikes (>15%). Output results to `data/quality/YYYY-MM-DD.json`.
3. **Indicators:** Calculate multi-timeframe moving averages (SMA 5..200, EMA 9..200), RSI 14, Stoch RSI, MACD, ATR 14, 20D/60D volatility, Bollinger Bands, OBV, and price distances.
4. **Regimes:** Classify macro regime (Inflationary, Easing, Tightening, Risk-On, Risk-Off, Stagflationary) and asset-specific market regime (Trend + Volatility + Macro).
5. **Pattern Matching:** Use normalized Euclidean distance over selected feature dimensions (RSI, distance from SMA20, 20D volatility, MACD) to discover top 25 historical analogue dates.
6. **Probabilistic Forecasts:** Derive empirical probability distributions for 1D, 3D, 5D, 10D, 20D, and 60D forward returns from analogue outcomes.
7. **Scoring & Alerts:** Calculate weighted transparent sub-scores and composite score (0-100). Check for extreme price moves (>2.5% Gold, >3.5% Silver) or ratio extremes (>90 or <65).
8. **Reporting:** Write Markdown daily report and structured JSON payload.
