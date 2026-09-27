# Gold & Silver Market Data, Multi-Factor Analysis, Research, and Forecasting Pipeline

A production-quality precious metals market research database and probabilistic forecasting pipeline built for Indian and global financial markets.

## Primary Capabilities
1. **Multi-Asset Ingestion**: Ingests Gold (USD/oz, INR/10g), Silver (USD/oz, INR/kg), USD/INR, DXY, US Treasury Yields, Real Yields, Crude Oil, Equities (S&P 500, NIFTY 50), and Volatility (VIX, India VIX).
2. **Indian Market Focus**: First-class support for INR measurements (INR/10g for Gold, INR/kg for Silver), recording exact FX rates and conversion timestamps.
3. **Multi-Factor Analysis**: Technical indicators, rolling cross-asset correlations, and Gold/Silver ratio relative-value Z-scores.
4. **Regime & Historical Analogue Engine**: Classification of Macro Regimes (Inflationary, Risk-On, Risk-Off, Tightening) and Market Regimes, coupled with Euclidean similarity search for historical analogues.
5. **Probabilistic Research Signals**: Generates 1D, 3D, 5D, 10D, 20D, and 60D forward return conditional distributions.
6. **ML Feature Store & Backtester**: Builds temporal feature datasets without look-ahead bias and backtests signal strategies with transaction costs and slippage.
7. **Automated Daily Reporting & Alerting**: Generates daily Markdown research reports and machine-readable JSON outputs.

## Pipeline Architecture
```
src/
├── ingestion/       # MarketDataProvider & yfinance integration
├── validation/      # Data quality classification (valid, suspicious, invalid)
├── storage/         # Parquet and JSON storage manager
├── indicators/      # Moving Averages, RSI, MACD, ATR, Bollinger Bands, OBV
├── macro/           # Rolling cross-asset correlation analyzer
├── metals/          # Dedicated Gold and Silver multi-factor modules
├── ratios/          # Gold/Silver Ratio relative-value Z-score & percentiles
├── regimes/         # Rule-based Macro and Market Regime classifiers
├── patterns/        # Historical Analogue Engine
├── forecasting/     # Forward Probability Engine
├── features/        # ML Feature Store generator
├── models/          # Walk-forward validation model pipeline
├── backtesting/     # Event-driven backtesting engine
├── evaluation/      # Confidence scoring & Forecast evaluation tracker
├── reporting/       # Alert generator & Markdown/JSON report builder
└── pipeline/        # Main pipeline runner (runner.py)
```

## Running the Pipeline
To execute the daily pipeline locally:
```bash
PYTHONPATH=. python3 src/pipeline/runner.py [YYYY-MM-DD]
```

## Running Tests
```bash
pytest tests/
```

## Documentation
Detailed documentation is available in `docs/`:
- `docs/architecture.md`
- `docs/methodology.md`
- `docs/technical-analysis.md`
- `docs/forecasting.md`
