# System Architecture

The pipeline consists of modular Python packages:
1. **Ingestion & Validation**: Fetches multi-asset market data and applies strict OHLC sanity checks.
2. **Technical & Relative Value Analysis**: Computes technical indicators, price structure, and Gold/Silver ratio metrics.
3. **Regimes & Historical Analogue Engine**: Determines composite macro and market regimes, finding past dates with similar multi-factor states.
4. **Feature Store & Forecasting**: Builds daily point-in-time features and generates probabilistic signals via walk-forward models.
5. **Backtesting & Scoring**: Runs realistic backtests (with transaction costs and slippage) and computes multi-factor composite scores.
6. **Reporting**: Outputs daily Markdown research reports and JSON analysis data.
