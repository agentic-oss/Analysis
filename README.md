# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

> **Research Disclaimer**: All price predictions and forecasting outputs produced by this system are strictly framed as **probabilistic research signals** derived from historical empirical distributions, and NOT as guaranteed future prices.

A production-quality precious metals research database and intelligence pipeline designed for continuous data collection, multi-factor technical and macroeconomic analysis, historical analogue search, probabilistic forecasting, ML feature store generation, backtesting, and automated daily research reporting.

---

## Key Capabilities

1. **Precious Metals Data Ingestion & Normalization**: Collects Gold (spot & COMEX futures) and Silver (spot & COMEX futures) prices, converted Indian spot representations (INR/10g and INR/kg), and MCX proxies.
2. **Indian Market First-Class Support**: Tracks USD/INR exchange rates, converts USD/oz prices to INR/10g (Gold) and INR/kg (Silver) using exact exchange rates, recording exact conversion timestamps.
3. **Automated Data Validation**: Checks OHLC sanity, negative/zero prices, abnormal price jumps (>15%), duplicate timestamps, and outputs `data/quality/YYYY-MM-DD.json`.
4. **Comprehensive Technical Analysis**: Calculates SMAs, EMAs, RSI 14, Stochastic RSI, MACD, ATR, Bollinger Bands, OBV, trend strength, distance from 20/50/200 DMA and 52-week highs/lows, breakouts, and crosses.
5. **Macro Environment & Correlation Tracking**: Analyzes DXY, US 10Y/2Y yields, real yields, WTI crude oil, S&P 500, NIFTY 50, VIX, and India VIX along with rolling 20, 60, 120, and 252-day cross-asset correlations.
6. **Gold/Silver Ratio & Relative Value**: Tracks ratio levels, historical percentiles, Z-scores, mean-reversion behavior, and forward return spreads when ratio reaches extremes.
7. **Transparent Multi-Factor Regimes**: Rule-based macro regimes (Inflationary, Risk-On, Risk-Off, Stagflationary, Tightening, Easing, Disinflationary, Neutral) and asset-specific trend/volatility regimes.
8. **Historical Analogue Engine**: Matches today's state vector against historical observations (guaranteeing no look-ahead bias) to derive empirical forward return distributions for 1D, 3D, 5D, 10D, 20D, and 60D horizons.
9. **Explainable Scoring & Prediction Confidence**: Calculates Technical, Macro, Momentum, Volatility, Relative Value, and Composite scores (0-100) with confidence levels and explicit research reasons.
10. **Signal Backtesting & Forecast Tracking**: Evaluates signal-based strategies accounting for transaction costs and slippage; tracks past forecast accuracy over time in `data/models/model-performance.json`.
11. **Automated Reporting**: Generates daily Markdown research reports (`reports/daily/YYYY-MM-DD.md`) and machine-readable JSON outputs (`data/analysis/daily/YYYY-MM-DD.json`).

---

## Directory Architecture

```
data/
  raw/YYYY/MM/YYYY-MM-DD/
  processed/metals/
  processed/macro/
  features/daily/
  quality/
  analysis/daily/
  analysis/alerts/
  predictions/daily/
  models/
  backtests/
reports/
  daily/
src/
  ingestion/      # MarketDataProvider, yfinance ingestion, Indian market converters
  validation/     # OHLC data validation & quality reporting
  storage/        # Parquet, CSV, and JSON storage manager
  indicators/     # Technical indicators & signal detection
  macro/          # Macro drivers, interest rates, equities, FX, VIX
  metals/         # Gold & Silver specific analyzers & rolling correlations
  ratios/         # Gold/Silver ratio & relative-value analysis
  regimes/        # Multi-factor macro regimes & asset market regimes
  patterns/       # Historical analogue search engine
  features/       # ML daily feature dataset builder & macro events
  scoring/        # Score calculator, confidence framework, alert detector
  forecasting/    # Probabilistic forecast engine
  models/         # Baseline ML models & walk-forward time-series validation
  backtesting/    # Signal backtester with costs & slippage
  evaluation/     # Model metrics & forecast performance tracker
  reporting/      # Markdown & JSON daily report generator
  pipeline/       # Main daily pipeline execution runner
config/
  instruments.json
  config.yaml
docs/             # Architectural & methodology documentation
```

---

## Running the Pipeline

To run the pipeline for any date (defaults to today):

```bash
PYTHONPATH=. python3 src/pipeline/runner.py [YYYY-MM-DD]
```

To run the automated test suite:

```bash
pytest
```

---

## Documentation

Comprehensive documentation is available under `docs/`:
- [Architecture](docs/architecture.md)
- [Data Sources](docs/data-sources.md)
- [Data Schema](docs/data-schema.md)
- [Methodology](docs/methodology.md)
- [Technical Analysis](docs/technical-analysis.md)
- [Macro Analysis](docs/macro-analysis.md)
- [Forecasting](docs/forecasting.md)
- [Backtesting](docs/backtesting.md)
- [Model Evaluation](docs/model-evaluation.md)
