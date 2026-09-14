# System Architecture

The Gold & Silver Research Pipeline is designed as a modular time-series processing system.

## Project Structure

```
config/
  config.yaml               # System parameters, lookback windows, score weights, model settings
  instruments.json          # Instrument catalog (Gold, Silver, FX, Rates, Energy, Equities, Volatility)

src/
  ingestion/                # Abstract MarketDataProvider layer (yfinance & synthetic fallbacks)
  validation/               # Quality engine classifying observations (valid, suspicious, invalid)
  storage/                  # Parquet, CSV, and JSON persistence manager
  indicators/               # Technical indicator engine & pattern detection
  metals/                   # Dedicated Gold & Silver multi-window cross-asset correlation analysis
  ratios/                   # Relative value analysis and Gold/Silver ratio metrics
  regimes/                  # Transparent macro regime and dual-axis market regime classifiers
  features/                 # ML-ready daily feature store and event feature generators
  patterns/                 # Historical Analogue Engine searching multidimensional similarity
  scoring/                  # 0-100 multi-factor scoring engine and configurable alert detector
  forecasting/              # Probabilistic signals, walk-forward ML models, backtesting engine
  evaluation/               # Historical forecast performance tracking against real future outcomes
  reporting/                # Markdown and JSON report generators
  pipeline/
    runner.py               # End-to-end daily pipeline runner

tests/
  test_pipeline.py          # Automated pytest test suite

.github/
  workflows/
    daily-metals-analysis.yml # Scheduled GitHub Actions runner
```
