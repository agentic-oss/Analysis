# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

A production-quality precious-metals research system and feature database.

## Overview
This system continuously collects precious-metals market data (spot, futures, Indian prices, USD/INR, rates, macro drivers), normalizes and validates incoming datasets, computes technical indicators, performs relative-value ratio analysis, classifies macro and market regimes, searches historical analogues, engineers ML features without look-ahead bias, trains probabilistic forecasting models, backtests strategies, and generates daily research reports.

## Architecture & Directory Structure
- `config/`: System configuration (`config.yaml`) and instrument specifications (`instruments.json`).
- `src/ingestion/`: Provider abstraction (`MarketDataProvider`, `YFinanceProvider`), retry handling, USD to INR unit conversions.
- `src/validation/`: Data quality classification (`valid`, `suspicious`, `invalid`) and quality JSON outputs.
- `src/storage/`: Parquet/CSV storage manager maintaining raw snapshots and master processed time-series.
- `src/indicators/`: Technical indicators (SMA, EMA, RSI, MACD, ATR, Bollinger, OBV) and price structure metrics.
- `src/metals/` & `src/ratios/`: Gold cross-market correlations, Silver relative value, and Gold/Silver ratio metrics.
- `src/regimes/` & `src/patterns/`: Composite macro regime model, asset trend/volatility regimes, and multi-factor historical analogue search engine.
- `src/features/` & `src/forecasting/`: Daily ML feature store builder, walk-forward validation, and baseline forecasting models.
- `src/evaluation/`: Historical prediction vs actual outcome evaluation store (`data/models/model-performance.json`).
- `src/backtesting/` & `src/scoring/`: Strategy backtester, multi-factor scoring engine, and automated market alert detector.
- `src/reporting/` & `src/pipeline/`: Markdown report generator, machine-readable JSON generator, and orchestrator runner (`src/pipeline/runner.py`).
- `.github/workflows/`: Automated daily GitHub Actions workflow (`daily-metals-analysis.yml`).

## Quickstart

### Installation
```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running Daily Pipeline
To run the full end-to-end daily pipeline for today or a specific date:
```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2024-05-10
```

### Running Tests
```bash
PYTHONPATH=. pytest tests/ -v
```

## Documentation
See `docs/` for detailed methodology, data schemas, forecasting models, and backtesting frameworks:
- `docs/architecture.md`
- `docs/data-schema.md`
- `docs/methodology.md`
- `docs/forecasting.md`
- `docs/backtesting.md`
