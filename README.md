# Precious Metals Market Data, Research & Forecasting System

A production-grade **Gold & Silver Market Data, Multi-Factor Research, Historical Analogue Search, and Probabilistic Forecasting Pipeline**.

## Overview

This repository operates as a continuously growing precious metals research database and intelligence engine. It fetches prices for spot gold, spot silver, COMEX futures, Indian INR prices (MCX & spot conversions), currency rates (USD/INR, DXY), interest rate yields (US 10Y, 2Y, Real Yields), commodities (Crude Oil WTI/Brent), equity markets (S&P 500, NIFTY 50), and volatility indices (VIX, India VIX).

> **Important Research Disclaimer**: All price predictions and forecasting outputs in this system are strictly framed as probabilistic historical research signals rather than guaranteed future prices.

---

## Key System Architecture & Capabilities

1. **Multi-Asset Ingestion & Abstraction**:
   - Provider abstraction layer (`MarketDataProvider`) with Yahoo Finance provider and fallback Synthetic provider for offline/resilient execution.
   - Comprehensive unit and currency conversions (`USD/oz`, `INR/10g`, `INR/kg`).

2. **Data Validation & Quality Tracking**:
   - Automatic classification of data batches into `valid`, `suspicious`, or `invalid`.
   - Logging daily data quality summaries to `data/quality/YYYY-MM-DD.json`.

3. **Technical Indicators & Cross-Market Correlation**:
   - Calculates SMAs (5, 10, 20, 50, 100, 200), EMAs (9, 20, 50, 200), RSI 14, Stochastic RSI, MACD, ATR 14, Bollinger Bands, OBV, and distance structure metrics.
   - Computes rolling correlations across 20d, 60d, 120d, and 252d windows against DXY, yields, oil, equities, and VIX.

4. **Gold/Silver Ratio & Relative Value Analysis**:
   - Computes historical percentiles, Z-scores, mean-reversion behavior, spread returns, and extreme episode forward outcomes.

5. **Macro & Market Regime Classification**:
   - Transparent rule-based macro regime detection (Inflationary, Risk-On, Risk-Off, Tightening, Easing, Neutral).
   - Independent asset trend & volatility regime classification.

6. **Historical Analogue Engine**:
   - Searches historical feature space using normalized distance to find closest matching market states.
   - Computes forward-return probability distributions (1D, 3D, 5D, 10D, 20D, 60D) without look-ahead bias.

7. **Transparent Multi-Factor Scoring (0-100)**:
   - Component scores: Technical, Macro, Momentum, Volatility, Relative Value, Historical Pattern.
   - Weighted composite score generation.

8. **ML Forecasting & Backtesting Engine**:
   - Baseline statistical & ML models (Logistic Regression, Ridge Regression, Random Forest) evaluated using walk-forward time-series validation.
   - Strategy backtester accounting for transaction costs, slippage, and position sizing.

9. **Daily Markdown & Machine-Readable JSON Reporting**:
   - Executive daily research report (`reports/daily/YYYY-MM-DD.md`).
   - Machine-readable structured outputs (`data/analysis/daily/YYYY-MM-DD.json`).
   - Automated alerts generation (`data/analysis/alerts/YYYY-MM-DD.json`).

---

## Directory Structure

```text
├── config/
│   ├── config.yaml               # System parameters, indicator parameters, weights
│   └── instruments.json          # Target instruments, exchanges, tickers, currencies
├── data/
│   ├── raw/                      # Raw daily ingestion batches (data/raw/YYYY/MM/YYYY-MM-DD/)
│   ├── processed/                # Processed Parquet & CSV files (metals, macro)
│   ├── features/                 # ML-ready daily feature store (data/features/daily/)
│   ├── analysis/                 # Machine-readable daily research & alerts JSON
│   ├── predictions/              # Forecast predictions logged by date
│   ├── quality/                  # Data quality report outputs
│   ├── models/                   # Model performance and evaluation metrics
│   └── backtests/                # Strategy backtest results
├── reports/
│   └── daily/                    # Daily Markdown research reports (YYYY-MM-DD.md)
├── docs/                         # Detailed architecture and methodology documentation
├── src/
│   ├── ingestion/                # Data provider interface, Yahoo provider, converters
│   ├── validation/               # Data batch validator
│   ├── storage/                  # Parquet/CSV/JSON storage manager
│   ├── indicators/               # Technical indicator calculations
│   ├── macro/                    # Macro analysis & event schedules
│   ├── metals/                   # Gold & Silver specific analyzers
│   ├── ratios/                   # Gold/Silver ratio analyzer
│   ├── regimes/                  # Macro & market regime detectors
│   ├── patterns/                 # Historical analogue engine
│   ├── features/                 # Daily feature dataset builder
│   ├── scoring/                  # Multi-factor composite scorer
│   ├── forecasting/              # Probabilistic signals & confidence scorer
│   ├── models/                   # Statistical & ML forecasting models
│   ├── backtesting/              # Strategy backtesting engine
│   ├── evaluation/               # Forecast evaluator & drift tracker
│   ├── reporting/                # Markdown report & alert generator
│   └── pipeline/
│       └── runner.py             # End-to-end pipeline orchestrator
└── tests/                        # Automated unit test suite
```

---

## How to Run

### Execute Daily Pipeline Runner

```bash
PYTHONPATH=. python3 src/pipeline/runner.py [YYYY-MM-DD]
```

Example:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-08
```

### Run Automated Tests

```bash
PYTHONPATH=. pytest
```

---

## Documentation

Full documentation is available in the `docs/` directory:
* [`docs/architecture.md`](docs/architecture.md)
* [`docs/data-sources.md`](docs/data-sources.md)
* [`docs/data-schema.md`](docs/data-schema.md)
* [`docs/methodology.md`](docs/methodology.md)
* [`docs/technical-analysis.md`](docs/technical-analysis.md)
* [`docs/macro-analysis.md`](docs/macro-analysis.md)
* [`docs/forecasting.md`](docs/forecasting.md)
* [`docs/backtesting.md`](docs/backtesting.md)
* [`docs/model-evaluation.md`](docs/model-evaluation.md)
