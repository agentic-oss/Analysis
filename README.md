# Precious Metals Market Data, Deep Analysis, Historical Research & Forecasting Pipeline

Production-quality daily market data collection, multi-factor technical/macro analysis, historical analogue pattern detection, relative-value ratio modeling, and probabilistic forward research signal engine for **Gold** and **Silver**.

> **Research Philosophy & Disclaimer**: All price predictions and forecasting outputs produced by this repository are strictly framed as **probabilistic research signals** derived from historical conditional distributions and quantitative models rather than guaranteed future prices.

---

## Key Features

* **Continuous Data Ingestion & Storage**: Automated ingestion across Precious Metals (Spot & Futures), Currencies (USD/INR, DXY), Rates (US10Y, Real Yields), Energy (WTI, Brent), Equities (S&P 500, Nasdaq, Dow Jones, NIFTY 50), Volatility (VIX, India VIX).
* **Indian Market Specifics**: Support for INR/10g (Gold) and INR/kg (Silver) measurements with exact exchange rate and conversion timestamp tracking.
* **Data Validation Engine**: Automated check for missing dates, duplicates, invalid OHLC relationships, negative prices, price jumps, and stale data writing to `data/quality/YYYY-MM-DD.json`.
* **Technical & Multi-Factor Indicators**: Full technical suite (SMA, EMA, RSI, Stoch RSI, MACD, ATR, Bollinger Bands, Volume, OBV, Price Structure, Golden/Death Cross, Breakouts).
* **Gold/Silver Relative Value Ratio**: Rolling ratio, 252d Z-score, mean reversion behavior, spread volatility, and forward return spread statistics.
* **Macro & Market Regime Classification**: Transparent, explainable rule-based classifiers for Macro Regimes (Inflationary, Risk-On, Risk-Off, Stagflationary, Tightening) and Market Regimes (Bullish, Bearish, Range, High/Low Volatility).
* **Historical Analogue Engine**: Multi-metric similarity search without look-ahead bias, computing forward return distributions across 1D, 3D, 5D, 10D, 20D, and 60D horizons.
* **ML Feature Store**: Daily feature dataset saved in Parquet/CSV format with separated target horizon columns.
* **Transparent Scoring & Alerts**: 0–100 component scores (Technical, Macro, Momentum, Volatility, Relative Value, Pattern, Model, Composite) and threshold-driven alert generation (`data/analysis/alerts/YYYY-MM-DD.json`).
* **Daily Research Reports**: Automated Markdown research reports (`reports/daily/YYYY-MM-DD.md`) and machine-readable JSON analysis (`data/analysis/daily/YYYY-MM-DD.json`).
* **Automated CI/CD Pipeline**: GitHub Actions workflow (`.github/workflows/daily-metals-analysis.yml`) with automated scheduling, secret protection, testing, and git committing.

---

## Directory Structure

```
.
├── config/
│   ├── config.yaml          # Pipeline, lookback, scoring, and alert parameters
│   └── instruments.json     # Asset and market ticker specifications
├── data/
│   ├── raw/                 # YYYY/MM/YYYY-MM-DD/ raw observation Parquet files
│   ├── processed/           # Processed time-series Parquet and CSV files
│   ├── features/            # Daily ML feature store datasets
│   ├── analysis/            # Daily analysis and alert JSON files
│   ├── predictions/         # Daily forecast outputs
│   ├── quality/             # Ingestion batch data quality JSON reports
│   └── models/              # Model performance tracking
├── docs/                    # Detailed technical architecture & methodology documentation
├── reports/
│   └── daily/               # Formatted daily Markdown research reports
├── src/                     # Modular Python packages
│   ├── ingestion/           # Data provider abstractions & FX/unit converters
│   ├── validation/          # Data quality validator engine
│   ├── storage/             # Data storage manager
│   ├── indicators/          # Technical indicators calculator
│   ├── metals/              # Gold & Silver multi-factor correlation engines
│   ├── ratios/              # Gold/Silver relative value ratio engine
│   ├── regimes/             # Macro and Market regime classifiers
│   ├── patterns/            # Historical analogue search engine
│   ├── features/            # Daily ML feature builder
│   ├── scoring/             # Score and alert engines
│   ├── forecasting/         # Probabilistic signals & confidence framework
│   ├── models/              # Walk-forward statistical and ML baseline models
│   ├── backtesting/         # Strategy backtesting engine
│   ├── evaluation/          # Forecast performance tracking engine
│   ├── reporting/           # Daily report & JSON generator
│   └── pipeline/            # End-to-end execution runner
└── tests/                   # Pytest automated test suite
```

---

## Getting Started

### Prerequisites

* Python 3.12+
* Required packages: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `pyarrow`, `yfinance`, `pytest`

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running the Daily Pipeline

Run the pipeline for today's date:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py
```

Run the pipeline for a specific historical or target date:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-10-02
```

### Running Tests

Execute the full test suite with `pytest`:

```bash
pytest -v
```

---

## Documentation

For in-depth methodologies and schemas, inspect the `docs/` directory:
* [Architecture & Flow](docs/architecture.md)
* [Research Methodology](docs/methodology.md)
* [Forecasting Engine](docs/forecasting.md)
* [Data Schema Specification](docs/data-schema.md)
