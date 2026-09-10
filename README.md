# Precious Metals Intelligence & Analysis Pipeline

A production-quality **Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline**.

This repository serves as a continuously growing **precious-metals research database** that collects market data, calculates multi-factor technical indicators, evaluates macro drivers, detects market regimes and historical analogues, generates probabilistic forward-looking research signals, builds ML-ready feature datasets, and produces daily research reports.

> **Research Disclaimer:** All forecasting outputs are expressed strictly as historical conditional probabilistic research signals, not guaranteed future price predictions or trading advice.

---

## Key Features

1. **Multi-Asset Market Data Ingestion:** Gold, Silver, Gold INR, Silver INR, USD/INR, DXY, US 10Y/2Y Yields, Crude Oil, Equities (S&P 500, Nasdaq, Dow, NIFTY 50), and Volatility Indices (VIX, India VIX).
2. **Indian Market First-Class Support:** Direct support for Indian spot/futures/ETF benchmarks, USD/INR rates, and automatic conversions to INR/10g (Gold) and INR/kg (Silver).
3. **Data Quality & Validation:** Comprehensive checks for OHLC integrity, zero/negative prices, and abnormal price spikes logged daily in `data/quality/YYYY-MM-DD.json`.
4. **Technical & Relative-Value Analytics:** Moving averages, RSI, MACD, ATR, Bollinger Bands, OBV, and Gold/Silver Ratio z-scores.
5. **Macro & Market Regime Classification:** Transparent rule-based classifiers for Inflationary, Disinflationary, Risk-On, Risk-Off, Easing, Tightening, and Stagflationary environments.
6. **Historical Analogue Engine:** Distance-based k-NN pattern matching over historical feature states to estimate 1D, 3D, 5D, 10D, 20D, and 60D forward return distributions.
7. **ML Feature Store:** Leakage-free daily feature store with event-based features and future return targets.
8. **Walk-Forward Validation & Backtesting:** Expanding-window time-series validation and backtesting accounting for transaction costs and slippage.
9. **Daily Reporting & Alerts:** Automated daily Markdown reports (`reports/daily/YYYY-MM-DD.md`), JSON outputs (`data/analysis/daily/YYYY-MM-DD.json`), and alert notifications (`data/analysis/alerts/YYYY-MM-DD.json`).
10. **Automated CI/CD:** GitHub Actions workflow (`.github/workflows/daily-metals-analysis.yml`).

---

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running the Daily Pipeline

To run the pipeline for today or a specific historical date:

```bash
# Run for current date using live/yf provider
PYTHONPATH=. python3 src/pipeline/runner.py

# Run for a specific date using mock provider (offline / deterministic)
PYTHONPATH=. python3 src/pipeline/runner.py 2024-01-10 --mock
```

### Running Tests

```bash
pytest
```

---

## Directory Structure

```
.
├── config/                  # Instrument and system configuration files
├── data/                    # Scalable historical research database
│   ├── raw/                 # YYYY/MM/YYYY-MM-DD raw snapshot parquet files
│   ├── processed/           # Continuous Parquet time-series datasets
│   ├── features/            # Daily ML-ready feature store snapshot
│   ├── analysis/            # Daily JSON summaries and alerts
│   ├── quality/             # Daily data quality JSON reports
│   └── models/              # Model metrics and drift performance
├── docs/                    # Detailed technical documentation
├── reports/daily/           # Daily Markdown research reports
├── src/                     # Core codebase modular packages
│   ├── ingestion/           # Data provider abstraction and engines
│   ├── validation/          # Data validation and quality assurance
│   ├── storage/             # Parquet/JSON data storage management
│   ├── indicators/          # Technical indicators engine
│   ├── metals/              # Gold & Silver dedicated analytics
│   ├── regimes/             # Macro and market regime classifiers
│   ├── patterns/            # Historical analogue & probability engines
│   ├── features/            # ML feature store builder
│   ├── backtesting/         # Backtesting & walk-forward validation engines
│   ├── scoring/             # Transparent scoring and alert engines
│   ├── reporting/           # Markdown and JSON report generators
│   └── pipeline/            # End-to-end daily runner script
└── tests/                   # Test suite with deterministic fixtures
```

---

## Documentation

For detailed technical methodologies, refer to the documentation in `docs/`:
- [Architecture & Design](docs/architecture.md)
- [Data Sources & Providers](docs/data-sources.md)
- [Data Schema](docs/data-schema.md)
- [Methodology](docs/methodology.md)
- [Technical Analysis](docs/technical-analysis.md)
- [Macro Analysis](docs/macro-analysis.md)
- [Forecasting & Analogue Engine](docs/forecasting.md)
- [Backtesting Framework](docs/backtesting.md)
- [Model Evaluation & Monitoring](docs/model-evaluation.md)
