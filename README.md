# Precious Metals Research & Forecasting Pipeline

Production-grade research pipeline for **Gold, Silver, Indian Markets, Macro Drivers, and Probabilistic Forward Signals**.

## Overview
This system continuously ingests global precious metals and macro market data, performs multi-factor technical and quantitative analysis, classifies macro and market regimes, identifies historical analogues, calculates probabilistic forward return statistics, and outputs daily research reports and ML-ready datasets.

> **Research Principle:** All price predictions and forecasting outputs are strictly framed as **probabilistic research signals** derived from historical conditional statistics, never as guaranteed future price predictions.

---

## Directory Architecture

```
.
├── config/                  # Configuration files (instruments.json, config.yaml)
├── src/                     # Core Python modules
│   ├── ingestion/           # Data providers (YFinance) and currency conversion
│   ├── validation/          # Data quality checks & audit classification
│   ├── storage/             # Parquet, CSV, JSON data persistence
│   ├── indicators/          # Technical indicators & price structure metrics
│   ├── macro/               # Cross-market macro driver analysis
│   ├── metals/              # Gold & Silver specific correlation analysis
│   ├── ratios/              # Relative value (Gold/Silver ratio Z-score, spreads)
│   ├── regimes/             # Rule-based Macro & Market regime models
│   ├── patterns/            # Historical Analogue Engine without look-ahead bias
│   ├── features/            # Daily ML feature store builder
│   ├── scoring/             # Transparent multi-factor composite scoring engine
│   ├── forecasting/         # Probabilistic research signal generator
│   ├── models/              # Walk-forward ML forecasting models
│   ├── backtesting/         # Time-series backtest engine (costs & slippage)
│   ├── evaluation/          # Forecast tracking & performance evaluation
│   ├── reporting/           # Daily Markdown and JSON report generator
│   └── pipeline/            # End-to-end pipeline orchestration runner
├── tests/                   # Automated pytest test suite
├── data/                    # Research database (raw, processed, features, quality, analysis, predictions)
├── reports/                 # Daily Markdown research reports
└── docs/                    # Detailed architecture & methodology documentation
```

---

## Quick Start

### 1. Installation
```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### 2. Run Daily Pipeline
```bash
PYTHONPATH=. python3 src/pipeline/runner.py --date YYYY-MM-DD
```

### 3. Run Test Suite
```bash
pytest
```

---

## Core Features

- **Indian Market Focus:** Native support for USD/INR, INR/10g Gold, INR/kg Silver, MCX Gold (GOLDBEES), MCX Silver (SILVERBEES), NIFTY 50, and India VIX.
- **Data Quality Audit:** Ingestion batch validation classifying records into `valid`, `suspicious`, or `invalid` saved in `data/quality/YYYY-MM-DD.json`.
- **Historical Analogue Engine:** Euclidean distance similarity search matching today's multi-factor state against past historical states without look-ahead bias.
- **Probabilistic Forecasting:** Multi-horizon conditional statistics (1D, 3D, 5D, 10D, 20D, 60D) with multi-factor confidence scoring.
- **Forecast Tracking:** Prediction logging comparing historical research signals against actual future outcomes saved in `data/models/model-performance.json`.

---

## License
MIT
