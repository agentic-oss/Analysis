# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

A production-quality precious metals market intelligence database, feature store, and probabilistic research forecasting engine.

---

## 🌟 Highlights

- **Precious Metals & Indian Market Focus**: Tracks Spot & Futures Gold/Silver (USD/oz), domestic Indian prices (INR/10g Gold, INR/kg Silver via NSE GoldBEES & SilverBEES), and USD/INR exchange rates.
- **Multi-Factor Macro Drivers**: Integrates ICE US Dollar Index (DXY), US 10Y and 2Y Treasury Yields, Crude Oil (WTI), S&P 500, NIFTY 50, CBOE VIX, and India VIX.
- **Historical Analogue Engine**: Multidimensional pattern matching based on normalized Euclidean feature similarity, computing historical forward return statistics across 1D, 3D, 5D, 10D, 20D, and 60D horizons without look-ahead bias.
- **Probabilistic Forecasting Signals**: Expresses all forward expectations as probabilistic conditional signals with confidence scoring rather than guaranteed future price predictions.
- **ML Feature Store & Storage**: Parquet-backed feature store (`data/features/daily/`) and time-series validation baseline models.
- **Daily Automated Research Reports**: Generates daily Markdown research reports (`reports/daily/YYYY-MM-DD.md`) and machine-readable JSON outputs (`data/analysis/daily/YYYY-MM-DD.json`).

---

## 📁 Repository Directory Architecture

```text
├── config/
│   ├── instruments.json
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── quality/
│   ├── analysis/
│   └── models/
├── docs/
│   ├── architecture.md
│   ├── data-schema.md
│   ├── methodology.md
│   ├── forecasting.md
│   └── backtesting.md
├── reports/
│   └── daily/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── storage/
│   ├── indicators/
│   ├── macro/
│   ├── metals/
│   ├── ratios/
│   ├── regimes/
│   ├── patterns/
│   ├── features/
│   ├── scoring/
│   ├── forecasting/
│   ├── models/
│   ├── backtesting/
│   ├── evaluation/
│   ├── reporting/
│   └── pipeline/
└── tests/
```

---

## 🚀 Getting Started

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running the Daily Pipeline

To execute the daily pipeline for a specific date:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-31
```

To run using synthetic deterministic provider (for offline / testing mode):

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-31 --synthetic
```

### Running Tests

```bash
pytest -v
```
