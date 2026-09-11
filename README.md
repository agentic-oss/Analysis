# Gold & Silver Market Data, Deep Analysis, and Forecasting Pipeline

A production-quality precious metals intelligence platform designed for continuous market data collection, multi-factor analysis, historical pattern matching, probabilistic return forecasting, and daily research report generation.

---

## Key Capabilities

* **Multi-Asset Ingestion & Resiliency:** Automated daily ingestion of Gold, Silver, COMEX Futures, Indian BEES/MCX proxies, USD/INR, DXY, US Treasuries, TIPS Real Yield proxies, Crude Oil, S&P 500, NIFTY 50, and VIX/India VIX with exponential backoff and synthetic fallback.
* **Indian Market First-Class Support:** Calculates INR price representations (INR/10g for Gold, INR/kg for Silver) with custom duty tracking, exchange rate recording, and Indian market indices.
* **Data Validation Core:** Inspects OHLC consistency, negative prices, duplicate dates, and price spikes, classifying records into `valid`, `suspicious`, or `invalid` and saving quality reports.
* **Technical & Relative Value Engines:** Computes moving averages, momentum (RSI, Stoch RSI, MACD, ROC), volatility (ATR, Historical Volatility, Bollinger Bands), and Gold/Silver Ratio percentiles/Z-scores.
* **Macro & Market Regimes:** Rule-based explainable macro regime classifier (Inflationary, Stagflationary, Risk-On, Risk-Off, Tightening, Easing) and asset-specific trend/volatility regimes.
* **Historical Analogue Engine:** Multi-factor Euclidean similarity search finding closest past market states and evaluating forward return distributions (1D, 3D, 5D, 10D, 20D, 60D).
* **Probabilistic Forecasting Signals:** Horizon-specific directional probabilities and expected return estimates disclaiming guaranteed future price action.
* **ML Feature Store & Backtesting:** Temporal look-ahead protected daily feature datasets, walk-forward expanding window time-series models, transaction cost backtesting, and prediction evaluation tracking.
* **Automated Daily Reporting & Orchestration:** Generates structured Markdown research reports (`reports/daily/YYYY-MM-DD.md`), stable JSON metrics (`data/analysis/daily/YYYY-MM-DD.json`), and GitHub Actions CI/CD automation.

---

## Getting Started

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest requests
```

### Running the Daily Pipeline

To execute the daily pipeline for a specific date:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-31
```

To run for today's date automatically:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py
```

### Running Tests

```bash
PYTHONPATH=. pytest tests/
```

---

## Documentation Index

Detailed documentation is available in the `docs/` directory:

* [Architecture Overview](docs/architecture.md)
* [Data Sources & Ingestion](docs/data-sources.md)
* [Data Schemas & Storage](docs/data-schema.md)
* [Research Methodology](docs/methodology.md)
* [Technical Analysis Engine](docs/technical-analysis.md)
* [Macro Analysis & Regimes](docs/macro-analysis.md)
* [Forecasting & Analogue Engine](docs/forecasting.md)
* [Backtesting Framework](docs/backtesting.md)
* [Model Evaluation & Tracking](docs/model-evaluation.md)

---

## Project Structure

```
├── config/
│   ├── config.yaml
│   └── instruments.json
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── analysis/
│   ├── quality/
│   └── predictions/
├── docs/
├── reports/
│   └── daily/
├── src/
│   ├── ingestion/
│   ├── validation/
│   ├── storage/
│   ├── indicators/
│   ├── metals/
│   ├── ratios/
│   ├── macro/
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
├── tests/
└── .github/workflows/
```
