# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

A production-quality **Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline**.

The system continuously collects precious-metals market data, performs deep multi-factor quantitative and macroeconomic analysis, generates daily research reports, builds an ML-ready feature store, evaluates historical analogues, and generates probabilistic forward-looking research signals.

---

## Key Features

- **Multi-Asset Ingestion**: Supports Spot Gold, Spot Silver, Gold Futures, Silver Futures, Indian Gold/Silver ETFs (GOLDBEES, SILVERBEES), USD/INR FX rates, DXY Dollar Index, US 10Y/2Y Yields, TIPS Real Yields, Brent/WTI Crude Oil, Equities (S&P 500, Nasdaq, NIFTY 50), and Volatility (VIX, India VIX).
- **Data Validation & Quality Logging**: Ingested batches are checked for OHLC sanity, negative/zero prices, gaps, and abnormal jumps (>25% single-day). Records are classified into `valid`, `suspicious`, or `invalid` with log reports saved to `data/quality/YYYY-MM-DD.json`.
- **Quantitative Technical Indicators**: Calculates SMAs (5 to 200), EMAs (9 to 200), RSI 14, Stoch RSI, MACD, ATR 14, Historical Volatility, Bollinger Bands, OBV, MA distances, 52-week high/low distances, drawdown, and breakout detections.
- **Gold/Silver Ratio & Relative Value Analysis**: Computes current ratio, historical mean, standard deviation, Z-score, percentile ranking, and 1D/5D/20D return spreads.
- **Macro & Market Regime Detection**: Classifies macro regime (Inflationary, Risk-on, Risk-off, Tightening, Easing, Stagflationary) and instrument market regimes (Trend & Volatility).
- **Historical Analogue Engine**: Look-ahead-bias-free search engine matching current multi-factor state against historical database to calculate forward return distributions (1D, 3D, 5D, 10D, 20D, 60D).
- **Probabilistic Forecasting & Confidence Framework**: Outputs forward win rates, expected mean returns, median returns, expected volatility, and confidence scores with transparent reasons.
- **Machine Learning Feature Store & Baseline Models**: Builds leak-free daily feature store with future target returns and evaluates Logistic Regression and Random Forest models via walk-forward validation.
- **Backtesting Engine**: Reusable strategy backtesting engine accounting for transaction costs, slippage, max drawdown, Sharpe ratio, and win rates.
- **Automated Reporting & CI/CD**: Generates daily Markdown reports (`reports/daily/YYYY-MM-DD.md`), daily JSON (`data/analysis/daily/YYYY-MM-DD.json`), daily alerts (`data/analysis/alerts/YYYY-MM-DD.json`), and GitHub Actions workflow (`.github/workflows/daily-metals-analysis.yml`).

---

## Directory Structure

```
├── config/
│   ├── config.yaml
│   └── instruments.json
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── analysis/
│   ├── predictions/
│   ├── models/
│   ├── backtests/
│   └── quality/
├── docs/
│   ├── architecture.md
│   ├── data-sources.md
│   ├── data-schema.md
│   ├── methodology.md
│   ├── technical-analysis.md
│   ├── macro-analysis.md
│   ├── forecasting.md
│   ├── backtesting.md
│   └── model-evaluation.md
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

## Quick Start

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running the Daily Pipeline

```bash
# Run pipeline for a specific date (e.g. 2026-03-31)
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-31

# Run with mock provider (for testing/offline verification)
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-31 --mock
```

### Running Tests

```bash
PYTHONPATH=. pytest
```

---

## Documentation Guide

Detailed documentation is available in `docs/`:
- [Architecture](docs/architecture.md)
- [Data Sources](docs/data-sources.md)
- [Data Schema](docs/data-schema.md)
- [Methodology](docs/methodology.md)
- [Technical Analysis](docs/technical-analysis.md)
- [Macro Analysis](docs/macro-analysis.md)
- [Forecasting](docs/forecasting.md)
- [Backtesting](docs/backtesting.md)
- [Model Evaluation](docs/model-evaluation.md)

---

## Research Disclaimer

All forecasting outputs, probabilistic estimates, and research signals produced by this system represent historical conditional statistics and quantitative research models. They are provided solely for research purposes and do not represent financial advice or guarantees of future market performance.
