# Precious Metals Research & Forecasting Pipeline

Continuous market data collection, deep multi-factor analysis, historical pattern analogue search, probabilistic research forecasting, feature store generation, and daily reporting pipeline for **Gold & Silver** (USD & INR representations).

---

## 📌 Architecture Overview

The platform operates as a modular, production-ready precious-metals research system. All price predictions and forecasting outputs are strictly framed as **probabilistic research signals** rather than guaranteed future prices.

```
data/
  raw/          # Date-partitioned raw OHLCV market observations (Parquet)
  processed/    # Cleaned, deduplicated time-series dataset (Parquet)
  features/     # Daily ML-ready feature datasets with target isolation (Parquet)
  analysis/     # Daily research signals and alert outputs (JSON)
  predictions/  # Tracked forward probability predictions for outcome evaluation (JSON)
  models/       # Walk-forward validation metrics and model drift tracking (JSON)
  quality/      # Daily ingestion quality audit reports (JSON)
  backtests/    # Historical strategy backtesting results
reports/
  daily/        # Human-readable daily Markdown research reports (YYYY-MM-DD.md)
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### 2. Run Daily Pipeline
To execute the end-to-end pipeline for today (or a specific date):
```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-09-08
```

### 3. Run Test Suite
```bash
PYTHONPATH=. pytest tests/
```

---

## 🛠 Modules & Key Components

- **`src/ingestion/`**: Provider layer with retries, exponential backoff, rate-limit handling, yfinance fetching, and fallback simulation for high availability.
- **`src/validation/`**: Batch data validator checking missing timestamps, OHLC logic, jumps, duplicate records, and producing audit reports in `data/quality/`.
- **`src/storage/`**: Parquet/JSON storage manager managing raw, processed, feature, and report paths.
- **`src/indicators/`**: Moving averages (SMA/EMA), momentum (RSI 14, MACD, Stochastic RSI, ROC), volatility (ATR 14, 20D Ann. Vol, Bollinger Bands), and price structure.
- **`src/metals/`**: Multi-factor driver correlation analysis for Gold & Silver across DXY, Real Yields, Rates, Oil, Equities, and Volatility over 20D, 60D, 120D, and 252D rolling windows.
- **`src/ratios/`**: Gold/Silver Ratio analysis, historical percentile, z-scores, mean-reversion, and conversion to Indian units (INR/10g for Gold, INR/kg for Silver).
- **`src/regimes/`**: Macro regime detection (Inflationary, Disinflationary, Tightening, Easing, Risk-Off, Neutral) and asset market regime classification (Trend + Volatility).
- **`src/patterns/`**: Historical Analogue Engine searching for normalized Euclidean nearest neighbors without look-ahead bias and calculating forward 1D, 3D, 5D, 10D, 20D, and 60D return distributions.
- **`src/forecasting/`**: Probabilistic Forward Probability Engine generating directional and expected return probabilities.
- **`src/features/`**: Feature Store builder enforcing strict temporal boundary separation between input features on date T and future return target horizons.
- **`src/models/`**: Baseline ML classifiers (Logistic Regression, Random Forest, Gradient Boosting) using walk-forward time-series validation.
- **`src/backtesting/`**: Backtesting framework with transaction costs, slippage, and position sizing computing Sharpe, Sortino, max drawdown, win rate, and profit factor.
- **`src/scoring/`**: Composite transparent scoring (0–100) aggregating Technical, Macro, Momentum, Volatility, Relative Value, and Historical Pattern dimensions.
- **`src/evaluation/`**: Forecast evaluation framework comparing prior date T prediction records against actual realized market outcomes as time elapses.
- **`src/reporting/`**: Markdown report generator (`reports/daily/YYYY-MM-DD.md`) and JSON alert detection (`data/analysis/alerts/YYYY-MM-DD.json`).

---

## 📖 Detailed Documentation

See the [`docs/`](docs/) directory for complete methodology specifications:
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/data-sources.md`](docs/data-sources.md)
- [`docs/data-schema.md`](docs/data-schema.md)
- [`docs/methodology.md`](docs/methodology.md)
- [`docs/technical-analysis.md`](docs/technical-analysis.md)
- [`docs/macro-analysis.md`](docs/macro-analysis.md)
- [`docs/forecasting.md`](docs/forecasting.md)
- [`docs/backtesting.md`](docs/backtesting.md)
- [`docs/model-evaluation.md`](docs/model-evaluation.md)

---

## 🤖 GitHub Actions Automation

The automated daily pipeline runs on schedule via `.github/workflows/daily-metals-analysis.yml` at market close, executing ingestion, feature generation, signal creation, test suites, report generation, secret scanning, and automated commit/push.
