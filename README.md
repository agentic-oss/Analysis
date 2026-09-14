# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

A production-quality precious-metals market intelligence, historical research, feature engineering, and probabilistic forecasting system.

## Features

- **Multi-Asset Ingestion & Indian Focus**: Collects Gold & Silver spot/futures, INR prices, USD/INR exchange rates, DXY, US 10Y/2Y Treasury yields, Crude Oil, S&P 500, NIFTY 50, VIX, and India VIX.
- **Data Quality & Validation**: Classifies observations into `valid`, `suspicious`, or `invalid` and produces daily audit reports (`data/quality/YYYY-MM-DD.json`).
- **Technical & Multi-Factor Indicators**: SMAs, EMAs, RSI, MACD, ATR, Bollinger Bands, Volume metrics, MA distances, 52-week highs/lows, and pattern triggers.
- **Gold & Silver Relative-Value Analysis**: Gold/Silver Ratio, Z-scores, historical percentiles, rolling spread, spread volatility, and forward return distributions across 1D, 3D, 5D, 10D, 20D, and 60D.
- **Macro & Market Regime Models**: Rule-based macro regime detector (Inflationary, Stagflationary, Risk-off, Tightening, etc.) and dual-axis market regime classifier (Trend / Volatility).
- **Historical Analogue Engine**: Searches historical database for similar market conditions using multidimensional normalized distance scoring and computes forward return statistics without look-ahead bias.
- **ML Feature Store & Forecasting**: Generates daily feature datasets with temporal protection, baseline probabilistic research signals, walk-forward validation (Logistic Regression, Random Forest), confidence scoring, and strategy backtesting.
- **Forecast Performance Evaluation**: Tracks stored predictions against actual outcomes over time in `data/models/model-performance.json`.
- **Daily Markdown & JSON Reports**: Automatically generates `reports/daily/YYYY-MM-DD.md` and `data/analysis/daily/YYYY-MM-DD.json`.
- **Automated CI/CD**: GitHub Actions workflow (`.github/workflows/daily-metals-analysis.yml`).

## Quickstart

```bash
# Install dependencies
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest

# Run pipeline for a specific date
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-30

# Run unit test suite
PYTHONPATH=. pytest tests/ -v
```

## System Architecture

See `docs/architecture.md` and `docs/methodology.md` for details.
