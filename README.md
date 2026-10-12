# Precious Metals Market Data, Deep Analysis & Forecasting Pipeline

A production-quality precious metals market data ingestion, quantitative analysis, historical research, and probabilistic forecasting system built in Python.

> **Disclaimer**: All price predictions and forecasting outputs produced by this system must strictly be framed as **probabilistic research signals** rather than guaranteed future prices.

---

## 🌟 Key Features

* **Multi-Asset Ingestion**: Continuously ingests spot, futures, and ETF prices for Gold and Silver along with USD/INR FX rates, DXY, US Treasuries, TIPS (Real Yields), Crude Oil, Equity Indices (S&P 500, NIFTY 50), and Volatility (VIX, India VIX).
* **Indian Market First-Class Support**: Captures original market prices and calculates normalized INR equivalents (INR/10g for gold, INR/kg for silver) storing exact exchange rates and conversion timestamps.
* **Data Quality & Validation**: Non-destructive data validation classifying records into `valid`, `suspicious`, or `invalid` with detailed quality logs (`data/quality/YYYY-MM-DD.json`).
* **Technical Analysis**: 50+ quantitative indicators including SMAs, EMAs, RSI, Stochastic RSI, MACD, ATR, Bollinger Bands, rolling volatility, and price structure flags (Golden/Death Cross, breakouts).
* **Multi-Asset Correlations & Gold/Silver Ratio**: Rolling 20, 60, 120, and 252-day correlation matrix and Gold/Silver relative value Z-scores, historical percentiles, and mean-reversion statistics.
* **Explainable Macro & Market Regimes**: Rule-based classifiers categorizing macro states (*Inflationary*, *Stagflationary*, *Risk-Off*, *Tightening*, *Easing*) and asset trend/volatility regimes.
* **Historical Analogue Search Engine**: Multi-dimensional similarity scoring locating closest historical market states to estimate forward return probabilities across 1D, 3D, 5D, 10D, 20D, and 60D horizons without look-ahead bias.
* **ML Feature Store & Walk-Forward Validation**: Point-in-time daily feature store (`data/features/daily/`) evaluated using time-series expanding-window walk-forward validation.
* **Event-Driven Backtester**: Reusable backtesting engine accounting for transaction costs, slippage, and position sizing.
* **Forecast Performance Tracking**: Continuously compares past date $T$ forecasts with actual historical outcomes (`data/models/forecast_evaluation.json`).
* **Daily Research Reports & Alerts**: Automated daily Markdown (`reports/daily/YYYY-MM-DD.md`) and JSON (`data/analysis/daily/YYYY-MM-DD.json`) research reports and alert triggers (`data/analysis/alerts/YYYY-MM-DD.json`).

---

## 📁 System Architecture

```text
├── config/
│   ├── config.yaml
│   └── instruments.json
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── quality/
│   ├── predictions/
│   └── analysis/
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── ...
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
│   ├── models/
│   ├── forecasting/
│   ├── backtesting/
│   ├── evaluation/
│   ├── scoring/
│   ├── reporting/
│   └── pipeline/
└── tests/
```

---

## 🚀 Quickstart

### Run Daily Pipeline
```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-01
```

### Run Tests
```bash
pytest
```
