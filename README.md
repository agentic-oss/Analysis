# Precious Metals Market Data, Multi-Factor Analysis & Forecasting Pipeline

Production-quality automated system for continuous gold and silver market data ingestion, Indian-market conversions, multi-factor analysis, historical analogue pattern detection, probabilistic forecasting, and automated daily research report generation.

---

## 🏛️ Architecture Overview

```
                          ┌──────────────────────────┐
                          │    Market Data Sources   │
                          │   (yfinance / Mock)      │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Data Ingestion & Quality │
                          │ Validator (DataValidator)│
                          └─────────────┬────────────┘
                                        │
                                        ▼
                          ┌──────────────────────────┐
                          │ Technical & Macro Engine │
                          │  (SMA, RSI, MACD, etc)   │
                          └─────────────┬────────────┘
                                        │
                                        ▼
┌──────────────────────────┬────────────┴─────────────┬──────────────────────────┐
│                          │                          │                          │
▼                          ▼                          ▼                          ▼
┌──────────────────────┐ ┌─────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│ Historical Analogue  │ │ Gold/Silver Relative│ │ Macro & Market       │ │ ML Feature Store     │
│ Similarity Engine    │ │ Value Ratio Analysis│ │ Regime Classifier    │ │ & Forecasting Engine │
└──────────┬───────────┘ └──────────┬──────────┘ └──────────┬───────────┘ └──────────┬───────────┘
           │                        │                       │                        │
           └────────────────────────┴───────────┬───────────┴────────────────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │ Daily Research Reporting │
                                  │ (Markdown / JSON / Alerts)│
                                  └──────────────────────────┘
```

---

## 🚀 Quickstart

### Installation

```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Run Full Daily Pipeline

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2024-03-01
```

### Run Tests

```bash
PYTHONPATH=. pytest
```

---

## 📁 Repository Structure

* `config/`: Instrument definitions (`instruments.json`) and system configurations (`config.yaml`).
* `src/ingestion/`: Market data providers, retries, and Indian unit conversions (USD/oz to INR/10g & INR/kg).
* `src/validation/`: Data quality checks, anomaly detection, and classification.
* `src/indicators/`: Technical indicators (SMAs, EMAs, RSI, Stoch RSI, MACD, ATR, Bollinger Bands, Volume, Structure).
* `src/metals/`: Multi-asset rolling correlation engine.
* `src/ratios/`: Gold/Silver ratio relative value spread analysis.
* `src/regimes/`: Rule-based macro and market regime classifiers.
* `src/patterns/`: Historical analogue similarity search engine without look-ahead bias.
* `src/features/`: ML-ready daily feature store generator.
* `src/models/`: Baseline forecasting models and expanding-window walk-forward validation.
* `src/forecasting/`: Probabilistic forward return research signals.
* `src/scoring/`: Transparent scoring engine and alert detection.
* `src/backtesting/`: Strategy backtesting engine with transaction costs and slippage.
* `src/evaluation/`: Historical forecast vs actual return performance tracker.
* `src/reporting/`: Daily Markdown report generator.
* `src/pipeline/`: Pipeline orchestrator (`runner.py`).
* `docs/`: Comprehensive technical documentation.
