# Architecture Overview

The system is structured as a modular Python pipeline for continuous collection, multi-factor analysis, probabilistic forecasting, and reporting for gold and silver markets.

## Pipeline Lifecycle

1. **Data Ingestion (`src/ingestion/`)**:
   - `MarketDataProvider` abstraction defines standard interface.
   - `YFinanceProvider` retrieves price time-series for Gold, Silver, Currencies (USDINR, DXY), Rates (US10Y, US2Y, Real Yields), Commodities (Oil), Equities (S&P500, NIFTY50), and Volatility (VIX).
   - Retries with exponential backoff handle rate-limiting.

2. **Validation & Normalization (`src/validation/`)**:
   - Classifies each record as `valid`, `suspicious`, or `invalid` based on zero/negative values, invalid OHLC logic, or abnormal single-day price jumps (>15%).
   - Quality metrics saved to `data/quality/YYYY-MM-DD.json`.

3. **Storage & Currency Conversion (`src/storage/`, `src/ingestion/currency.py`)**:
   - Converts USD/oz gold and silver prices to INR/10g and INR/kg using exact recorded USDINR exchange rates.
   - Stores raw and processed time-series in Parquet and CSV formats.

4. **Technical & Multi-Factor Indicators (`src/indicators/`, `src/macro/`, `src/metals/`, `src/ratios/`)**:
   - Calculates SMAs, EMAs, RSI, Stoch RSI, MACD, ATR, Bollinger Bands, OBV, and distance from 20/50/200 SMAs and 52-week extremes.
   - Computes rolling correlations across 20d, 60d, 120d, and 252d windows.
   - Analyzes Gold/Silver ratio z-scores, percentiles, and mean-reversion signals.

5. **Regimes & Historical Analogue Engine (`src/regimes/`, `src/patterns/`, `src/forecasting/`)**:
   - Multi-factor rule-based Macro Regime and Market Regime classification.
   - Euclidean distance similarity search over historical feature spaces to identify closest historical market analogues without temporal leakage.
   - Evaluates forward 1D, 3D, 5D, 10D, 20D, and 60D return probability distributions.

6. **Feature Store, Backtester & Performance Tracking (`src/features/`, `src/models/`, `src/backtesting/`, `src/evaluation/`)**:
   - Generates daily ML-ready feature datasets and future return targets.
   - Walk-forward expanding-window validation for machine learning models.
   - Backtests signal strategies with transaction costs and slippage.
   - Evaluates past forecast performance against realized actual prices.

7. **Reporting & Alerts (`src/reporting/`, `src/pipeline/`)**:
   - Checks threshold breaches to generate daily alert JSONs.
   - Compiles executive research summaries into Markdown reports (`reports/daily/YYYY-MM-DD.md`) and JSON outputs (`data/analysis/daily/YYYY-MM-DD.json`).
