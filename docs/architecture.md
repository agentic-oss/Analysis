# System Architecture

The Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline is structured as a modular end-to-end Python system.

## Modular Component Overview

```
src/
├── ingestion/        # MarketDataProvider abstraction & Yahoo Finance implementation
├── validation/       # Batch data validation & data quality reporting
├── storage/          # Local Parquet/CSV raw, processed, and feature stores
├── indicators/       # Moving averages, RSI, MACD, ATR, Bollinger, OBV, price structure
├── metals/           # Gold/Silver cross-market rolling correlation analytics
├── ratios/           # Gold/Silver ratio metrics, Z-scores, and relative value
├── regimes/          # Rule-based Macro and Market regime detectors
├── patterns/         # Historical Analogue Engine and multi-factor similarity matching
├── forecasting/      # Forward Probability Engine generating probabilistic signals
├── features/         # ML-ready Daily Feature Store builder with temporal isolation
├── scoring/          # Transparent 0-100 component & composite scoring engine
├── models/           # Baseline ML forecasting models with walk-forward validation
├── backtesting/      # Strategy backtesting engine with cost & slippage modeling
├── evaluation/       # Forecast performance tracking & model degradation monitoring
├── reporting/        # Daily Markdown research reports, JSON analysis, and alerts
└── pipeline/         # Orchestrating runner (`src/pipeline/runner.py`)
```
