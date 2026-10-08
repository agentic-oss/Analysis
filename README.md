# Gold & Silver Market Data, Deep Analysis, Historical Research, and Forecasting Pipeline

A production-quality precious metals intelligence, multi-factor research, feature store, and probabilistic forecasting pipeline designed for continuous automated daily execution.

> **Disclaimer & Research Philosophy:**
> All price predictions, forward probabilities, and research signals produced by this repository are strictly framed as **historical conditional probabilistic research signals**, NOT guaranteed future price outcomes or financial advice.

---

## Architecture Overview

The system is organized into modular Python components:

* `config/`: Instrument configurations (`instruments.json`) and system parameters (`config.yaml`).
* `src/ingestion/`: Abstract provider interface (`provider.py`) and `YFinanceProvider` implementation with exponential retries and rate limiting.
* `src/validation/`: `DataValidator` checking for missing fields, price jumps, zero/negative prices, and invalid OHLC relationships.
* `src/storage/`: `DataRepository` handling Parquet time-series datasets and JSON/CSV reports.
* `src/indicators/`: `TechnicalIndicators` engine computing SMAs, EMAs, RSI, Stochastic RSI, MACD, ATR, Bollinger Bands, OBV, and price structure signals.
* `src/metals/`: Dedicated `GoldAnalyzer` and `SilverAnalyzer` modules supporting USD/oz, INR/10g, and INR/kg conversions.
* `src/ratios/`: `GoldSilverRatio` relative-value engine with rolling Z-scores and percentiles.
* `src/macro/`: `MacroAnalyzer` computing multi-window rolling correlations across FX (DXY, USD/INR), bond yields, crude oil, equity indices (NIFTY 50, S&P 500, Nasdaq, Dow Jones), and volatility (VIX, India VIX).
* `src/regimes/`: Rule-based `MacroRegimeDetector` and `MarketRegimeDetector`.
* `src/patterns/`: `AnalogueEngine` searching historical multi-factor nearest neighbors without look-ahead bias.
* `src/features/`: `FeatureBuilder` creating daily ML-ready feature store tables and target variables.
* `src/scoring/`: `CompositeScorer` generating transparent 0-100 scores across Technical, Momentum, Volatility, Macro, Relative Value, and Historical Pattern dimensions.
* `src/forecasting/`: `ProbabilisticForecaster` outputting multi-horizon research signals (1D, 3D, 5D, 10D, 20D, 60D).
* `src/models/`: `BaselineModels` supporting walk-forward time-series validation.
* `src/backtesting/`: `BacktestEngine` simulating signal performance with commission and slippage costs.
* `src/evaluation/`: `ForecastTracker` measuring directional accuracy and calibration over time.
* `src/reporting/`: `DailyReportGenerator` and `AlertEngine`.
* `src/pipeline/`: `runner.py` master orchestrator execution script.

---

## Execution & Usage

### 1. Run Pipeline
To run the automated pipeline for the current or specified date:

```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2026-03-30
```

### 2. Run Tests
To execute the automated test suite:

```bash
PYTHONPATH=. pytest
```

---

## Documentation

Detailed documentation on methodology and architecture can be found under `docs/`:
- `docs/architecture.md`
- `docs/methodology.md`
- `docs/forecasting.md`
- `docs/backtesting.md`
