# Gold & Silver Market Data, Analysis, Research, and Forecasting Pipeline

A production-quality precious metals market data ingestion, deep analysis, historical analogue, feature store, backtesting, and probabilistic forecasting pipeline with a focus on global and Indian markets.

## System Architecture

The pipeline consists of modular components under `src/`:

1. **Ingestion (`src/ingestion/`)**: Fetches market data for Gold, Silver, USDINR, DXY, Rates, Equities, Energy, Volatility with rate-limiting, retries, and unit conversions (USD/oz to INR/10g and INR/kg).
2. **Validation (`src/validation/`)**: Validates prices, OHLC logic, jumps, duplicate timestamps, classifying observations into `valid`, `suspicious`, or `invalid`.
3. **Storage (`src/storage/`)**: Manages structured Parquet/CSV/JSON datasets under `data/`.
4. **Indicators (`src/indicators/`)**: Calculates SMAs, EMAs, RSI, MACD, ATR, Bollinger Bands, Volume indicators, price structure, and crosses.
5. **Macro Analysis (`src/macro/`)**: Multi-asset rolling correlations across 20d, 60d, 120d, 252d windows.
6. **Relative Value (`src/ratios/`)**: Gold/Silver ratio metrics, percentiles, Z-scores, and mean-reversion analysis.
7. **Regimes (`src/regimes/`)**: Rule-based macro and market regime classification.
8. **Historical Analogue Engine (`src/patterns/`)**: Similarity search comparing current market state to historical states and computing forward probability distributions.
9. **Feature Store (`src/features/`)**: Builds ML daily feature store with explicit, look-ahead bias protected future return targets (`future_return_1d` to `60d`).
10. **Forecasting Engine (`src/forecasting/`)**: Statistical and ML models predicting direction and returns.
11. **Backtesting Engine (`src/backtesting/`)**: Strategy backtester with transaction costs, slippage, position sizing, Sharpe/Sortino/Drawdown metrics.
12. **Forecast Evaluation (`src/evaluation/`)**: Historical forecast outcome tracking against realized actuals.
13. **Reporting & Alerts (`src/reporting/`)**: Daily Markdown research reports, machine-readable JSON, and alert detection.
14. **Pipeline Runner (`src/pipeline/`)**: Daily execution runner (`src/pipeline/runner.py`).

## Quick Start

### Installation
```bash
pip install pandas numpy scikit-learn pyyaml pyarrow yfinance pytest
```

### Running Tests
```bash
PYTHONPATH=. pytest
```

### Running Daily Pipeline
```bash
PYTHONPATH=. python3 src/pipeline/runner.py 2025-01-15 --synthetic
```

## Documentation
Detailed documentation is available in `docs/`:
- `docs/architecture.md`: System design and component interactions.
- `docs/data-sources.md`: Data providers and instrument specifications.
- `docs/data-schema.md`: Data schemas and quality standards.
- `docs/methodology.md`: Research philosophy and methodology.
- `docs/technical-analysis.md`: Indicator definitions and calculations.
- `docs/macro-analysis.md`: Macro regime model and cross-market analysis.
- `docs/forecasting.md`: Historical analogue search and ML models.
- `docs/backtesting.md`: Backtester design and transaction assumptions.
- `docs/model-evaluation.md`: Performance tracking and drift monitoring.
