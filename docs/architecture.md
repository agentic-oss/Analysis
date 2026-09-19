# System Architecture

The pipeline follows a modular architecture organized under `src/`:

1. **`src/ingestion/`**: Data providers implementing `MarketDataProvider` interface (`YFinanceProvider`, `MockMarketDataProvider`) with retries, exponential backoff, rate limiting, and metadata enrichment.
2. **`src/validation/`**: Data batch validator (`DataValidator`) performing OHLC bounds checks, missing value detection, jump classification, and quality reporting (`data/quality/YYYY-MM-DD.json`).
3. **`src/storage/`**: Persistence layer (`StorageManager`) managing raw, processed, feature, and report files in Parquet, CSV, and JSON formats.
4. **`src/indicators/`**: Technical indicator calculator (`TechnicalIndicators`) for moving averages, momentum, volatility, volume, and price structure metrics.
5. **`src/macro/`**: Macro analysis engine (`MacroAnalyzer`) computing rolling cross-market correlations across 20d, 60d, 120d, and 252d windows.
6. **`src/metals/` & `src/ratios/`**: Dedicated metal analyzer (`MetalAnalyzer`) and ratio analyzer (`RatioAnalyzer`) for Gold/Silver ratio, Z-scores, percentiles, and return spreads.
7. **`src/regimes/`**: Rule-based macro regime engine and instrument trend/volatility regime classifier (`RegimeDetector`).
8. **`src/patterns/`**: Look-ahead-bias-free historical analogue search engine (`HistoricalAnalogueEngine`).
9. **`src/features/`**: ML feature store builder (`FeatureBuilder`) producing daily feature tables with future forward targets (1d to 60d).
10. **`src/scoring/`**: Multi-factor scoring engine (`ScoringEngine`) computing sub-scores and composite research score (0-100).
11. **`src/forecasting/`**: Probabilistic forecaster (`ProbabilisticForecaster`) producing conditional probabilities, expected return/volatility, and confidence scores.
12. **`src/models/` & `src/evaluation/`**: Baseline ML model trainer (`BaselineModels`) and forecast performance tracker (`ForecastEvaluator`).
13. **`src/backtesting/`**: Strategy backtesting engine (`BacktestEngine`) with transaction costs, slippage, and performance metrics.
14. **`src/reporting/`**: Markdown and JSON report generator (`ReportGenerator`).
15. **`src/pipeline/`**: Daily pipeline runner (`DailyPipelineRunner`).
