# Pipeline System Architecture

```
[ Data Ingestion ] ---> [ Data Validation ] ---> [ Storage Layer (Parquet/CSV/JSON) ]
                              |
                              v
                   [ Technical Indicators ]
                   [ Multi-Factor Macro & Correlations ]
                   [ Gold/Silver Relative Value ]
                              |
                              v
                   [ Macro & Market Regimes ]
                   [ Historical Analogue Search Engine ]
                   [ Feature Store Builder ]
                              |
                              v
                   [ Multi-Factor Composite Scoring ]
                   [ Walk-Forward ML Forecasting Models ]
                   [ Probabilistic Signal Engine ]
                              |
                              v
                   [ Backtesting Engine & Forecast Evaluator ]
                   [ Daily Markdown & JSON Research Reports ]
```

## System Guiding Principles
1. **No Look-Ahead Bias:** Strict temporal isolation where features on date T only utilize information available up to date T.
2. **Data Integrity:** Historical observations are never discarded or unsuitably overwritten.
3. **Probabilistic Research Signals:** Outputs present conditional statistical likelihoods, not deterministic guarantees.
