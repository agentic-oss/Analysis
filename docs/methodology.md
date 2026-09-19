# Methodology & Philosophy

## Research Philosophy
The objective of this pipeline is to establish a systematic, reproducible historical research framework. The system distinguishes between:
- **Observed Data** → **Indicators** → **Historical Patterns** → **Statistical Relationships** → **Model Output** → **Forward-Looking Research Signals**

All forecasts are expressed as conditional historical statistics and probabilistic research signals rather than deterministic predictions.

## Strict Look-Ahead Bias Prevention
1. Historical analogue searches restrict the candidate match window to dates $t_{hist} \le t_{current} - \max(\text{horizons})$.
2. Features for date $t$ are calculated strictly using observations available on or before date $t$.
3. ML walk-forward train/test splits perform temporal non-shuffled evaluation.
4. Strategy backtests lag signals by 1 trading day before trade execution.
