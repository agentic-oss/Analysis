# Research Methodology

The pipeline follows strict quantitative financial principles:
1. **No Look-Ahead Bias**: Features computed on date T only utilize data available at or before date T. Future targets are stored separately.
2. **Probabilistic Research Signals**: Forecasts are expressed as historical conditional distributions (mean return, median return, positive return probability) rather than guaranteed prices.
3. **Multi-Factor Correlation**: Multi-period rolling window correlations track driver dynamics against DXY, rates, equities, and inflation expectations.
