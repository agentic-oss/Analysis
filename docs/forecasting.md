# Forecasting Framework Specification

## Probabilistic Signal Generation
1. Historical Analogue Similarity Search computes multi-factor normalized vector distances.
2. Evaluates forward 1D, 3D, 5D, 10D, 20D, and 60D return distributions across top K matches.
3. Computes positive return probability $P(\text{Return} > 0)$, expected median return, and historical min/max ranges.
4. Expresses outputs strictly as probabilistic research signals.
