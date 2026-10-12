# Research Methodology

The pipeline follows strict quantitative financial research principles:

1. **Zero Look-Ahead Bias**: Features computed on date $T$ only utilize information available up to date $T$. Forward target returns $R_{T+h} = \frac{P_{T+h} - P_T}{P_T}$ are strictly stored as targets shifted by $-h$.
2. **Probabilistic Research Signals**: All model outputs and analogue engine returns are framed as conditional historical statistics, not guaranteed price trajectories.
3. **Transparent Rules & Feature Weights**: Regime classifications and similarity distance calculations utilize explicit, documented feature weights.
