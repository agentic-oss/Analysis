import numpy as np
import pandas as pd
import pytest
from src.patterns.analogue import HistoricalAnalogueEngine
from src.forecasting.probability import ForwardProbabilityEngine

def test_historical_analogue_engine():
    engine = HistoricalAnalogueEngine(top_k=5)

    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "close": 2000.0 + np.cumsum(np.random.randn(100) * 5),
        "rsi_14": 50 + np.random.randn(100) * 10,
        "dist_sma_200": np.random.randn(100) * 0.02,
        "future_return_5d": np.random.randn(100) * 0.01,
    })

    current_state = {"rsi_14": 52.0, "dist_sma_200": 0.01}
    res = engine.find_analogues(df, current_state, feature_cols=["rsi_14", "dist_sma_200"], horizons=[5])

    assert "top_analogues" in res
    assert len(res["top_analogues"]) <= 5
    assert "5d" in res["forward_statistics"]

    prob_out = ForwardProbabilityEngine.generate_probabilities(res, horizons=[5])
    assert "5d" in prob_out
    assert "disclaimer" in prob_out["5d"]
