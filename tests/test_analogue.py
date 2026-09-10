import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue import AnalogueEngine, ForwardProbabilityEngine

def test_analogue_engine():
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(30, 70, n),
        "dist_sma_20": np.random.normal(0, 0.02, n),
        "volatility_20d": np.random.uniform(0.10, 0.25, n),
        "future_return_1d": np.random.normal(0.001, 0.01, n),
        "future_return_5d": np.random.normal(0.003, 0.02, n),
        "future_return_10d": np.random.normal(0.005, 0.03, n)
    })

    target = df.iloc[-1]
    hist = df.iloc[:-1]

    engine = AnalogueEngine(top_k=10)
    res = engine.find_analogues(
        target_features=target,
        historical_features_df=hist,
        feature_cols=["rsi_14", "dist_sma_20", "volatility_20d"],
        horizons=[1, 5, 10]
    )

    assert res["sample_count"] == 10
    assert "forward_stats" in res
    assert "5d" in res["forward_stats"]

    prob_engine = ForwardProbabilityEngine()
    signals = prob_engine.generate_probabilistic_signals(res)
    assert "signals" in signals
    assert "5d" in signals["signals"]
    assert "disclaimer" in signals
