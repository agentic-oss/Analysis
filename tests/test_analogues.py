import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue_engine import AnalogueEngine

def test_analogue_engine():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    df_feat = pd.DataFrame({
        "rsi_14": np.random.uniform(30, 70, size=100),
        "macd": np.random.normal(0, 2, size=100),
        "volatility_20d": np.random.uniform(0.1, 0.2, size=100),
        "distance_sma_50": np.random.normal(0, 0.02, size=100),
        "distance_sma_200": np.random.normal(0, 0.05, size=100),
        "gold_silver_ratio_zscore": np.random.normal(0, 1, size=100),
        "dxy_return_20d": np.random.normal(0, 0.01, size=100),
        "us10y_change_20d": np.random.normal(0, 0.05, size=100),
        "price": np.cumsum(np.random.normal(1, 5, size=100)) + 2000.0
    }, index=dates)

    engine = AnalogueEngine()
    result = engine.find_analogues(df_feat, dates[-1], top_n=3)

    assert "top_analogues" in result
    assert len(result["top_analogues"]) == 3
    assert "forward_horizon_statistics" in result
    assert "5d" in result["forward_horizon_statistics"]
