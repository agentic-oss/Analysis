import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_builder import DailyFeatureBuilder

def test_historical_analogue_engine():
    dates = pd.date_range("2024-01-01", periods=100)
    hist_df = pd.DataFrame({
        "date": dates[:-1],
        "rsi_14": np.random.uniform(30, 70, 99),
        "volatility_20d": np.random.uniform(0.1, 0.3, 99),
        "dist_sma_200": np.random.uniform(-5, 10, 99),
        "dist_sma_20": np.random.uniform(-2, 5, 99),
        "gold_silver_ratio_zscore": np.random.uniform(-1.5, 1.5, 99),
        "dxy_return_20d": np.random.uniform(-0.02, 0.02, 99),
        "us10y_change_20d": np.random.uniform(-0.2, 0.2, 99),
        "vix_level": np.random.uniform(12, 25, 99),
        "future_return_5d": np.random.uniform(-0.03, 0.03, 99),
        "future_return_10d": np.random.uniform(-0.05, 0.05, 99)
    })

    curr = hist_df.iloc[-1]
    engine = HistoricalAnalogueEngine(top_k=5)
    res = engine.find_analogues(curr, hist_df.iloc[:-1])

    assert res["num_analogues"] == 5
    assert "forward_statistics" in res

def test_daily_feature_builder():
    builder = DailyFeatureBuilder()
    metal_df = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-02"],
        "close": [2500.0, 2520.0],
        "sma_200": [2400.0, 2405.0]
    })
    feats = builder.build_feature_dataset(metal_df, pd.DataFrame(), pd.DataFrame())
    assert "return_1d" in feats.columns
    assert "future_return_1d" in feats.columns
