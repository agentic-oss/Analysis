import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue import HistoricalAnalogueEngine
from src.features.builder import FeatureStoreBuilder
from src.scoring.composite import TransparentScorer

def test_historical_analogue_engine():
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = 1800.0 + np.cumsum(np.random.normal(0.5, 8, 300))
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": prices,
        "rsi_14": np.random.uniform(30, 70, 300),
        "dist_sma_20": np.random.uniform(-3, 3, 300),
        "dist_sma_50": np.random.uniform(-5, 5, 300),
        "dist_sma_200": np.random.uniform(-10, 10, 300),
        "volatility_20d": np.random.uniform(10, 25, 300),
        "gold_silver_ratio": np.random.uniform(70, 90, 300),
        "return_20d": np.random.uniform(-5, 5, 300)
    })

    engine = HistoricalAnalogueEngine(top_k=5)
    curr_state = df.iloc[-1]
    res = engine.find_analogues(df.iloc[:-10], curr_state)
    assert res["sample_count"] > 0
    assert "5d" in res["forward_statistics"]

def test_feature_store_builder():
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    metal_df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "close": 2000.0 + np.cumsum(np.random.normal(0, 5, 50))
    })

    builder = FeatureStoreBuilder(forecast_horizons=[1, 5])
    feat_df = builder.build_feature_dataset(metal_df)
    assert "sma_20" in feat_df.columns
    assert "target_future_return_1d" in feat_df.columns
    assert "target_future_return_5d" in feat_df.columns

def test_transparent_scorer():
    row = pd.Series({"close": 2000.0, "sma_20": 1980.0, "sma_50": 1950.0, "sma_200": 1900.0, "rsi_14": 60.0, "volatility_20d": 14.0})
    macro_sum = {"DXY": {"60d": -0.4}, "VIX_level": 18.0}
    ratio_sum = {"current_zscore": 0.5}
    analogue_sum = {"forward_statistics": {"5d": {"positive_probability_pct": 65.0}}}

    scorer = TransparentScorer()
    res = scorer.compute_composite_score(row, macro_sum, ratio_sum, analogue_sum)
    assert res["composite_score"] > 0
    assert "technical" in res["component_scores"]
