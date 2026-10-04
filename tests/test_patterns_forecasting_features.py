import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue import AnalogueEngine
from src.forecasting.probabilistic import ForwardProbabilityEngine
from src.features.builder import FeatureStoreBuilder

def test_analogue_engine_and_forward_prob():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    hist_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "rsi_14": np.random.uniform(30, 70, 100),
        "macd": np.random.uniform(-2, 2, 100),
        "volatility_20d": np.random.uniform(0.1, 0.3, 100),
        "future_return_1d": np.random.normal(0.001, 0.01, 100),
        "future_return_5d": np.random.normal(0.003, 0.02, 100)
    })

    curr_state = hist_df.iloc[-1]

    engine = AnalogueEngine(feature_cols=["rsi_14", "macd", "volatility_20d"], top_k=10, min_sample=5)
    res = engine.find_analogues(hist_df, curr_state, horizons=[1, 5])

    assert res["sample_count"] > 0
    assert "forward_statistics" in res
    assert "1d" in res["forward_statistics"]

    prob_signals = ForwardProbabilityEngine.generate_probabilistic_signals(res["forward_statistics"], horizons=[1, 5])
    assert "1d" in prob_signals
    assert "signal_direction" in prob_signals["1d"]
    assert "prob_positive" in prob_signals["1d"]

def test_feature_store_no_lookahead():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    gold_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "symbol": ["GOLD"] * 10,
        "close": [100, 102, 101, 104, 105, 107, 106, 108, 110, 112],
        "open": [100] * 10,
        "high": [105] * 10,
        "low": [95] * 10,
        "volume": [1000] * 10
    })

    processed_dict = {"GOLD": gold_df}
    feat_df = FeatureStoreBuilder.build_daily_features(processed_dict, ratio_df=None, horizons=[1, 3])

    # Date 0 price = 100, Date 1 price = 102 -> Date 0 future_return_1d should be (102-100)/100 = 0.02
    assert feat_df.iloc[0]["future_return_1d"] == pytest.approx(0.02)
    # The last row future target must be NaN because future price is not yet known (prevents look-ahead)
    assert pd.isna(feat_df.iloc[-1]["future_return_1d"])
