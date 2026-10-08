import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder
from src.patterns.analogues import AnalogueEngine

def test_no_lookahead_bias_in_features():
    dates = pd.date_range("2026-01-01", periods=10).strftime("%Y-%m-%d")
    df = pd.DataFrame({"date": dates, "close": range(100, 110)})
    feat_df = FeatureBuilder.build_daily_features(df, "GOLD", horizons=[1, 5])

    # Feature for day t (e.g. return_1d) must depend only on t and t-1
    assert feat_df["return_1d"].iloc[1] == pytest.approx((101 - 100) / 100.0)
    # Target variable future_return_1d for day t must be (close[t+1] - close[t]) / close[t]
    assert feat_df["future_return_1d"].iloc[0] == pytest.approx((101 - 100) / 100.0)
