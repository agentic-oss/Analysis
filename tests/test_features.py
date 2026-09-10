import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureStoreBuilder

def test_feature_store_builder():
    n = 50
    dates = pd.date_range("2023-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
    df = pd.DataFrame({
        "date": dates,
        "open": prices*0.99,
        "high": prices*1.01,
        "low": prices*0.98,
        "close": prices,
        "volume": 10000
    })

    events_df = pd.DataFrame({
        "date": ["2023-01-15", "2023-02-15"],
        "event": ["FOMC", "CPI"]
    })

    builder = FeatureStoreBuilder(horizons=[1, 5])
    feat_df = builder.create_features_for_instrument("GOLD", df, events_df=events_df)

    assert not feat_df.empty
    assert "instrument" in feat_df.columns
    assert "days_to_event" in feat_df.columns
    assert "macro_regime" in feat_df.columns
    assert "market_regime" in feat_df.columns
    assert "future_return_1d" in feat_df.columns
    assert "future_return_5d" in feat_df.columns
    # Check no target leakage on current features: future_return_1d at t=0 must match (close[1]-close[0])/close[0]
    expected_f1 = (df["close"].iloc[1] - df["close"].iloc[0]) / df["close"].iloc[0]
    assert np.isclose(feat_df["future_return_1d"].iloc[0], expected_f1)
