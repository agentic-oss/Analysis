import numpy as np
import pandas as pd
import pytest
from src.features.builder import FeatureStoreBuilder

def test_feature_builder():
    dates = pd.date_range("2025-01-01", periods=50, freq="D")
    inst_df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": 2000.0 + np.cumsum(np.random.randn(50) * 5),
        "high": 2010.0 + np.cumsum(np.random.randn(50) * 5),
        "low": 1990.0 + np.cumsum(np.random.randn(50) * 5),
        "close": 2005.0 + np.cumsum(np.random.randn(50) * 5),
        "volume": 1000
    })

    events_df = pd.DataFrame({
        "date": ["2025-01-15", "2025-02-01"],
        "event_type": ["FOMC", "CPI"]
    })

    builder = FeatureStoreBuilder(horizons=[1, 5])
    feat_df = builder.create_features_for_instrument(inst_df, macro_datasets={}, events_df=events_df)

    assert "days_to_next_event" in feat_df.columns
    assert "future_return_5d" in feat_df.columns

    # Check temporal correctness: last row future return must be NaN because future is unknown
    assert pd.isna(feat_df["future_return_5d"].iloc[-1])
