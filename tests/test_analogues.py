"""
Tests for historical analogue search engine and look-ahead bias protection.
"""

import pytest
import pandas as pd
import numpy as np
from src.patterns.analogues import HistoricalAnalogueEngine


def test_historical_analogue_engine():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D").strftime("%Y-%m-%d")
    close = 1000 + np.cumsum(np.random.normal(0, 10, 300))
    rsi = 30 + np.random.uniform(0, 40, 300)
    dist_sma = np.random.uniform(-0.1, 0.1, 300)
    vol = np.random.uniform(0.1, 0.3, 300)
    gs_ratio = np.random.uniform(70, 90, 300)

    df = pd.DataFrame({
        "date": dates,
        "close": close,
        "rsi_14": rsi,
        "dist_sma_200_pct": dist_sma,
        "volatility_20d": vol,
        "gold_silver_ratio": gs_ratio
    })

    for h in [1, 3, 5, 10, 20, 60]:
        df[f"future_gold_ret_{h}d"] = df["close"].pct_change(h).shift(-h)

    engine = HistoricalAnalogueEngine(top_n=5, min_history_gap_days=10)
    target_date = dates[250]
    res = engine.find_analogues(df, target_date)

    assert res["target_date"] == target_date
    assert len(res["analogues"]) <= 5

    # Strict temporal check: All returned analogue dates MUST be prior to target_date - 10 days
    max_allowed_date = dates[240]
    for anal in res["analogues"]:
        assert anal["date"] <= max_allowed_date

    # Check forward stats output
    assert "5d" in res["forward_statistics"]
    assert "positive_probability_pct" in res["forward_statistics"]["5d"]
