"""
Tests for Gold, Silver, and Relative Value analysis modules.
"""

import pytest
import pandas as pd
import numpy as np
from src.metals.gold import GoldAnalysis
from src.metals.silver import SilverAnalysis
from src.ratios.relative_value import RelativeValueAnalysis


def test_gold_silver_relative_value():
    dates = pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    gold_close = np.linspace(2000, 2200, 100) + np.random.normal(0, 5, 100)
    silver_close = np.linspace(25, 30, 100) + np.random.normal(0, 0.2, 100)
    dxy_close = np.linspace(102, 105, 100) + np.random.normal(0, 0.1, 100)

    gold_df = pd.DataFrame({"date": dates, "close": gold_close})
    silver_df = pd.DataFrame({"date": dates, "close": silver_close})
    dxy_df = pd.DataFrame({"date": dates, "close": dxy_close})

    # Test Gold rolling correlation
    gold_analysis = GoldAnalysis(windows=[20])
    gold_corr = gold_analysis.calculate_rolling_correlations(gold_df, {"dxy": dxy_df})
    assert "corr_gold_dxy_20d" in gold_corr.columns

    # Test Relative Value
    rv = RelativeValueAnalysis(lookback_window=30)
    rv_df = rv.analyze_ratio(gold_df, silver_df)
    assert "gold_silver_ratio" in rv_df.columns
    assert "ratio_zscore_252d" in rv_df.columns
    assert "spread_1d" in rv_df.columns

    # Forward stats check
    fwd_stats = RelativeValueAnalysis.calculate_forward_return_stats(rv_df, "future_gold_ret_5d")
    assert "sample_count" in fwd_stats
    assert "win_rate_pct" in fwd_stats
