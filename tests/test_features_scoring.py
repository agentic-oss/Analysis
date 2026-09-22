"""
Tests for FeatureBuilder and CompositeScorer.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder
from src.scoring.composite import CompositeScorer


def test_feature_builder_and_scoring():
    dates = pd.date_range("2025-01-01", periods=100, freq="D").strftime("%Y-%m-%d")
    gold_close = np.linspace(2000, 2200, 100)
    silver_close = np.linspace(25, 30, 100)
    dxy_close = np.linspace(102, 105, 100)

    gold_df = pd.DataFrame({
        "date": dates,
        "close": gold_close,
        "return_1d": np.gradient(gold_close) / gold_close,
        "rsi_14": 55.0,
        "dist_sma_200_pct": 0.05,
        "volatility_20d": 0.15,
        "macd_histogram": 0.5
    })

    silver_df = pd.DataFrame({
        "date": dates,
        "close": silver_close,
        "return_1d": np.gradient(silver_close) / silver_close,
        "rsi_14": 50.0,
        "volatility_20d": 0.18
    })

    dxy_df = pd.DataFrame({
        "date": dates,
        "close": dxy_close
    })

    features = FeatureBuilder.build_daily_features(gold_df, silver_df, {"dxy": dxy_df})

    assert "gold_price" in features.columns
    assert "silver_price" in features.columns
    assert "gold_silver_ratio" in features.columns
    assert "dxy_close" in features.columns
    assert "future_gold_ret_5d" in features.columns

    # Test Scorer
    scorer = CompositeScorer()
    scores = scorer.calculate_scores(features.iloc[50], analogue_stats={"5d": {"positive_probability_pct": 65.0}})

    assert "composite_score" in scores
    assert 0.0 <= scores["composite_score"] <= 100.0
    assert scores["sub_scores"]["historical_pattern"] == 65.0
