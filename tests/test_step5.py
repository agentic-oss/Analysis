import pytest
import pandas as pd
import numpy as np
from src.features.builder import FeatureBuilder
from src.scoring.composite_score import CompositeScorer
from src.forecasting.probabilistic import ProbabilisticForecaster

def test_feature_builder():
    dates = pd.date_range("2026-01-01", periods=30, freq="B").strftime("%Y-%m-%d")
    close_prices = np.linspace(100, 130, 30)
    df = pd.DataFrame({
        "date": dates,
        "close": close_prices,
        "rsi_14": [50.0] * 30
    })

    feat_df = FeatureBuilder.build_daily_features(df, "GOLD", horizons=[1, 5])
    assert "return_1d" in feat_df.columns
    assert "future_return_1d" in feat_df.columns
    assert "future_direction_1d" in feat_df.columns
    assert len(feat_df) == 30

def test_composite_scorer_and_forecaster():
    latest_row = {"symbol": "GOLD", "rsi_14": 55.0, "dist_sma_200": 0.05, "golden_cross": True}
    macro_regime = {"primary_regime": "Risk-Off"}
    ratio_analysis = {"zscore": 1.0}
    analogue_stats = {"forward_statistics": {"20d": {"positive_probability_pct": 65.0}}}

    scores = CompositeScorer.calculate_scores(latest_row, macro_regime, ratio_analysis, analogue_stats)
    assert "composite_score" in scores
    assert 0 <= scores["composite_score"] <= 100

    forecast = ProbabilisticForecaster.generate_probabilistic_signal(
        "GOLD", analogue_stats.get("forward_statistics", {}), scores
    )
    assert "research_bias" in forecast
    assert "disclaimer" in forecast
    assert "20d" in forecast["horizons"]
