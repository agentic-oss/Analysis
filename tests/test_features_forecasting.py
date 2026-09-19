import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.scoring.scoring_engine import ScoringEngine
from src.forecasting.probabilistic_engine import ProbabilisticForecaster

def test_feature_builder_and_analogue_engine():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=120, freq="B").strftime("%Y-%m-%d")
    prices = 2000 + np.cumsum(np.random.randn(120) * 5)

    df = pd.DataFrame({
        "date": dates,
        "open": prices,
        "high": prices + 3,
        "low": prices - 3,
        "close": prices,
        "volume": 1000,
        "rsi_14": 50 + np.sin(np.linspace(0, 10, 120)) * 20,
        "macd": np.random.randn(120)
    })

    feat_df = FeatureBuilder.build_feature_dataset(df)
    assert "future_return_5d" in feat_df.columns
    assert "future_direction_5d" in feat_df.columns

    # Test Analogue search at current index 110
    analogue_res = HistoricalAnalogueEngine.find_analogues(
        df=feat_df,
        feature_cols=["rsi_14", "macd"],
        current_idx=110,
        top_k=5,
        horizons=[1, 3, 5, 10]
    )

    assert analogue_res["sample_count"] > 0
    assert "forward_stats" in analogue_res

def test_scoring_and_probabilistic_forecaster():
    scores = ScoringEngine.calculate_scores(
        metal_analysis={"rsi_14": 65, "macd": 2.5, "dist_sma_200_pct": 5.0, "volatility_20d": 14.0, "symbol": "GOLD_USD"},
        macro_regime={"primary_regime": "Risk-off"},
        ratio_analysis={"is_extreme_high": False},
        analogue_stats={"forward_stats": {"10d": {"win_rate_pct": 60.0}}},
        config={}
    )

    assert scores["composite_score"] > 50.0

    forecast_out = ProbabilisticForecaster.generate_forecasts(
        analogue_output={"sample_count": 10, "forward_stats": {"10d": {"win_rate_pct": 60.0, "mean_return_pct": 1.5}}},
        scores=scores,
        horizons=[10]
    )

    assert forecast_out["confidence_score"] > 0
    assert forecast_out["forecasts"]["10d"]["probability_positive_pct"] == 60.0
