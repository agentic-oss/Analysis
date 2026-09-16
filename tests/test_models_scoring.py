import numpy as np
import pandas as pd
import pytest
from src.models.baseline import MetalsForecastingModels
from src.scoring.engine import ScoringEngine

def test_forecasting_models():
    dates = pd.date_range("2024-01-01", periods=300, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": 2000.0 + np.cumsum(np.random.randn(300) * 5),
        "rsi_14": 50 + np.random.randn(300) * 10,
        "dist_sma_200": np.random.randn(300) * 0.02,
        "future_return_5d": np.random.randn(300) * 0.01,
        "future_direction_5d": np.random.choice([0, 1], 300)
    })

    models = MetalsForecastingModels(target_horizon=5)
    res = models.train_walk_forward(df, feature_cols=["rsi_14", "dist_sma_200"], train_window=200, test_window=20)

    assert res["status"] == "success"
    assert "accuracy" in res["metrics"]

    curr_pred = models.predict_current(df, feature_cols=["rsi_14", "dist_sma_200"])
    assert "prob_positive" in curr_pred

def test_scoring_engine():
    engine = ScoringEngine()
    scores = engine.compute_all_scores(
        technical_metrics={"dist_sma_200": 0.05, "rsi_14": 60, "macd_hist": 1.2, "volatility_20d": 0.15},
        macro_metrics={"sentiment": "Risk-Off", "dollar_stance": "Dollar Weak"},
        relative_value_metrics={"gs_ratio_zscore": -1.0},
        analogue_metrics={"forward_statistics": {"5d": {"prob_positive": 0.65, "sample_count": 25}}},
        model_metrics={"prob_positive": 0.62}
    )

    assert "composite_score" in scores
    assert "confidence_score" in scores
    assert len(scores["confidence_reasons"]) > 0
