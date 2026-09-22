"""
Tests for baseline ML forecasting models and probabilistic signal engine.
"""

import pytest
import pandas as pd
import numpy as np
from src.models.baseline import MetalsForecastingModels
from src.forecasting.signals import ProbabilisticSignalEngine


def test_models_and_signals():
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=150, freq="D").strftime("%Y-%m-%d")
    close = 2000.0 + np.cumsum(np.random.normal(0.5, 5.0, 150))
    rsi = 40.0 + np.random.uniform(0, 30, 150)
    macd = np.random.uniform(-2, 2, 150)

    df = pd.DataFrame({
        "date": dates,
        "gold_price": close,
        "rsi_14": rsi,
        "macd": macd,
    })
    df["future_gold_ret_5d"] = df["gold_price"].pct_change(5).shift(-5)

    models = MetalsForecastingModels(target_horizon_days=5)
    pred_res = models.train_and_predict(
        df,
        feature_cols=["rsi_14", "macd"],
        target_col="future_gold_ret_5d",
        current_date=dates[120],
    )

    assert "ensemble_positive_probability" in pred_res
    assert 0.0 <= pred_res["ensemble_positive_probability"] <= 1.0

    # Test Signal Engine
    signal = ProbabilisticSignalEngine.generate_signal(
        instrument="GOLD",
        current_price=2100.0,
        composite_score=62.0,
        analogue_stats={"5d": {"positive_probability_pct": 68.0, "mean_return_pct": 1.5, "sample_count": 12}},
        model_predictions=pred_res,
        market_regime="Bullish Trend / Low Volatility",
    )

    assert signal["instrument"] == "GOLD"
    assert "horizon_signals" in signal
    assert "5d" in signal["horizon_signals"]
    assert "disclaimer" in signal["horizon_signals"]["5d"]
    assert signal["horizon_signals"]["5d"]["confidence_score"] > 0
