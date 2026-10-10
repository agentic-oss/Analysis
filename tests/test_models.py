import pytest
import pandas as pd
from src.ingestion.synthetic_provider import SyntheticProvider
from src.features.feature_builder import FeatureBuilder
from src.models.forecaster import MetalsForecaster
from src.forecasting.probabilistic_engine import ProbabilisticEngine

def test_models_and_probabilistic_signals():
    provider = SyntheticProvider(seed=10)
    gold_df = provider.get_spot_prices("GOLD", "2022-01-01", "2023-06-01")
    feat_df = FeatureBuilder.build_feature_dataset("GOLD", gold_df, {})

    forecaster = MetalsForecaster()
    pred = forecaster.train_and_predict(feat_df, horizon_days=5)

    assert "direction_prob" in pred
    assert 0.0 <= pred["direction_prob"] <= 1.0

    analogue_stats = {"5d": {"prob_positive_pct": 60.0, "mean_return_pct": 1.5, "sample_count": 5}}
    signal = ProbabilisticEngine.generate_probabilistic_signals(5, analogue_stats, pred, gold_df["Close"].pct_change())

    assert "prob_positive_return" in signal
    assert "confidence_score" in signal
    assert "disclaimer" in signal
