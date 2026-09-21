import pytest
from src.features.builder import FeatureBuilder
from src.models.baseline import TimeSeriesModelEvaluator
from src.ingestion.collector import DataCollector
from src.indicators.technical import TechnicalIndicators


def test_features_and_models():
    collector = DataCollector(use_fallback=True)
    data = collector.collect_all("2024-01-01", "2026-01-15")
    gold_ti = TechnicalIndicators.calculate_all(data["GOLD"])

    features = FeatureBuilder.build_daily_features("GOLD", gold_ti, data)
    assert not features.empty
    assert "target_future_return_5d" in features.columns
    assert "target_future_direction_5d" in features.columns

    eval_res = TimeSeriesModelEvaluator.train_and_evaluate_walk_forward(features, horizon=5)
    assert eval_res["status"] == "success"
    assert "random_forest_accuracy_pct" in eval_res
