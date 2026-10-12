import pytest
import pandas as pd
import numpy as np
from src.ingestion.provider import SyntheticMarketDataProvider
from src.features.builder import DailyFeatureStoreBuilder
from src.models.forecaster import TimeSeriesModelEvaluator
from src.forecasting.confidence import PredictionConfidenceFramework
from src.backtesting.engine import BacktestEngine

def test_feature_builder():
    provider = SyntheticMarketDataProvider(seed=42)
    gold_df = provider.fetch_historical_series("GC=F", "2025-01-01", "2026-01-01")
    silver_df = provider.fetch_historical_series("SI=F", "2025-01-01", "2026-01-01")

    builder = DailyFeatureStoreBuilder()
    features_df = builder.build_feature_dataset(gold_df, silver_df)

    assert "rsi_14" in features_df.columns
    assert "gold_silver_ratio" in features_df.columns
    assert "future_return_5d" in features_df.columns
    assert len(features_df) == len(gold_df)

def test_walk_forward_evaluator():
    provider = SyntheticMarketDataProvider(seed=42)
    gold_df = provider.fetch_historical_series("GC=F", "2024-01-01", "2026-01-01")
    silver_df = provider.fetch_historical_series("SI=F", "2024-01-01", "2026-01-01")

    builder = DailyFeatureStoreBuilder()
    features_df = builder.build_feature_dataset(gold_df, silver_df)

    feature_cols = ["rsi_14", "dist_sma_20", "hist_vol_20", "gold_silver_ratio"]
    evaluator = TimeSeriesModelEvaluator(features=feature_cols, target_horizon=5, min_train_size=100, step_size=20)
    res = evaluator.walk_forward_validation(features_df)

    assert res["status"] == "success"
    assert "classification" in res
    assert "accuracy" in res["classification"]

def test_confidence_framework():
    framework = PredictionConfidenceFramework()
    res = framework.calculate_confidence(
        analogue_sample_size=15,
        analogue_positive_prob=68.0,
        model_agreement_pct=0.8,
        macro_alignment=True,
        current_volatility=0.15
    )
    assert res["confidence_score"] > 60.0
    assert len(res["reasons"]) > 0

def test_backtest_engine():
    dates = pd.date_range("2025-01-01", periods=100, freq="B").astype(str)
    np.random.seed(42)
    prices = 2000.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 100)))
    signals = np.random.choice([1, 0, -1], size=100)

    df = pd.DataFrame({"date": dates, "close": prices, "signal": signals})
    engine = BacktestEngine()
    res = engine.run_backtest(df, signal_column="signal")

    assert res["status"] == "success"
    assert "sharpe_ratio" in res
    assert "cumulative_return_pct" in res
