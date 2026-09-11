"""
Unit tests for Historical Analogue Engine, Probabilistic Signals, Feature Store, ML Models, and Backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.forecasting.probabilistic import ProbabilisticSignalEngine
from src.features.feature_store import FeatureStoreEngine
from src.models.forecasting_models import MetalsForecastingModel, BacktestEngine, ForecastEvaluationTracker


@pytest.fixture
def sample_feature_df():
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    np.random.seed(42)
    prices = 2000.0 * np.cumprod(1 + np.random.normal(0.0005, 0.01, 200))

    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "close": prices,
        "rsi_14": np.random.uniform(30, 70, 200),
        "macd": np.random.normal(0, 5, 200),
        "macd_hist": np.random.normal(0, 2, 200),
        "volatility_20d": np.random.uniform(0.1, 0.25, 200),
        "dist_sma_20_pct": np.random.normal(0, 2, 200),
        "dist_sma_50_pct": np.random.normal(0, 3, 200),
        "dist_sma_200_pct": np.random.normal(0, 5, 200),
        "return_1d": np.random.normal(0, 0.01, 200),
        "return_5d": np.random.normal(0, 0.02, 200),
        "return_20d": np.random.normal(0, 0.04, 200),
        "gold_silver_ratio": np.random.uniform(75, 85, 200),
        "ratio_zscore": np.random.normal(0, 1, 200),
    })

    fs = FeatureStoreEngine(forecast_horizons=[1, 3, 5, 10, 20, 60])
    return fs.build_daily_features(df)


def test_analogue_engine(sample_feature_df):
    engine = HistoricalAnalogueEngine(top_k=10)
    result = engine.find_analogues(sample_feature_df, current_idx=150, min_history_gap=20)

    assert "analogues" in result
    assert "forward_stats" in result
    assert "5d" in result["forward_stats"]
    assert len(result["analogues"]) <= 10


def test_probabilistic_signals():
    analogue_stats = {
        "5d": {"sample_count": 10, "prob_positive": 0.70, "expected_return": 0.015}
    }
    engine = ProbabilisticSignalEngine()
    signals = engine.generate_probabilistic_signals(analogue_stats, 75.0, 65.0, {})

    assert "5d" in signals
    assert "Bullish" in signals["5d"]["direction"]
    assert signals["5d"]["prob_positive"] > 0.5


def test_feature_store_lookahead_protection(sample_feature_df):
    # Verify future targets are strictly shifted
    row_100 = sample_feature_df.iloc[100]
    row_105 = sample_feature_df.iloc[105]

    expected_return = (row_105["close"] - row_100["close"]) / row_100["close"]
    assert pytest.approx(row_100["future_return_5d"], rel=1e-5) == expected_return


def test_forecasting_model_and_backtest(sample_feature_df):
    model = MetalsForecastingModel(model_type="rf", horizon=5)
    res = model.train_and_evaluate_walk_forward(sample_feature_df, min_train_size=100, step_size=20)

    assert res["accuracy"] >= 0.0
    assert "predictions" in res

    # Test backtest
    sample_feature_df["signal"] = np.where(sample_feature_df["rsi_14"] < 45, 1, -1)
    bt = BacktestEngine()
    bt_res = bt.run_signal_backtest(sample_feature_df, "signal")

    assert "final_capital" in bt_res
    assert "sharpe_ratio" in bt_res
