"""Unit and integration test suite for Precious Metals Research Pipeline."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.ingestion.provider import YahooFinanceProvider
from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalIndicators
from src.metals.analyzer import MetalAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.macro.regime import MacroRegimeClassifier
from src.regimes.market import MarketRegimeClassifier
from src.patterns.analogue import HistoricalAnalogueEngine
from src.forecasting.engine import ForwardProbabilityEngine
from src.features.generator import FeatureStoreGenerator
from src.models.forecaster import MetalsForecastingModels
from src.backtesting.engine import BacktestEngine
from src.evaluation.tracker import ForecastEvaluator
from src.scoring.composite import CompositeScorer
from src.scoring.alerts import AlertDetector
from src.reporting.generator import ReportGenerator


@pytest.fixture
def dummy_ohlcv_data():
    dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    close_prices = 2000.0 + np.cumsum(np.random.normal(1.0, 10.0, size=100))
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices - 2.0,
        "high": close_prices + 5.0,
        "low": close_prices - 5.0,
        "close": close_prices,
        "volume": 10000
    })
    return df


def test_data_validation(dummy_ohlcv_data):
    validator = DataValidator()
    val_df, rep = validator.validate_dataframe(dummy_ohlcv_data, "GOLD")
    assert rep["valid_records"] == 100
    assert "classification" in val_df.columns


def test_technical_indicators(dummy_ohlcv_data):
    df_tech = TechnicalIndicators.calculate_all(dummy_ohlcv_data)
    assert "sma_50" in df_tech.columns
    assert "sma_200" in df_tech.columns
    assert "rsi_14" in df_tech.columns
    assert "macd" in df_tech.columns
    assert "atr_14" in df_tech.columns


def test_gold_silver_ratio(dummy_ohlcv_data):
    silver_df = dummy_ohlcv_data.copy()
    silver_df["close"] = silver_df["close"] / 80.0
    ratio_df = RelativeValueAnalyzer.calculate_ratio(dummy_ohlcv_data, silver_df)
    assert "gold_silver_ratio" in ratio_df.columns
    assert ratio_df["gold_silver_ratio"].iloc[0] == pytest.approx(80.0, 0.1)


def test_feature_store_generator(dummy_ohlcv_data):
    features_df = FeatureStoreGenerator.build_feature_dataset("GOLD", dummy_ohlcv_data, pd.DataFrame(), {})
    assert "return_1d" in features_df.columns
    assert "future_return_5d" in features_df.columns
    assert "future_direction_5d" in features_df.columns


def test_analogue_engine_no_lookahead(dummy_ohlcv_data):
    df_tech = TechnicalIndicators.calculate_all(dummy_ohlcv_data)
    features_df = FeatureStoreGenerator.build_feature_dataset("GOLD", df_tech, pd.DataFrame(), {})

    engine = HistoricalAnalogueEngine()
    target_date = features_df["date"].iloc[80]
    res = engine.find_analogues(features_df, target_date)

    for match in res.get("top_analogues", []):
        assert match["date"] < target_date


def test_backtest_engine(dummy_ohlcv_data):
    engine = BacktestEngine()
    dummy_ohlcv_data["signal"] = 1.0
    res = engine.run_backtest(dummy_ohlcv_data, "signal")
    assert "cumulative_return_pct" in res
    assert "sharpe_ratio" in res


def test_forecast_evaluator(dummy_ohlcv_data):
    evaluator = ForecastEvaluator()
    rec = evaluator.record_forecast("2024-01-05", "GOLD", "5d", "BULLISH", 1.5, 75.0)
    res = evaluator.evaluate_historical_forecasts(dummy_ohlcv_data, [rec])
    assert res["evaluated_count"] == 1


def test_composite_scorer():
    scores = CompositeScorer.calculate_scores({"dist_sma_200_pct": 5.0}, {"primary_regime": "Risk-off"}, {}, {})
    assert 0 <= scores["composite_score"] <= 100
