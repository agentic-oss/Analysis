import pytest
import pandas as pd
import numpy as np

from src.ingestion.conversions import convert_usd_oz_to_inr_10g, convert_usd_oz_to_inr_kg, process_indian_currency_conversions
from src.validation.data_validator import DataValidator
from src.indicators.technical_indicators import TechnicalIndicators
from src.ratios.gold_silver_ratio import GoldSilverRatioAnalysis
from src.regimes.macro_regimes import MacroRegimeClassifier
from src.regimes.market_regimes import MarketRegimeClassifier
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_builder import FeatureBuilder
from src.models.baseline_models import BaselineForecastingModels
from src.forecasting.probability_engine import ProbabilisticForecastEngine
from src.scoring.scoring_system import ScoringSystem, ConfidenceEngine, AlertEngine
from src.backtesting.backtest_engine import BacktestEngine
from src.evaluation.forecast_tracker import ForecastTracker


def test_conversions():
    inr_10g = convert_usd_oz_to_inr_10g(2000.0, 83.0)
    assert inr_10g is not None
    assert round(inr_10g, 2) == 53370.24

    inr_kg = convert_usd_oz_to_inr_kg(25.0, 83.0)
    assert inr_kg is not None
    assert round(inr_kg, 2) == 66712.80


def test_data_validation():
    df = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02"],
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [98.0, 100.0],
        "close": [103.0, 104.0],
        "volume": [1000, 1200],
    })
    validator = DataValidator()
    val_df, summary = validator.validate_dataset(df, "TEST_ASSET")
    assert summary["valid_count"] == 2
    assert len(summary["errors"]) == 0


def test_technical_indicators():
    dates = pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d")
    np.random.seed(123)
    close_prices = np.linspace(2000, 2100, 60) + np.random.normal(0, 5, 60)
    df = pd.DataFrame({
        "date": dates,
        "open": close_prices - 2,
        "high": close_prices + 5,
        "low": close_prices - 5,
        "close": close_prices,
        "volume": 10000,
    })
    res_df = TechnicalIndicators.calculate_indicators(df)
    assert "sma_20" in res_df.columns
    assert "rsi_14" in res_df.columns
    assert "macd" in res_df.columns
    assert "atr_14" in res_df.columns
    assert "bollinger_upper" in res_df.columns


def test_gold_silver_ratio():
    dates = pd.date_range("2025-01-01", periods=60).strftime("%Y-%m-%d")
    gold_df = pd.DataFrame({"date": dates, "close": np.linspace(2000, 2100, 60)})
    silver_df = pd.DataFrame({"date": dates, "close": np.linspace(25, 30, 60)})
    ratio_df, summary = GoldSilverRatioAnalysis.calculate_ratio_metrics(gold_df, silver_df)
    assert not ratio_df.empty
    assert "current_ratio" in summary
    assert summary["current_ratio"] > 0


def test_macro_and_market_regimes():
    macro = MacroRegimeClassifier.classify_macro_regime(0.1, 0.01, -0.05, 0.02, 25.0)
    assert macro["primary_regime"] == "Risk-Off"

    mkt = MarketRegimeClassifier.classify_market_regime(2100, 2050, 2000, 1900, 0.15, 65.0)
    assert mkt["trend_regime"] == "Strong Bullish Trend"


def test_analogue_engine():
    dates = pd.date_range("2020-01-01", periods=300).strftime("%Y-%m-%d")
    np.random.seed(42)
    hist_df = pd.DataFrame({
        "date": dates,
        "rsi_14": np.random.uniform(30, 70, 300),
        "macd_norm": np.random.normal(0, 1, 300),
        "dist_sma_20": np.random.normal(0, 2, 300),
        "dist_sma_200": np.random.normal(0, 5, 300),
        "volatility_20d": np.random.uniform(0.1, 0.3, 300),
        "gold_silver_ratio_z": np.random.normal(0, 1, 300),
        "dxy_return_20d": np.random.normal(0, 0.02, 300),
        "us10y_change_20d": np.random.normal(0, 0.1, 300),
        "vix_level": np.random.uniform(12, 30, 300),
        "future_return_10d": np.random.normal(0.01, 0.03, 300),
    })
    engine = HistoricalAnalogueEngine(top_k=10)
    curr_state = {
        "rsi_14": 50.0,
        "macd_norm": 0.0,
        "dist_sma_20": 0.0,
        "dist_sma_200": 0.0,
        "volatility_20d": 0.15,
        "gold_silver_ratio_z": 0.0,
        "dxy_return_20d": 0.0,
        "us10y_change_20d": 0.0,
        "vix_level": 18.0,
    }
    res = engine.find_analogues(hist_df, curr_state)
    assert res["top_k"] == 10
    assert "10d" in res["forward_stats"]


def test_backtest_engine():
    dates = pd.date_range("2025-01-01", periods=100).strftime("%Y-%m-%d")
    prices = np.cumprod(1 + np.random.normal(0.001, 0.01, 100)) * 100.0
    signals = [1 if i % 2 == 0 else 0 for i in range(100)]
    df = pd.DataFrame({"date": dates, "close": prices, "signal": signals})

    engine = BacktestEngine(initial_capital=10000.0)
    res = engine.run_signal_backtest(df, "signal")
    assert "final_capital" in res
    assert "sharpe_ratio" in res
    assert "max_drawdown_pct" in res
