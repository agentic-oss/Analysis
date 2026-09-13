import pytest
import pandas as pd
import numpy as np

from src.storage.manager import StorageManager
from src.ingestion.yfinance_provider import MockMarketDataProvider
from src.ingestion.indian_conversions import convert_usd_oz_to_inr_10g, convert_usd_oz_to_inr_kg, enrich_indian_conversions
from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalIndicators
from src.metals.correlation import MetalCorrelationAnalyzer
from src.ratios.ratio_analysis import GoldSilverRatioAnalyzer
from src.regimes.regime_classifier import MacroRegimeClassifier, MarketRegimeClassifier
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.features.feature_store import FeatureStoreBuilder
from src.models.forecasting_models import MetalsForecastingModels
from src.forecasting.forecast_engine import ForecastEngine
from src.scoring.scoring_system import TransparentScoringSystem
from src.backtesting.backtester import BacktestingEngine
from src.evaluation.performance_tracker import ForecastPerformanceTracker

def test_storage_manager(tmp_path):
    sm = StorageManager(base_data_dir=str(tmp_path / "data"), base_reports_dir=str(tmp_path / "reports"))
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    path = sm.save_dataframe(df, "test.csv", format="csv")
    loaded = sm.load_dataframe("test.csv", format="csv")
    assert len(loaded) == 2

def test_indian_conversions():
    # Gold: $2000/oz, USDINR = 83.0 -> 2000 / 31.1034768 * 83.0 * 10
    inr_10g = convert_usd_oz_to_inr_10g(2000.0, 83.0)
    assert round(inr_10g, 1) == 53370.2

    # Silver: $25/oz, USDINR = 83.0 -> 25 / 31.1034768 * 83.0 * 1000
    inr_kg = convert_usd_oz_to_inr_kg(25.0, 83.0)
    assert round(inr_kg, 1) == 66712.8

def test_data_validator():
    provider = MockMarketDataProvider()
    df = provider.fetch_historical("GOLD", "GC=F", "2024-01-01", "2024-02-01")
    validator = DataValidator()
    val_df, summary = validator.validate_series(df, "GOLD")
    assert summary["valid_count"] == len(df)
    assert "quality_classification" in val_df.columns

def test_technical_indicators():
    provider = MockMarketDataProvider()
    df = provider.fetch_historical("GOLD", "GC=F", "2023-01-01", "2024-01-01")
    ind_df = TechnicalIndicators.calculate_all(df)
    assert "sma_50" in ind_df.columns
    assert "rsi_14" in ind_df.columns
    assert "macd" in ind_df.columns
    assert "atr" in ind_df.columns

def test_regime_classification():
    macro_res = MacroRegimeClassifier.classify_macro_regime(0.2, 0.1, -0.02, 16.0, 0.05, 0.02)
    assert "macro_regime" in macro_res
    market_res = MarketRegimeClassifier.classify_market_regime(2000, 1950, 1900, 65, 15.0)
    assert market_res["trend_regime"] == "Strong Bullish Trend"

def test_analogue_engine():
    provider = MockMarketDataProvider()
    gold = provider.fetch_historical("GOLD", "GC=F", "2020-01-01", "2023-01-01")
    fs = FeatureStoreBuilder()
    f_df = fs.build_daily_features("GOLD", gold, {})
    engine = HistoricalAnalogueEngine()
    res = engine.find_analogues(f_df, f_df.index[-1].strftime("%Y-%m-%d"))
    assert "forward_statistics" in res
    assert "1d" in res["forward_statistics"]

def test_forecasting_and_backtesting():
    provider = MockMarketDataProvider()
    gold = provider.fetch_historical("GOLD", "GC=F", "2020-01-01", "2023-01-01")
    fs = FeatureStoreBuilder()
    f_df = fs.build_daily_features("GOLD", gold, {})

    val_res = MetalsForecastingModels.walk_forward_validation(f_df, target_horizon=5)
    assert "walk_forward_accuracy" in val_res

    backtester = BacktestingEngine()
    sig = np.sign(f_df["return_1d"]).fillna(0)
    bt = backtester.run_backtest(f_df["Close"], sig)
    assert "sharpe_ratio" in bt
