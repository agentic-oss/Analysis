"""
Unit Tests for Gold & Silver Pipeline.
"""

import pandas as pd
import numpy as np

from src.ingestion.provider import convert_usd_to_inr_gold_silver
from src.validation.validator import DataValidator
from src.indicators.technical import calculate_technical_indicators, detect_technical_signals
from src.ratios.silver_ratio import SilverRelativeValueAnalyzer
from src.regimes.classifier import RegimeClassifier
from src.patterns.analogue import AnalogueEngine
from src.features.builder import FeatureStoreBuilder
from src.forecasting.models import MetalsForecaster
from src.backtesting.engine import Backtester
from src.scoring.engine import ScoringAndAlertEngine


def test_currency_conversion():
    res = convert_usd_to_inr_gold_silver(2000.0, 83.0, "oz")
    assert "inr_per_10g" in res
    assert res["inr_per_10g"] > 50000.0


def test_validator():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [100.0, -10.0],
        "high": [105.0, 110.0],
        "low": [98.0, 108.0],
        "close": [102.0, 105.0]
    })
    val_df, warnings, errors = validator.validate_series(df, "GOLD")
    assert len(errors) > 0
    assert val_df["quality_classification"].iloc[1] == "invalid"


def test_technical_indicators():
    dates = pd.bdate_range("2023-01-01", "2024-01-01")
    df = pd.DataFrame({
        "date": dates,
        "open": 100 + np.arange(len(dates)),
        "high": 102 + np.arange(len(dates)),
        "low": 99 + np.arange(len(dates)),
        "close": 101 + np.arange(len(dates)),
        "volume": 1000.0
    })
    df_ind = calculate_technical_indicators(df)
    assert "sma_50" in df_ind.columns
    assert "rsi_14" in df_ind.columns
    signals = detect_technical_signals(df_ind)
    assert "golden_cross" in signals


def test_gold_silver_ratio():
    dates = pd.bdate_range("2023-01-01", "2024-01-01")
    gold = pd.DataFrame({"date": dates, "close": 2000 + np.arange(len(dates))})
    silver = pd.DataFrame({"date": dates, "close": 25 + np.arange(len(dates)) * 0.1})
    analyzer = SilverRelativeValueAnalyzer()
    ratio_df = analyzer.calculate_ratio_metrics(gold, silver)
    assert "gold_silver_ratio" in ratio_df.columns
    extremes = analyzer.analyze_ratio_extremes(ratio_df)
    assert "current_ratio" in extremes


def test_regime_classifier():
    classifier = RegimeClassifier()
    res = classifier.classify_macro_regime(rates_change_20d=0.2, cpi_surprise=0.2)
    assert res["macro_regime"] == "INFLATIONARY"


def test_analogue_engine():
    dates = pd.bdate_range("2020-01-01", "2024-01-01")
    df = pd.DataFrame({
        "date": dates,
        "close": 100 + np.cumsum(np.random.normal(0, 1, len(dates))),
        "rsi_14": np.random.uniform(30, 70, len(dates)),
        "dist_sma_50_pct": np.random.normal(0, 2, len(dates)),
        "volatility_20d": 15.0,
        "gold_silver_ratio": 80.0,
        "future_return_5d": np.random.normal(0.5, 2.0, len(dates))
    })
    engine = AnalogueEngine()
    res = engine.find_analogues(df, top_k=5)
    assert len(res["top_analogues"]) == 5


def test_backtester():
    dates = pd.bdate_range("2023-01-01", "2024-01-01")
    df = pd.DataFrame({
        "date": dates,
        "close": 100 + np.cumsum(np.random.normal(0.1, 1, len(dates))),
        "signal": np.random.choice([1, 0, -1], len(dates))
    })
    backtester = Backtester()
    res = backtester.run_backtest(df)
    assert "sharpe_ratio" in res
