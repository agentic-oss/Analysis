"""
Automated Test Suite for Gold & Silver Pipeline.
Tests data ingestion, validation, technical indicators, gold/silver relative value,
macro regime detection, feature builder, analogue search, backtesting, and reporting.
"""
import pytest
import pandas as pd
import numpy as np

from src.ingestion.provider import SyntheticMarketDataProvider
from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalAnalysisEngine
from src.metals.precious import GoldAnalyzer, SilverAnalyzer
from src.ratios.relative_value import RelativeValueAnalyzer
from src.regimes.macro_regime import MacroRegimeDetector
from src.regimes.market_regime import MarketRegimeDetector
from src.features.builder import FeatureBuilder, EventFeatureGenerator
from src.patterns.analogue import HistoricalAnalogueEngine
from src.scoring.composite import FactorScorer, AlertDetector
from src.forecasting.signals import ProbabilisticForecastEngine, ModelTrainer, BacktestingEngine
from src.evaluation.evaluator import ForecastEvaluator
from src.reporting.generator import ReportGenerator


@pytest.fixture
def synthetic_provider():
    return SyntheticMarketDataProvider()


@pytest.fixture
def gold_df(synthetic_provider):
    return synthetic_provider.fetch_historical_prices("GC=F", "2023-01-01", "2024-01-01")


@pytest.fixture
def silver_df(synthetic_provider):
    return synthetic_provider.fetch_historical_prices("SI=F", "2023-01-01", "2024-01-01")


def test_data_ingestion_and_validation(gold_df):
    assert not gold_df.empty
    assert "close" in gold_df.columns
    assert "symbol" in gold_df.columns
    assert gold_df["symbol"].iloc[0] == "GC=F"

    validator = DataValidator(max_daily_jump_pct=15.0)
    df_val, report = validator.validate_dataset(gold_df, "GC=F")
    assert "quality_status" in df_val.columns
    assert report["record_count"] == len(gold_df)


def test_technical_indicators(gold_df):
    tech_df = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    assert "sma_20" in tech_df.columns
    assert "sma_50" in tech_df.columns
    assert "sma_200" in tech_df.columns
    assert "rsi_14" in tech_df.columns
    assert "macd" in tech_df.columns
    assert "atr_14" in tech_df.columns
    assert "volatility_20d" in tech_df.columns
    assert "bollinger_upper" in tech_df.columns
    assert tech_df["rsi_14"].between(0, 100).all()


def test_gold_silver_relative_value(gold_df, silver_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    s_tech = TechnicalAnalysisEngine.compute_all_indicators(silver_df)
    rv_df = RelativeValueAnalyzer.compute_relative_value(g_tech, s_tech)
    assert not rv_df.empty
    assert "gold_silver_ratio" in rv_df.columns
    assert "ratio_zscore_60d" in rv_df.columns
    assert (rv_df["gold_silver_ratio"] > 0).all()

    outcomes = RelativeValueAnalyzer.calculate_forward_spread_outcomes(rv_df)
    assert "latest_ratio" in outcomes
    assert "forward_horizons" in outcomes


def test_macro_and_market_regimes(synthetic_provider, gold_df):
    dxy = synthetic_provider.fetch_historical_prices("DX-Y.NYB", "2023-01-01", "2024-01-01")
    us10y = synthetic_provider.fetch_historical_prices("^TNX", "2023-01-01", "2024-01-01")
    oil = synthetic_provider.fetch_historical_prices("CL=F", "2023-01-01", "2024-01-01")
    sp500 = synthetic_provider.fetch_historical_prices("^GSPC", "2023-01-01", "2024-01-01")
    vix = synthetic_provider.fetch_historical_prices("^VIX", "2023-01-01", "2024-01-01")

    macro_res = MacroRegimeDetector.detect_regime(dxy, us10y, oil, sp500, vix)
    assert "macro_regime" in macro_res
    assert macro_res["macro_regime"] in [
        "Inflationary", "Disinflationary", "Deflationary", "Risk-On", "Risk-Off",
        "Tightening", "Easing", "Stagflationary", "Neutral"
    ]

    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    mkt_res = MarketRegimeDetector.classify_market_regime(g_tech)
    assert "trend_regime" in mkt_res
    assert "volatility_regime" in mkt_res


def test_feature_builder_and_no_lookahead_bias(gold_df, silver_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    s_tech = TechnicalAnalysisEngine.compute_all_indicators(silver_df)
    rv_df = RelativeValueAnalyzer.compute_relative_value(g_tech, s_tech)

    feat_df = FeatureBuilder.build_daily_features(g_tech, {}, rv_df)
    assert "gold_silver_ratio" in feat_df.columns
    assert "future_return_5d" in feat_df.columns

    # Verify look-ahead target is shifted properly
    last_row = feat_df.iloc[-1]
    assert pd.isna(last_row["future_return_5d"])


def test_event_features(gold_df):
    calendar = [
        {"date": "2023-06-14", "event_type": "FOMC"},
        {"date": "2023-07-12", "event_type": "CPI"}
    ]
    evt_df = EventFeatureGenerator.attach_event_features(gold_df, calendar)
    assert "days_to_event" in evt_df.columns
    assert "days_since_event" in evt_df.columns


def test_historical_analogue_engine(gold_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    feat_df = FeatureBuilder.build_daily_features(g_tech, {}, pd.DataFrame())

    engine = HistoricalAnalogueEngine(top_n=10)
    res = engine.find_analogues(feat_df)
    assert res["analogue_count"] > 0
    assert len(res["top_analogues"]) <= 10
    assert "forward_statistics" in res


def test_scoring_alerts_forecasting(gold_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    feat_df = FeatureBuilder.build_daily_features(g_tech, {}, pd.DataFrame())
    analogue_engine = HistoricalAnalogueEngine()
    analogue_res = analogue_engine.find_analogues(feat_df)

    scores = FactorScorer.calculate_scores(g_tech, {"macro_regime": "Risk-Off"}, pd.DataFrame(), analogue_res)
    assert 0 <= scores["composite_score"] <= 100

    alerts = AlertDetector.detect_alerts("2024-01-01", "GOLD", g_tech)
    assert isinstance(alerts, list)

    signals = ProbabilisticForecastEngine.generate_probabilistic_signals("GOLD", feat_df, analogue_res, scores)
    assert "signals" in signals
    assert "5d" in signals["signals"]


def test_backtesting_engine(gold_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    feat_df = FeatureBuilder.build_daily_features(g_tech, {}, pd.DataFrame())
    signal = (feat_df["rsi_14"] < 45).astype(int)

    engine = BacktestingEngine()
    bt_res = engine.run_signal_backtest(feat_df, signal)
    assert "cumulative_return_pct" in bt_res
    assert "sharpe_ratio" in bt_res
    assert "max_drawdown_pct" in bt_res


def test_reporting(gold_df):
    g_tech = TechnicalAnalysisEngine.compute_all_indicators(gold_df)
    json_rep = ReportGenerator.generate_json_report(
        date_str="2024-01-01",
        gold_analysis={"price": 2000.0, "composite_score": 60.0},
        silver_analysis={"price": 24.0, "composite_score": 55.0},
        relative_value={"latest_ratio": 83.3},
        macro_regime={"macro_regime": "Risk-Off"},
        market_regimes={"gold": {}, "silver": {}},
        quality_report={"valid_count": 100},
        alerts=[]
    )
    assert json_rep["date"] == "2024-01-01"

    md_rep = ReportGenerator.generate_markdown_report(
        date_str="2024-01-01",
        gold_analysis=json_rep["gold"],
        silver_analysis=json_rep["silver"],
        relative_value=json_rep["gold_silver_ratio"],
        macro_regime=json_rep["macro"],
        gold_regime={"composite_regime": "Bullish"},
        silver_regime={"composite_regime": "Bullish"},
        gold_analogues={"analogue_count": 5, "top_analogues": []},
        silver_analogues={"analogue_count": 5, "top_analogues": []},
        quality_report=json_rep["data_quality"],
        alerts=[]
    )
    assert "# Precious Metals Daily Research Report" in md_rep
