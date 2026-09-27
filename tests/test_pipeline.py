import pytest
import pandas as pd
import numpy as np

from src.validation.validator import DataValidator
from src.indicators.technical import TechnicalIndicators
from src.ingestion.currency import convert_usd_oz_to_inr_10g, convert_usd_oz_to_inr_kg
from src.ratios.relative_value import RelativeValueAnalyzer
from src.patterns.analogue_engine import HistoricalAnalogueEngine
from src.regimes.regime_detector import MacroRegimeDetector, MarketRegimeDetector
from src.forecasting.probability_engine import ForwardProbabilityEngine
from src.features.feature_engine import FeatureEngine
from src.models.model_pipeline import ModelPipeline
from src.backtesting.backtest_engine import BacktestingEngine
from src.reporting.report_generator import ReportGenerator

@pytest.fixture
def sample_price_df():
    dates = pd.date_range("2023-01-01", periods=150, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    gold_prices = 2000.0 + np.cumsum(np.random.normal(0, 10, len(dates)))
    silver_prices = 25.0 + np.cumsum(np.random.normal(0, 0.3, len(dates)))
    usdinr = 82.0 + np.cumsum(np.random.normal(0, 0.1, len(dates)))

    df = pd.DataFrame({
        "date": dates,
        "GOLD_close": gold_prices,
        "GOLD_high": gold_prices + 5.0,
        "GOLD_low": gold_prices - 5.0,
        "GOLD_open": gold_prices,
        "GOLD_volume": 10000,
        "SILVER_close": silver_prices,
        "SILVER_high": silver_prices + 0.2,
        "SILVER_low": silver_prices - 0.2,
        "SILVER_open": silver_prices,
        "SILVER_volume": 50000,
        "USDINR_close": usdinr,
        "DXY_close": 102.0,
        "US10Y_close": 4.2,
        "CRUDE_OIL_close": 75.0,
        "SP500_close": 4500.0,
        "VIX_close": 15.0
    })
    return df

def test_currency_conversion():
    gold_usd = 2000.0
    usdinr = 80.0
    inr_10g = convert_usd_oz_to_inr_10g(gold_usd, usdinr)
    # (2000 / 31.1034768) * 80 * 10 = ~51441.3
    assert 51000 < inr_10g < 52000

    silver_usd = 25.0
    inr_kg = convert_usd_oz_to_inr_kg(silver_usd, usdinr)
    # (25 / 31.1034768) * 80 * 1000 = ~64299.1
    assert 64000 < inr_kg < 65000

def test_data_validator(sample_price_df):
    validator = DataValidator(jump_threshold_pct=0.10)
    df_val, metrics = validator.validate_series(sample_price_df, "GOLD")
    assert metrics["total_records"] == len(sample_price_df)
    assert metrics["valid_records"] > 0
    assert "quality_status" in df_val.columns

def test_technical_indicators(sample_price_df):
    tech = TechnicalIndicators.calculate_all(sample_price_df, price_col="GOLD_close")
    assert "sma_20" in tech.columns
    assert "rsi_14" in tech.columns
    assert "macd" in tech.columns
    assert "atr_14" in tech.columns
    assert not tech["rsi_14"].isna().all()

def test_relative_value(sample_price_df):
    rv = RelativeValueAnalyzer.analyze_gold_silver_ratio(sample_price_df["GOLD_close"], sample_price_df["SILVER_close"])
    assert "gold_silver_ratio" in rv.columns
    assert "ratio_zscore_252d" in rv.columns
    assert rv["gold_silver_ratio"].iloc[0] > 0

def test_regime_detection(sample_price_df):
    m_reg = MacroRegimeDetector.classify_macro_regime(sample_price_df.iloc[-1])
    assert "macro_regime" in m_reg

    mkt_df = MarketRegimeDetector.classify_market_regime(sample_price_df, prefix="gold_")
    assert "gold_trend_regime" in mkt_df.columns

def test_historical_analogue_engine(sample_price_df):
    engine = HistoricalAnalogueEngine()
    res = engine.find_analogues(sample_price_df, current_idx=len(sample_price_df)-1, price_col="GOLD_close", top_n=5)
    assert "sample_count" in res
    assert "forward_stats" in res

def test_probability_engine(sample_price_df):
    engine = ForwardProbabilityEngine()
    signals = engine.generate_probabilistic_signals(sample_price_df, current_idx=len(sample_price_df)-1, instrument="GOLD")
    assert "horizons" in signals
    assert "10d" in signals["horizons"]

def test_feature_engine(sample_price_df):
    fe = FeatureEngine()
    f_df = fe.build_features(sample_price_df)
    assert "GOLD_ret_1d" in f_df.columns
    assert "target_gold_future_ret_5d" in f_df.columns

def test_model_pipeline(sample_price_df):
    fe = FeatureEngine()
    f_df = fe.build_features(sample_price_df)
    f_df["feature_1"] = f_df["GOLD_close"].pct_change()
    f_df["feature_2"] = f_df["SILVER_close"].pct_change()

    pipeline = ModelPipeline(feature_cols=["feature_1", "feature_2"], target_col="target_gold_future_dir_5d")
    results = pipeline.walk_forward_evaluate(f_df, min_train_size=50, step_size=10)
    assert isinstance(results, dict)

def test_backtest_engine(sample_price_df):
    sample_price_df["signal"] = 1
    engine = BacktestingEngine()
    bt_res = engine.run_signal_backtest(sample_price_df, signal_col="signal", price_col="GOLD_close")
    assert "cumulative_return" in bt_res
    assert "sharpe_ratio" in bt_res

def test_report_generator():
    rg = ReportGenerator(reports_dir="reports/test", json_dir="data/analysis/test")
    md_p, json_p = rg.generate_daily_report(
        date_str="2026-03-30",
        gold_summary={"price": 2000.0},
        silver_summary={"price": 25.0},
        ratio_summary={"current_ratio": 80.0},
        macro_summary={"dxy": 100.0},
        scores={"gold_scores": {"composite_score": 60}},
        forecast_stats={},
        quality_metrics={"valid_records": 100}
    )
    assert md_p.endswith(".md")
    assert json_p.endswith(".json")
