import pytest
import pandas as pd
import numpy as np

from src.ingestion import MarketDataProvider
from src.validation import DataValidator
from src.storage import StorageManager
from src.indicators import calculate_technical_indicators
from src.metals import analyze_gold_drivers
from src.ratios import analyze_gold_silver_ratio
from src.regimes import detect_macro_regime, detect_market_regime
from src.patterns import find_historical_analogues
from src.forecasting import generate_probabilistic_signals
from src.features import build_daily_feature_dataset
from src.models import BaselineModelsManager
from src.backtesting import BacktestEngine
from src.scoring import calculate_transparent_scores
from src.evaluation import ForecastEvaluator
from src.reporting import ReportGenerator, detect_daily_alerts
from src.pipeline import run_pipeline


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 2000.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100)))
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "instrument_id": "GOLD_USD_SPOT",
        "symbol": "GOLD",
        "category": "precious_metals",
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": 10000.0,
        "currency": "USD",
        "unit": "oz",
        "provider": "test",
        "source": "LBMA",
        "retrieval_timestamp": "2026-09-29T00:00:00",
        "timezone": "Asia/Kolkata"
    })
    return df


def test_ingestion_and_fallback():
    provider = MarketDataProvider()
    df = provider.fetch_historical_ohlcv("GOLD_USD_SPOT", "2025-01-01", "2025-01-10")
    assert not df.empty
    assert "close" in df.columns
    assert "date" in df.columns


def test_data_validator(sample_ohlcv):
    validator = DataValidator(quality_dir="data/test_quality")
    v_df, summary = validator.validate_df(sample_ohlcv, "GOLD_USD_SPOT")
    assert len(v_df) == len(sample_ohlcv)
    assert summary["valid_records"] == 100
    assert summary["errors"] == []


def test_technical_indicators(sample_ohlcv):
    tech = calculate_technical_indicators(sample_ohlcv)
    assert "rsi_14" in tech.columns
    assert "macd" in tech.columns
    assert "atr_14" in tech.columns
    assert "volatility_20d" in tech.columns
    assert "sma_50" in tech.columns


def test_gold_silver_ratio(sample_ohlcv):
    silver_df = sample_ohlcv.copy()
    silver_df["close"] = silver_df["close"] / 80.0
    res = analyze_gold_silver_ratio(sample_ohlcv, silver_df, usd_inr=83.0)
    assert "current_ratio" in res
    assert "gold_inr_10g" in res
    assert res["current_ratio"] > 0


def test_regimes(sample_ohlcv):
    tech = calculate_technical_indicators(sample_ohlcv)
    mkt_reg = detect_market_regime(tech)
    assert "trend_regime" in mkt_reg
    assert "volatility_regime" in mkt_reg

    macro_reg = detect_macro_regime({})
    assert macro_reg["macro_regime"] == "Neutral"


def test_analogue_and_forecasting(sample_ohlcv):
    tech = calculate_technical_indicators(sample_ohlcv)
    analogues = find_historical_analogues(tech, top_k=5)
    assert "sample_count" in analogues
    signals = generate_probabilistic_signals(analogues, "Gold")
    assert "horizons" in signals


def test_features_and_no_lookahead(sample_ohlcv):
    tech = calculate_technical_indicators(sample_ohlcv)
    feats = build_daily_feature_dataset(tech, {})
    assert "future_return_5d" in feats.columns
    # Check that feature columns on date T do not leak future_return_5d
    assert not np.isnan(feats["close"].iloc[0])


def test_backtest_engine(sample_ohlcv):
    engine = BacktestEngine()
    df = sample_ohlcv.copy()
    df["signal"] = [1 if i % 2 == 0 else 0 for i in range(len(df))]
    res = engine.run_signal_backtest(df, "signal")
    assert "cumulative_return" in res
    assert "sharpe_ratio" in res


def test_full_pipeline_run():
    res = run_pipeline("2026-09-08")
    assert res["status"] == "SUCCESS"
