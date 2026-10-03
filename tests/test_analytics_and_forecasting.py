import pytest
import pandas as pd
import numpy as np
from src.ratios.gold_silver_ratio import GoldSilverRatioAnalyzer
from src.regimes.macro_regimes import MacroRegimeModel
from src.regimes.market_regimes import AssetMarketRegimeModel
from src.patterns.historical_analogues import HistoricalAnalogueEngine
from src.scoring.score_calculator import ScoreCalculator
from src.scoring.confidence import PredictionConfidenceFramework
from src.forecasting.forecast_engine import ProbabilisticForecastEngine
from src.backtesting.backtester import SignalBacktester

def test_gold_silver_ratio_analyzer():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    gold_df = pd.DataFrame({"Date": dates, "Close": np.linspace(2000, 2200, 100)})
    silver_df = pd.DataFrame({"Date": dates, "Close": np.linspace(25, 30, 100)})

    ratio_df = GoldSilverRatioAnalyzer.calculate_ratio_series(gold_df, silver_df)
    assert "ratio" in ratio_df.columns
    assert ratio_df["ratio"].iloc[0] == pytest.approx(80.0, abs=1e-4)

    summary = GoldSilverRatioAnalyzer.analyze_latest_ratio(ratio_df)
    assert "current_ratio" in summary
    assert "regime" in summary

def test_macro_and_market_regimes():
    macro_summary = {
        "dxy": {"pct_change": -0.5},
        "vix": {"latest": 25.0, "pct_change": 15.0},
        "sp500": {"pct_change": -1.5},
        "oil": {"pct_change": 0.0},
        "us10y": {"abs_change": 0.02},
    }

    regime = MacroRegimeModel.classify_macro_regime(macro_summary)
    assert regime["primary_regime"] in ["Risk-Off", "Inflationary", "Stagflationary", "Neutral", "Risk-On"]

def test_historical_analogue_engine():
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    np.random.seed(42)
    close = 2000.0 + np.cumsum(np.random.normal(0, 10, size=100))
    df = pd.DataFrame({
        "Date": dates,
        "Open": close,
        "High": close + 5,
        "Low": close - 5,
        "Close": close,
    })

    engine = HistoricalAnalogueEngine(top_k=5)
    analogues = engine.find_analogues(df, "2025-04-01")
    assert "forward_statistics" in analogues
    assert "5d" in analogues["forward_statistics"]

def test_probabilistic_forecasting_disclaimer():
    engine = ProbabilisticForecastEngine()
    forecasts = engine.generate_forecasts(None, {})
    assert "disclaimer" in forecasts
    assert "probabilistic research signals" in forecasts["disclaimer"].lower()

def test_backtester_execution():
    dates = pd.date_range("2025-01-01", periods=50, freq="B")
    df = pd.DataFrame({
        "Date": dates,
        "Close": np.linspace(100, 110, 50),
        "signal": [70]*25 + [30]*25,
    })

    bt = SignalBacktester()
    res = bt.run_backtest(df, "signal")
    assert "cumulative_return_pct" in res
    assert "final_portfolio_value" in res
