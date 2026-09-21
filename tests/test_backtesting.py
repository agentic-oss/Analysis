import pytest
from src.backtesting.engine import BacktestEngine
from src.scoring.composite import CompositeScoreCalculator
from src.scoring.alerts import AlertEngine
from src.ingestion.provider import MockMarketDataProvider
from src.indicators.technical import TechnicalIndicators


def test_backtesting_and_scoring():
    provider = MockMarketDataProvider()
    df = TechnicalIndicators.calculate_all(provider.fetch_historical_ohlcv("GOLD", "GC=F", "2024-01-01", "2026-01-15"))
    df["composite_score"] = 65.0

    backtester = BacktestEngine()
    res = backtester.run_signal_backtest(df)
    assert res["status"] == "success"
    assert "cumulative_return_pct" in res
    assert "sharpe_ratio" in res

    scores = CompositeScoreCalculator.calculate_scores(df)
    assert 0 <= scores["composite_score"] <= 100

    alert_eng = AlertEngine()
    alerts = alert_eng.detect_alerts("2026-01-15", {"gold": {"daily_move_pct": 4.0, "technical_indicators": {"rsi_14": 75.0}}})
    assert alerts["alert_count"] >= 1
