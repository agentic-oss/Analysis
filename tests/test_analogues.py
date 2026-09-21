import pytest
from src.patterns.analogues import HistoricalAnalogueEngine
from src.forecasting.probabilistic import ProbabilisticForwardEngine
from src.ingestion.provider import MockMarketDataProvider
from src.indicators.technical import TechnicalIndicators


def test_analogues_and_forecasting():
    provider = MockMarketDataProvider()
    raw = provider.fetch_historical_ohlcv("GOLD", "GC=F", "2024-01-01", "2026-01-15")
    df = TechnicalIndicators.calculate_all(raw)

    analogues, stats = HistoricalAnalogueEngine.find_analogues(df, top_n=10)
    assert len(analogues) <= 10
    assert "sample_count" in stats

    fwd = ProbabilisticForwardEngine.generate_forward_signals(df, stats)
    assert "horizons" in fwd
    assert "disclaimer" in fwd
    assert "probabilistic research signals" in fwd["disclaimer"]
    assert "5d" in fwd["horizons"]
    assert "probability_positive_pct" in fwd["horizons"]["5d"]
