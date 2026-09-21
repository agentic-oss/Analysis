import pytest
import pandas as pd
from src.ratios.relative_value import RelativeValueAnalyzer
from src.ingestion.provider import MockMarketDataProvider
from src.indicators.technical import TechnicalIndicators


def test_gold_silver_ratio():
    provider = MockMarketDataProvider()
    gold = TechnicalIndicators.calculate_all(provider.fetch_historical_ohlcv("GOLD", "GC=F", "2025-01-01", "2026-01-15"))
    silver = TechnicalIndicators.calculate_all(provider.fetch_historical_ohlcv("SILVER", "SI=F", "2025-01-01", "2026-01-15"))

    ratio_df = RelativeValueAnalyzer.calculate_gold_silver_ratio(gold, silver)
    assert not ratio_df.empty
    assert "ratio" in ratio_df.columns
    assert "ratio_zscore_252" in ratio_df.columns

    analysis = RelativeValueAnalyzer.analyze_ratio_extremes_and_forward_outcomes(ratio_df)
    assert "latest_ratio" in analysis
    assert "latest_zscore" in analysis
    assert analysis["latest_ratio"] > 0
