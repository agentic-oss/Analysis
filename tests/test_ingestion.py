import pytest
import pandas as pd
from src.ingestion.provider import MockMarketDataProvider
from src.ingestion.collector import DataCollector


def test_mock_provider_fetch_historical():
    provider = MockMarketDataProvider()
    df = provider.fetch_historical_ohlcv("GOLD", "GC=F", "2026-01-01", "2026-01-15")
    assert not df.empty
    assert "close" in df.columns
    assert "date" in df.columns
    assert len(df) > 0


def test_collector_conversions():
    collector = DataCollector(use_fallback=True)
    data = collector.collect_all("2026-01-01", "2026-01-10")
    assert "GOLD" in data
    assert "SILVER" in data
    assert "USDINR" in data

    gold_df = data["GOLD"]
    assert "price_inr" in gold_df.columns
    assert "usdinr_rate" in gold_df.columns
    assert gold_df["price_inr"].iloc[-1] > 0
