import pytest
import pandas as pd
from src.ingestion.provider import MockMarketDataProvider
from src.ingestion.engine import DataIngestionEngine

def test_mock_market_data_provider():
    provider = MockMarketDataProvider()
    meta = {"market": "spot", "currency": "USD", "unit": "oz"}
    df = provider.get_historical_prices("GOLD", "GC=F", "2024-01-01", "2024-01-10", meta)

    assert not df.empty
    assert "date" in df.columns
    assert "close" in df.columns
    assert "symbol" in df.columns
    assert df["symbol"].iloc[0] == "GOLD"
    assert len(df) > 0

def test_data_ingestion_engine_inr_conversion():
    provider = MockMarketDataProvider()
    engine = DataIngestionEngine(provider=provider)
    data = engine.fetch_all_instruments("2024-01-01", "2024-01-05")

    assert "GOLD" in data
    assert "SILVER" in data
    assert "USDINR" in data

    converted = engine.calculate_inr_converted_prices(data["GOLD"], data["SILVER"], data["USDINR"])
    assert "GOLD_DERIVED_INR" in converted
    assert "converted_inr_10g" in converted["GOLD_DERIVED_INR"].columns
    assert "SILVER_DERIVED_INR" in converted
    assert "converted_inr_kg" in converted["SILVER_DERIVED_INR"].columns
