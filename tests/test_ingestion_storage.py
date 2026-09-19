import pytest
import pandas as pd
import os
from src.ingestion.provider import MockMarketDataProvider, YFinanceProvider
from src.validation.validator import DataValidator
from src.storage.storage_manager import StorageManager

def test_mock_provider():
    df_sample = pd.DataFrame([
        {"date": "2026-03-30", "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "volume": 1000},
        {"date": "2026-03-31", "open": 2005.0, "high": 2020.0, "low": 2000.0, "close": 2015.0, "volume": 1200}
    ])
    provider = MockMarketDataProvider({"GOLD_USD": df_sample})

    fetched = provider.fetch_historical_prices("GOLD_USD", "GC=F", "2026-03-30", "2026-03-31")
    assert len(fetched) == 2
    assert "close" in fetched.columns

    quote = provider.fetch_latest_quote("GOLD_USD", "GC=F")
    assert quote["close"] == 2015.0
    assert quote["symbol"] == "GOLD_USD"

def test_data_validator():
    df_valid = pd.DataFrame([
        {"date": "2026-03-30", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0},
        {"date": "2026-03-31", "open": 102.0, "high": 106.0, "low": 101.0, "close": 104.0}
    ])
    val_df, summary = DataValidator.validate_df(df_valid, "GOLD_USD")
    assert summary["valid_count"] == 2
    assert summary["invalid_count"] == 0

    df_invalid = pd.DataFrame([
        {"date": "2026-03-30", "open": 100.0, "high": 90.0, "low": 95.0, "close": 102.0}, # high < low
        {"date": "2026-03-31", "open": 102.0, "high": 106.0, "low": 101.0, "close": -10.0} # negative close
    ])
    val_df_inv, summary_inv = DataValidator.validate_df(df_invalid, "GOLD_USD")
    assert summary_inv["invalid_count"] == 2

def test_storage_manager(tmp_path):
    storage = StorageManager(base_data_dir=str(tmp_path))
    df = pd.DataFrame([{"date": "2026-03-31", "close": 100.0}])
    storage.save_df(df, "processed/test_df")

    loaded = storage.load_df("processed/test_df")
    assert len(loaded) == 1
    assert loaded.iloc[0]["close"] == 100.0

    test_json = {"status": "ok", "count": 1}
    storage.save_json(test_json, "quality/test_quality")
    loaded_json = storage.load_json("quality/test_quality")
    assert loaded_json["status"] == "ok"
