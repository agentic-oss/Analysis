"""
Unit tests for data ingestion, validation, and storage core.
"""

import os
import pytest
import pandas as pd
from src.ingestion.provider import SyntheticProvider
from src.ingestion.ingest import DataIngestionService
from src.validation.validator import DataValidator
from src.storage.storage import StorageManager


def test_synthetic_provider():
    provider = SyntheticProvider(seed=42)
    df = provider.fetch_historical_data("GOLD", "GC=F", "2026-01-01", "2026-03-31")
    assert not df.empty
    assert "date" in df.columns
    assert "close" in df.columns
    assert df["symbol"].iloc[0] == "GOLD"


def test_data_validator():
    validator = DataValidator()
    df = pd.DataFrame([
        {"date": "2026-01-01", "open": 2000.0, "high": 2010.0, "low": 1990.0, "close": 2005.0, "volume": 100},
        {"date": "2026-01-02", "open": 2005.0, "high": 2000.0, "low": 1990.0, "close": 2005.0, "volume": 100}, # invalid high < open
    ])
    val_df, summary = validator.validate_dataframe(df, "GOLD")
    assert summary["total_records"] == 2
    assert summary["invalid_records"] == 1
    assert val_df["classification"].iloc[0] == "valid"
    assert val_df["classification"].iloc[1] == "invalid"


def test_storage_manager(tmp_path):
    storage = StorageManager(base_dir=str(tmp_path))
    df = pd.DataFrame([{"date": "2026-01-01", "close": 2000.0}])
    save_path = os.path.join(tmp_path, "test_file.csv")
    storage.save_dataframe(df, save_path)

    loaded_df = storage.load_dataframe(save_path)
    assert loaded_df is not None
    assert loaded_df["close"].iloc[0] == 2000.0
