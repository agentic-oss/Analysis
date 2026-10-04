import os
import json
import pytest
import pandas as pd
from src.ingestion.provider import MockDataProvider, YFinanceDataProvider
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager, enrich_with_inr_conversions

def test_mock_data_provider():
    provider = MockDataProvider()
    inst = {"symbol": "GOLD", "ticker": "GC=F", "market": "futures", "currency": "USD", "unit": "oz"}
    df = provider.fetch_series("GC=F", "2025-01-01", "2025-01-10", inst)
    assert not df.empty
    assert "close" in df.columns
    assert df["symbol"].iloc[0] == "GOLD"

def test_data_validator():
    validator = DataValidator(jump_threshold_pct=10.0)
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "open": [100.0, 105.0, -10.0],
        "high": [110.0, 108.0, 50.0],
        "low": [95.0, 102.0, 10.0],
        "close": [105.0, 104.0, 20.0],
        "volume": [1000, 1200, 500]
    })
    validated_df, report = validator.validate_dataset("TEST", df)
    assert report["invalid_count"] == 1
    assert "validation_status" in validated_df.columns
    assert validated_df.iloc[2]["validation_status"] == "invalid"

def test_storage_manager_and_inr_enrichment(tmp_path):
    storage = StorageManager(base_dir=str(tmp_path))
    df_gold = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "open": [2000.0, 2010.0],
        "high": [2020.0, 2030.0],
        "low": [1990.0, 2000.0],
        "close": [2010.0, 2025.0],
        "currency": "USD"
    })
    df_usdinr = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02"],
        "close": [83.0, 83.5]
    })

    # Oz to 10g factor: 3.11034768 / 31.1034768 = 0.1
    enriched = enrich_with_inr_conversions(df_gold, df_usdinr, 3.11034768 / 31.1034768)
    assert "close_inr" in enriched.columns
    assert enriched.iloc[0]["close_inr"] == pytest.approx(2010.0 * 83.0 * (3.11034768 / 31.1034768))

    raw_path = storage.save_raw("2025-01-02", "GOLD", enriched)
    assert os.path.exists(raw_path)

    proc_path = storage.save_processed("metals", "GOLD", enriched)
    assert os.path.exists(proc_path)

    loaded = storage.load_processed("metals", "GOLD")
    assert loaded is not None
    assert len(loaded) == 2
