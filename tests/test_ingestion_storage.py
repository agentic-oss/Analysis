import pytest
import pandas as pd
from src.ingestion.provider import SyntheticMarketDataProvider, convert_usd_to_inr_metrics
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager

def test_synthetic_provider():
    provider = SyntheticMarketDataProvider(seed=42)
    df = provider.fetch_historical_series("GC=F", "2026-01-01", "2026-01-10")
    assert not df.empty
    assert "close" in df.columns
    assert len(df) > 0

def test_currency_conversion():
    res = convert_usd_to_inr_metrics(gold_usd_oz=2000.0, silver_usd_oz=25.0, usdinr_rate=83.0)
    assert res["gold_inr_per_10g"] > 0
    assert res["silver_inr_per_kg"] > 0
    assert res["usdinr_rate"] == 83.0

def test_validator(tmp_path):
    validator = DataValidator(quality_dir=str(tmp_path))
    provider = SyntheticMarketDataProvider(seed=42)
    df = provider.fetch_historical_series("GC=F", "2026-01-01", "2026-01-10")
    report = validator.validate_dataset(df, "gold_test", "2026-01-10")
    assert report["records"] == len(df)
    assert report["valid_records"] > 0

def test_storage(tmp_path):
    storage = StorageManager(base_data_dir=str(tmp_path))
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    paths = storage.save_dataframe(df, "processed/test", "sample")
    assert "parquet" in paths
    df_loaded = storage.load_dataframe("processed/test", "sample.parquet")
    assert len(df_loaded) == 2
