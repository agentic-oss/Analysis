import pytest
import pandas as pd
from src.ingestion.synthetic_provider import SyntheticProvider
from src.storage.storage_manager import StorageManager

def test_synthetic_provider_ingestion():
    provider = SyntheticProvider(seed=123)
    df = provider.get_spot_prices("GOLD", "2024-01-01", "2024-01-20")
    assert not df.empty
    assert "Close" in df.columns
    assert "High" in df.columns
    assert "Low" in df.columns
    assert len(df) > 5

def test_storage_manager_dataframe_roundtrip(tmp_path):
    sm = StorageManager(base_dir=str(tmp_path))
    df = pd.DataFrame({"Close": [100.0, 102.0, 105.0]}, index=pd.date_range("2024-01-01", periods=3))

    saved_path = sm.save_dataframe(df, "processed/test.parquet", format="parquet")
    assert saved_path.endswith(".parquet")

    loaded_df = sm.load_dataframe("processed/test.parquet", format="parquet")
    assert len(loaded_df) == 3
    assert loaded_df["Close"].iloc[0] == 100.0
