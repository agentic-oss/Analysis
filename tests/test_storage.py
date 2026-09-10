import pytest
import pandas as pd
import json
from src.storage.storage import DataStorage

def test_data_storage_raw_and_processed(tmp_path):
    storage = DataStorage(base_dir=str(tmp_path))
    df = pd.DataFrame({"date": ["2024-01-01"], "close": [2000.0]})

    raw_path = storage.save_raw_data(df, "2024-01-01", "GOLD")
    assert (tmp_path / "raw" / "2024" / "01" / "2024-01-01" / "GOLD.parquet").exists()

    proc_path = storage.save_processed_data(df, "metals", "GOLD")
    loaded = storage.load_processed_data("metals", "GOLD")
    assert len(loaded) == 1
    assert loaded["close"].iloc[0] == 2000.0

def test_data_storage_analysis_json(tmp_path):
    storage = DataStorage(base_dir=str(tmp_path))
    analysis = {"date": "2024-01-01", "gold": {"price": 2000.0}}
    json_path = storage.save_daily_analysis_json(analysis, "2024-01-01")
    assert (tmp_path / "analysis" / "daily" / "2024-01-01.json").exists()
    with open(json_path, "r") as f:
        data = json.load(f)
        assert data["gold"]["price"] == 2000.0
