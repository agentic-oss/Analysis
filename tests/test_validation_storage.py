import os
import shutil
import pandas as pd
import pytest
from src.validation.validator import DataValidator
from src.storage.manager import StorageManager

def test_data_validator():
    validator = DataValidator(jump_threshold_pct=10.0)

    # Valid dataframe
    data = {
        "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "open": [100.0, 102.0, 103.0],
        "high": [105.0, 106.0, 107.0],
        "low": [98.0, 100.0, 101.0],
        "close": [102.0, 104.0, 105.0]
    }
    df = pd.DataFrame(data)
    val_df, metrics = validator.validate_dataset("GOLD", df)

    assert metrics["total_records"] == 3
    assert metrics["valid_records"] == 3
    assert metrics["invalid_records"] == 0

    # DataFrame with invalid OHLC
    data_bad = {
        "date": ["2026-01-01"],
        "open": [100.0],
        "high": [90.0],  # high lower than open
        "low": [98.0],
        "close": [102.0]
    }
    df_bad = pd.DataFrame(data_bad)
    val_df_bad, metrics_bad = validator.validate_dataset("GOLD", df_bad)
    assert metrics_bad["invalid_records"] == 1

def test_storage_manager(tmp_path):
    storage = StorageManager(base_data_dir=str(tmp_path))
    df = pd.DataFrame({
        "date": ["2026-01-01"],
        "close": [2000.0]
    })

    storage.save_processed(df, "GOLD", category="metals")
    loaded = storage.load_processed("GOLD", category="metals")
    assert not loaded.empty
    assert loaded["close"].iloc[0] == 2000.0
