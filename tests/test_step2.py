import pytest
import pandas as pd
from src.validation.validator import DataValidator
from src.storage.repository import DataRepository

def test_data_validator():
    validator = DataValidator(max_daily_jump_pct=0.10)

    # Valid data
    df_valid = pd.DataFrame({
        "date": ["2026-03-20", "2026-03-21"],
        "open": [100.0, 101.0],
        "high": [105.0, 106.0],
        "low": [99.0, 100.0],
        "close": [102.0, 104.0],
        "volume": [1000, 1200]
    })

    validated, summary = validator.validate_dataframe(df_valid, "GOLD")
    assert summary["invalid_records"] == 0
    assert summary["valid_records"] == 2

    # Invalid data (High < Low)
    df_invalid = pd.DataFrame({
        "date": ["2026-03-20"],
        "open": [100.0],
        "high": [90.0],
        "low": [99.0],
        "close": [95.0],
        "volume": [1000]
    })

    _, summary_inv = validator.validate_dataframe(df_invalid, "GOLD")
    assert summary_inv["invalid_records"] == 1

def test_repository(tmp_path):
    repo = DataRepository(base_dir=str(tmp_path))
    df = pd.DataFrame({"date": ["2026-03-20"], "close": [100.0]})
    repo.save_processed_data(df, "GOLD")

    loaded = repo.load_processed_data("GOLD")
    assert len(loaded) == 1
    assert loaded["close"].iloc[0] == 100.0
