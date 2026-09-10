import pytest
import pandas as pd
from src.validation.validator import DataValidator

def test_data_validator_valid_dataset():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [99.0, 101.0],
        "close": [102.0, 104.0]
    })
    res = validator.validate_dataset("TEST", df)
    assert res["classification"] == "valid"
    assert len(res["errors"]) == 0

def test_data_validator_invalid_ohlc():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "open": [100.0],
        "high": [90.0], # high < open
        "low": [99.0],
        "close": [102.0]
    })
    res = validator.validate_dataset("TEST", df)
    assert res["classification"] == "invalid"
    assert len(res["errors"]) > 0

def test_generate_daily_quality_report(tmp_path):
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "open": [100.0],
        "high": [105.0],
        "low": [99.0],
        "close": [102.0]
    })
    report = validator.generate_daily_quality_report("2024-01-01", {"GOLD": df}, output_dir=str(tmp_path))
    assert report["overall_status"] == "valid"
    assert (tmp_path / "2024-01-01.json").exists()
