import pytest
import pandas as pd
import numpy as np
from src.validation.validator import DataValidator


def test_validator_detects_ohlc_violations():
    validator = DataValidator()
    df = pd.DataFrame([
        {"date": "2026-01-01", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000},
        {"date": "2026-01-02", "open": 100.0, "high": 90.0, "low": 95.0, "close": 102.0, "volume": 1000}  # Invalid High < Low
    ])

    val_df, report = validator.validate_dataframe("TEST", df)
    assert report["invalid"] >= 1
    assert "Invalid OHLC geometry" in val_df.iloc[1]["quality_issues"]


def test_validator_detects_duplicates():
    validator = DataValidator()
    df = pd.DataFrame([
        {"date": "2026-01-01", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000},
        {"date": "2026-01-01", "open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000}
    ])
    val_df, report = validator.validate_dataframe("TEST", df)
    assert report["suspicious"] >= 1
