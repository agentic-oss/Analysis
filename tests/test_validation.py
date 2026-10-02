import pytest
import pandas as pd
import numpy as np
from src.validation.validator import DataValidator

def test_data_validator_valid():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-02", "2026-03-03"],
        "symbol": ["GOLD", "GOLD", "GOLD"],
        "open": [2500.0, 2510.0, 2520.0],
        "high": [2520.0, 2530.0, 2540.0],
        "low": [2490.0, 2500.0, 2510.0],
        "close": [2515.0, 2525.0, 2535.0]
    })
    v_df, report = validator.validate_dataframe(df)
    assert report["summary"]["valid_count"] == 3
    assert report["summary"]["invalid_count"] == 0

def test_data_validator_invalid_ohlc():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2026-03-01"],
        "symbol": ["GOLD"],
        "open": [2500.0],
        "high": [2400.0], # high < open -> invalid
        "low": [2490.0],
        "close": [2515.0]
    })
    v_df, report = validator.validate_dataframe(df)
    assert report["summary"]["invalid_count"] == 1
