import pytest
import pandas as pd
from src.validation.validator import DataValidator

def test_data_validator_valid_and_invalid():
    validator = DataValidator(jump_threshold_pct=15.0)

    # Valid data
    valid_data = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=5),
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, 104, 105, 106],
    })

    v_df, report = validator.validate_dataset(valid_data, "TEST_VALID")
    assert report["valid_records"] == 5
    assert report["invalid_records"] == 0

    # Invalid data (High < Low)
    invalid_data = pd.DataFrame({
        "Date": pd.date_range("2026-01-01", periods=2),
        "Open": [100, 100],
        "High": [90, 105], # Invalid High < Low
        "Low": [95, 95],
        "Close": [98, 100],
    })

    inv_df, inv_report = validator.validate_dataset(invalid_data, "TEST_INVALID")
    assert inv_report["invalid_records"] == 1

def test_daily_quality_report_aggregation():
    validator = DataValidator()
    r1 = {"valid_records": 10, "suspicious_records": 1, "invalid_records": 0, "warnings": ["w1"], "errors": []}
    r2 = {"valid_records": 10, "suspicious_records": 0, "invalid_records": 1, "warnings": [], "errors": ["e1"]}

    daily_report = validator.create_daily_quality_report("2026-01-01", [r1, r2])
    assert daily_report["total_valid_records"] == 20
    assert daily_report["total_suspicious_records"] == 1
    assert daily_report["total_invalid_records"] == 1
    assert len(daily_report["warnings"]) == 1
    assert len(daily_report["errors"]) == 1
