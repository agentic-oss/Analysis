import pytest
import pandas as pd
from src.validation.validator import DataValidator

def test_data_validator_rules():
    validator = DataValidator(max_jump_pct=10.0)

    df = pd.DataFrame({
        "Open": [100.0, 102.0, 105.0],
        "High": [103.0, 104.0, 120.0],
        "Low": [99.0, 101.0, 104.0],
        "Close": [102.0, 103.0, 118.0] # 118 vs 103 is > 10% jump
    }, index=pd.date_range("2024-01-01", periods=3))

    df_val, summary = validator.validate_series("GOLD", df)

    assert summary["total_records"] == 3
    assert summary["suspicious_records"] == 1
    assert "quality_status" in df_val.columns
    assert df_val["quality_status"].iloc[-1] == "suspicious"
