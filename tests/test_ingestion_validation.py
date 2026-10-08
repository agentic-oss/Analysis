import pytest
import pandas as pd
from src.ingestion.yfinance_provider import YFinanceProvider
from src.validation.validator import DataValidator

def test_yfinance_provider_structure():
    provider = YFinanceProvider(max_retries=1)
    inst_info = {"symbol": "GOLD", "ticker": "GC=F", "currency": "USD", "unit": "oz"}
    df = provider.get_historical_prices(inst_info, start_date="2026-03-01", end_date="2026-03-10")
    if not df.empty:
        assert "date" in df.columns
        assert "close" in df.columns
        assert df["symbol"].iloc[0] == "GOLD"

def test_data_validator_rules():
    validator = DataValidator()
    df = pd.DataFrame({
        "date": ["2026-03-20"],
        "open": [100.0],
        "high": [105.0],
        "low": [95.0],
        "close": [102.0],
        "volume": [500]
    })
    val_df, summary = validator.validate_dataframe(df, "GOLD")
    assert summary["valid_records"] == 1
