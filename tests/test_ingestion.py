"""
Unit test for data ingestion, Indian price converter, data validator, and storage layer.
"""

import os
import pytest
import pandas as pd
import numpy as np
from src.ingestion.provider import IndianPriceConverter
from src.validation.validator import DataValidator
from src.storage.storage import DataStorage


def test_indian_price_converter():
    usd_price = 2000.0  # $2000 per oz
    usdinr = 83.0      # 83 INR per USD

    # 2000 / 31.1034768 * 83 * 10 = ~53370.239
    inr_10g = IndianPriceConverter.convert_usd_oz_to_inr_10g(usd_price, usdinr)
    assert inr_10g is not None
    assert round(inr_10g, 1) == 53370.2

    inr_kg = IndianPriceConverter.convert_usd_oz_to_inr_kg(usd_price, usdinr)
    assert round(inr_kg, 1) == 5337023.9


def test_data_validator():
    validator = DataValidator(jump_threshold_pct=0.20)
    data = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "open": [100.0, 102.0, 105.0, 200.0],
        "high": [105.0, 104.0, 106.0, 210.0],
        "low": [98.0, 100.0, 103.0, 190.0],
        "close": [101.0, 103.0, 104.0, 200.0]  # 200 is >20% jump from 104
    })

    df, audit = validator.validate_series(data, "TEST")
    assert audit["total_records"] == 4
    assert audit["suspicious_records"] == 1
    assert df.loc[3, "quality_status"] == "suspicious"


def test_storage(tmp_path):
    storage = DataStorage(base_dir=str(tmp_path))
    df = pd.DataFrame({"date": ["2026-01-01"], "close": [100.0]})
    storage.save_dataframe(df, "processed/test_df", formats=["parquet", "csv"])

    loaded_df = storage.load_dataframe("processed/test_df")
    assert len(loaded_df) == 1
    assert loaded_df["close"].iloc[0] == 100.0

    json_data = {"date": "2026-01-01", "status": "ok"}
    storage.save_json(json_data, "quality/test_audit")
    assert os.path.exists(os.path.join(tmp_path, "quality/test_audit.json"))
