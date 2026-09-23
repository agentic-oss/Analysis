import pytest
import os
import pandas as pd
from src.ingestion.provider import SyntheticProvider, convert_gold_usd_oz_to_inr_10g, convert_silver_usd_oz_to_inr_kg
from src.validation.validator import DataValidator, save_quality_report
from src.storage.manager import StorageManager

def test_synthetic_provider():
    provider = SyntheticProvider(seed=42)
    df = provider.get_historical_prices("GOLD", "2025-01-01", "2025-01-31")
    assert not df.empty
    assert "close" in df.columns
    assert (df["instrument"] == "GOLD").all()

def test_conversions():
    # Gold: USD 2000/oz, USDINR = 80.0
    # 2000 / 31.1034768 * 80.0 * 10 = 51441.1945...
    gold_inr = convert_gold_usd_oz_to_inr_10g(2000.0, 80.0)
    assert round(gold_inr, 2) == 51441.19

    # Silver: USD 25/oz, USDINR = 80.0
    # 25 / 0.0311034768 * 80.0 = 64301.493...
    silver_inr = convert_silver_usd_oz_to_inr_kg(25.0, 80.0)
    assert round(silver_inr, 2) == 64301.49

def test_data_validator():
    validator = DataValidator(jump_threshold_pct=20.0)
    data = pd.DataFrame([
        {"timestamp": "2025-01-01", "open": 100, "high": 105, "low": 95, "close": 102},
        {"timestamp": "2025-01-02", "open": 102, "high": 108, "low": 100, "close": 105},
        {"timestamp": "2025-01-03", "open": 105, "high": 100, "low": 90, "close": -10} # Invalid high < low, close < 0
    ])
    validated_df, report = validator.validate_dataset(data, "GOLD", "2025-01-03")
    assert report["total_records"] == 3
    assert report["valid_records"] == 2
    assert report["invalid_records"] == 1

def test_storage_manager(tmp_path):
    storage = StorageManager(base_data_dir=str(tmp_path))
    df = pd.DataFrame([
        {"timestamp": "2025-01-01", "close": 2000.0}
    ])
    p_path = storage.save_raw_observation(df, "GOLD", "2025-01-01")
    assert os.path.exists(p_path)

    storage.save_processed_data(df, "metals", "GOLD")
    loaded = storage.load_processed_data("metals", "GOLD")
    assert len(loaded) == 1
    assert loaded["close"].iloc[0] == 2000.0
